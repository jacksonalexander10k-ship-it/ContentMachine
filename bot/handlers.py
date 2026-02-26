import io
import logging

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from bot.config import DEFAULT_AVATAR_URL, MAX_AUDIO_DURATION_SECONDS, MAX_TEXT_LENGTH
from services.database import (
    clear_user_avatar,
    get_user_avatar,
    set_user_avatar,
)
from services.segmenter import segment_script
from services.talking_head import VideoGenerationError, generate_clip
from services.transcribe import transcribe_audio
from utils.media import download_telegram_file

logger = logging.getLogger(__name__)


# ── Commands ──────────────────────────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    await update.message.reply_text(
        "Welcome to ContentMachine!\n\n"
        "How it works:\n"
        "1. Send me a photo — that's your starting frame\n"
        "2. Send me a script — I'll split it into segments and generate "
        "a talking-head video clip for each one\n\n"
        "Same starting frame, same lighting, same look — every clip.\n"
        "Kling 3.0 generates the speech audio natively.\n\n"
        "Commands:\n"
        "  /avatar — View or reset your starting frame\n"
        "  /help — Show this message again"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    await start_command(update, context)


async def avatar_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /avatar — show current starting frame info."""
    avatar = await get_user_avatar(update.effective_user.id)
    if avatar:
        await update.message.reply_text(
            "You have a starting frame set.\n"
            "Send a new photo to change it, or use /avatar_reset to clear it."
        )
    else:
        await update.message.reply_text(
            "No starting frame set yet.\n"
            "Send me a photo of a face to use as your starting frame!"
        )


async def avatar_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /avatar_reset — clear starting frame."""
    await clear_user_avatar(update.effective_user.id)
    await update.message.reply_text("Starting frame cleared.")


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _get_avatar_bytes(user_id: int) -> bytes | None:
    """Get starting frame bytes for a user — custom from DB, or downloaded from default URL."""
    avatar = await get_user_avatar(user_id)
    if avatar:
        return avatar

    if DEFAULT_AVATAR_URL:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(DEFAULT_AVATAR_URL)
                resp.raise_for_status()
                return resp.content
        except Exception:
            logger.warning("Failed to download default avatar from %s", DEFAULT_AVATAR_URL)

    return None


async def _edit_status(msg, text: str) -> None:
    """Edit a status message, ignoring errors (rate-limit, unchanged text)."""
    try:
        await msg.edit_text(text)
    except Exception:
        pass


async def _generate_clips(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    script: str,
) -> None:
    """Core pipeline: segment script -> Kling 3.0 video per segment (with native audio) -> send clips."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Get starting frame
    avatar_bytes = await _get_avatar_bytes(user_id)
    if not avatar_bytes:
        await update.message.reply_text(
            "You don't have a starting frame set yet.\n"
            "Send me a photo of a face first, then send your script!"
        )
        return

    # Step 1: Segment the script
    status_msg = await update.message.reply_text("Segmenting script...")
    try:
        segments = await segment_script(script)
    except Exception:
        logger.exception("Script segmentation failed")
        await _edit_status(status_msg, "Failed to segment the script. Please try again.")
        return

    total = len(segments)
    await _edit_status(
        status_msg,
        f"Script split into {total} clip{'s' if total != 1 else ''}. Starting generation..."
    )

    # Step 2: Generate each clip — same starting frame, Kling generates audio natively
    generated = 0
    for i, segment in enumerate(segments, 1):
        label = f"[Clip {i}/{total}]"

        await _edit_status(status_msg, f"{label} Generating video + audio...")
        await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

        async def progress_cb(status_text: str) -> None:
            await _edit_status(status_msg, f"{label} {status_text}")
            await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

        try:
            video_bytes = await generate_clip(
                avatar_bytes, segment, progress_callback=progress_cb
            )
        except VideoGenerationError as e:
            logger.exception("Video generation error for clip %d", i)
            await update.message.reply_text(f"{label} Failed: {e.user_message}")
            continue
        except Exception:
            logger.exception("Unexpected error for clip %d", i)
            await update.message.reply_text(f"{label} Failed, skipping.")
            continue

        # Send the clip
        await _edit_status(status_msg, f"{label} Uploading...")
        await update.message.reply_video(
            video=io.BytesIO(video_bytes),
            filename=f"clip_{i}.mp4",
            caption=f"Clip {i}/{total}",
        )
        generated += 1

    # Done
    if generated == total:
        await _edit_status(status_msg, f"Done — all {total} clips generated.")
    elif generated > 0:
        await _edit_status(status_msg, f"Done — {generated}/{total} clips generated.")
    else:
        await _edit_status(status_msg, "All clips failed. Please try again.")


# ── Message handlers ─────────────────────────────────────────────────────────


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — segment script and generate clips."""
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text("Please send a script to generate clips.")
        return

    if len(text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"Script is too long ({len(text)} chars). Max is {MAX_TEXT_LENGTH}."
        )
        return

    await _generate_clips(update, context, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio messages — transcribe, then segment and generate clips."""
    voice_msg = update.message.voice or update.message.audio
    if not voice_msg:
        return

    duration = getattr(voice_msg, "duration", None)
    if duration and duration > MAX_AUDIO_DURATION_SECONDS:
        await update.message.reply_text(
            f"Audio is too long ({duration}s). Maximum is {MAX_AUDIO_DURATION_SECONDS}s."
        )
        return

    await update.message.reply_text("Transcribing your audio...")
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    try:
        audio_bytes = await download_telegram_file(context.bot, voice_msg.file_id)
        transcript = await transcribe_audio(audio_bytes)
    except Exception:
        logger.exception("Failed to transcribe audio")
        await update.message.reply_text("Sorry, I couldn't transcribe that audio.")
        return

    if not transcript.strip():
        await update.message.reply_text("I couldn't detect any speech in that audio.")
        return

    await update.message.reply_text(f'Transcript: "{transcript}"')
    await _generate_clips(update, context, transcript)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — download and store as starting frame."""
    photos = update.message.photo
    if not photos:
        return

    photo = photos[-1]  # Largest resolution

    try:
        photo_bytes = await download_telegram_file(context.bot, photo.file_id)
        await set_user_avatar(update.effective_user.id, photo_bytes)
        await update.message.reply_text(
            "Starting frame set! I'll use this as the starting frame for all your clips.\n"
            "Now send me a script to generate videos."
        )
    except Exception:
        logger.exception("Failed to set avatar from photo")
        await update.message.reply_text("Sorry, I couldn't process that photo.")
