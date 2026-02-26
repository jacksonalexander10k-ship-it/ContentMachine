import io
import logging

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from bot.config import (
    AVAILABLE_VOICES,
    DEFAULT_AVATAR_URL,
    KLING_POLL_INTERVAL_SECONDS,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_TEXT_LENGTH,
)
from bot.keyboards import VOICE_CALLBACK_PREFIX, voice_selection_keyboard
from services.database import (
    clear_user_avatar,
    get_user_avatar,
    get_user_voice,
    set_user_avatar,
    set_user_voice,
)
from services.segmenter import segment_script
from services.talking_head import KlingAPIError, create_talking_head_video
from services.tts import text_to_speech
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
        "Same starting frame, same lighting, same look — every clip.\n\n"
        "Commands:\n"
        "  /voice — Choose your TTS voice\n"
        "  /avatar — View or reset your starting frame\n"
        "  /help — Show this message again"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    await start_command(update, context)


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /voice — show voice picker."""
    current = await get_user_voice(update.effective_user.id)
    keyboard = voice_selection_keyboard(current)
    await update.message.reply_text(
        f"Current voice: *{current.capitalize()}*\n\nPick a new voice:",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


async def avatar_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /avatar — show current avatar info."""
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
    """Handle /avatar_reset — clear custom avatar."""
    await clear_user_avatar(update.effective_user.id)
    await update.message.reply_text("Starting frame cleared.")


# ── Callback queries ─────────────────────────────────────────────────────────


async def voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice selection inline button."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith(VOICE_CALLBACK_PREFIX):
        return

    voice = data[len(VOICE_CALLBACK_PREFIX):]
    if voice not in AVAILABLE_VOICES:
        return

    await set_user_voice(update.effective_user.id, voice)
    keyboard = voice_selection_keyboard(voice)
    await query.edit_message_text(
        f"Voice set to *{voice.capitalize()}*!",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _get_avatar_bytes(user_id: int) -> bytes | None:
    """Get avatar bytes for a user — custom from DB, or downloaded from default URL."""
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
    """Core pipeline: segment script -> TTS per segment -> Kling video per segment -> send clips."""
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

    voice = await get_user_voice(user_id)

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

    # Step 2: Generate each clip sequentially
    generated = 0
    for i, segment in enumerate(segments, 1):
        label = f"[Clip {i}/{total}]"

        # TTS
        await _edit_status(status_msg, f"{label} Generating audio...")
        try:
            audio_bytes = await text_to_speech(segment, voice=voice)
        except Exception:
            logger.exception("TTS failed for clip %d", i)
            await update.message.reply_text(f"{label} Audio generation failed, skipping.")
            continue

        # Video generation with progress
        await _edit_status(status_msg, f"{label} Generating video...")
        await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

        async def progress_cb(status: str, attempt: int, max_attempts: int) -> None:
            elapsed = attempt * KLING_POLL_INTERVAL_SECONDS
            await _edit_status(status_msg, f"{label} Generating video... ({elapsed}s)")
            await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

        try:
            video_bytes = await create_talking_head_video(
                audio_bytes, avatar_bytes, progress_callback=progress_cb
            )
        except KlingAPIError as e:
            logger.exception("Kling API error for clip %d", i)
            await update.message.reply_text(f"{label} Failed: {e.user_message}")
            continue
        except TimeoutError:
            await update.message.reply_text(f"{label} Timed out, skipping.")
            continue
        except Exception:
            logger.exception("Video generation failed for clip %d", i)
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
