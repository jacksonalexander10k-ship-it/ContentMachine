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
from services.segmenter import ScriptRejected, segment_script
from services.talking_head import VideoGenerationError, generate_clip
from services.transcribe import transcribe_audio
from utils.media import download_telegram_file

logger = logging.getLogger(__name__)

# ── User state constants ─────────────────────────────────────────────────────

STATE_IDLE = "idle"
STATE_CONFIRMING = "confirming_segments"
STATE_GENERATING = "generating"

# Keywords to confirm or cancel segment review
_CONFIRM_WORDS = {"ok", "okay", "yes", "y", "go", "generate", "confirm", "do it", "yep", "yeah", "sure", "send it", "lets go"}
_CANCEL_WORDS = {"no", "n", "cancel", "redo", "nah", "nope", "stop", "abort", "nevermind", "never mind"}


def _get_state(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("state", STATE_IDLE)


def _set_state(context: ContextTypes.DEFAULT_TYPE, state: str) -> None:
    context.user_data["state"] = state


def _clear_pending(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("pending_segments", None)
    context.user_data.pop("pending_script", None)


# ── Commands ──────────────────────────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    await update.message.reply_text(
        "Welcome to ContentMachine!\n\n"
        "How it works:\n"
        "1. Send me a photo — that becomes your starting frame\n"
        "2. Send me a script — I'll split it into segments\n"
        "3. Review the segments — reply OK to generate, or NO to cancel\n"
        "4. I generate a talking-head video clip for each segment\n\n"
        "Commands:\n"
        "  /status — Check what's happening right now\n"
        "  /stop — Cancel generation or pending segments\n"
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


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — cancel generation or pending confirmation."""
    state = _get_state(context)

    if state == STATE_GENERATING:
        context.user_data["cancel"] = True
        await update.message.reply_text(
            "Stopping. The current clip will finish, then generation stops."
        )
    elif state == STATE_CONFIRMING:
        _clear_pending(context)
        _set_state(context, STATE_IDLE)
        await update.message.reply_text(
            "Cancelled. Segments discarded.\n"
            "Send a new script whenever you're ready."
        )
    else:
        await update.message.reply_text("Nothing is running right now.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show current state."""
    state = _get_state(context)

    if state == STATE_GENERATING:
        clip_num = context.user_data.get("current_clip", "?")
        total = context.user_data.get("total_clips", "?")
        generated = context.user_data.get("clips_done", 0)
        await update.message.reply_text(
            f"Generating clips: working on clip {clip_num}/{total}\n"
            f"Completed so far: {generated}\n\n"
            "Use /stop to cancel after the current clip."
        )
    elif state == STATE_CONFIRMING:
        segments = context.user_data.get("pending_segments", [])
        await update.message.reply_text(
            f"Waiting for your confirmation on {len(segments)} segment(s).\n\n"
            "Reply OK to generate, or NO to cancel."
        )
    else:
        avatar = await get_user_avatar(update.effective_user.id)
        if avatar:
            await update.message.reply_text(
                "Idle — ready for a script.\n"
                "Starting frame: set\n\n"
                "Send me a script to get started."
            )
        else:
            await update.message.reply_text(
                "Idle — no starting frame set.\n\n"
                "Send me a photo of a face first, then send your script."
            )


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


def _format_segments_preview(segments: list[str]) -> str:
    """Format segments as a numbered list for the user to review."""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}. \"{seg}\"")
    return "\n".join(lines)


async def _segment_and_confirm(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    script: str,
) -> None:
    """Segment a script and ask the user to confirm before generating."""
    user_id = update.effective_user.id

    # Check avatar first — no point segmenting if there's no starting frame
    avatar_bytes = await _get_avatar_bytes(user_id)
    if not avatar_bytes:
        await update.message.reply_text(
            "You don't have a starting frame set yet.\n"
            "Send me a photo of a face first, then send your script!"
        )
        return

    # Segment the script
    status_msg = await update.message.reply_text("Analysing your script...")

    try:
        segments = await segment_script(script)
    except ScriptRejected as e:
        await _edit_status(status_msg,
            f"That doesn't look like a speakable script.\n\n"
            f"Reason: {e.reason}\n\n"
            f"Send me actual spoken text — a monologue, pitch, tutorial script, "
            f"narration, etc. — and I'll segment it into clips."
        )
        return
    except Exception:
        logger.exception("Script segmentation failed")
        await _edit_status(status_msg,
            "Failed to segment the script. Please try again."
        )
        return

    # Store segments and switch to confirmation state
    context.user_data["pending_segments"] = segments
    context.user_data["pending_script"] = script
    _set_state(context, STATE_CONFIRMING)

    total = len(segments)
    preview = _format_segments_preview(segments)
    await _edit_status(status_msg,
        f"Script split into {total} clip{'s' if total != 1 else ''}:\n\n"
        f"{preview}\n\n"
        f"Reply OK to generate all {total} clips, or NO to cancel."
    )


async def _run_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Generate video clips from the confirmed segments."""
    segments = context.user_data.get("pending_segments", [])
    if not segments:
        await update.message.reply_text("No segments to generate. Send a new script.")
        _set_state(context, STATE_IDLE)
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    avatar_bytes = await _get_avatar_bytes(user_id)
    if not avatar_bytes:
        await update.message.reply_text(
            "Starting frame is missing. Send a photo first."
        )
        _clear_pending(context)
        _set_state(context, STATE_IDLE)
        return

    total = len(segments)
    _set_state(context, STATE_GENERATING)
    context.user_data["generating"] = True
    context.user_data["cancel"] = False
    context.user_data["total_clips"] = total
    context.user_data["clips_done"] = 0

    status_msg = await update.message.reply_text(
        f"Starting generation of {total} clip{'s' if total != 1 else ''}..."
    )

    generated = 0

    try:
        for i, segment in enumerate(segments, 1):
            # Check for /stop cancellation
            if context.user_data.get("cancel"):
                await _edit_status(
                    status_msg,
                    f"Stopped. {generated}/{total} clips generated. (Cancelled by /stop)",
                )
                return

            context.user_data["current_clip"] = i
            label = f"[Clip {i}/{total}]"

            preview = segment[:50] + ("..." if len(segment) > 50 else "")
            await _edit_status(
                status_msg,
                f"{label} Generating video + audio...\n\"{preview}\""
            )
            await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

            async def progress_cb(status_text: str, _label=label, _preview=preview) -> None:
                await _edit_status(status_msg, f"{_label} {status_text}\n\"{_preview}\"")
                await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

            try:
                video_bytes = await generate_clip(
                    avatar_bytes, segment, progress_callback=progress_cb
                )
            except VideoGenerationError as e:
                logger.error("Video generation error for clip %d: %s", i, e.user_message)
                await context.bot.send_message(
                    chat_id,
                    f"{label} FAILED: {e.user_message}"
                )
                continue
            except Exception as e:
                logger.exception("Unexpected error for clip %d", i)
                await context.bot.send_message(
                    chat_id,
                    f"{label} FAILED: Unexpected error — {type(e).__name__}: {e}"
                )
                continue

            # Check for /stop again after generation completes — still send this clip
            await _edit_status(status_msg, f"{label} Uploading...")

            try:
                await context.bot.send_video(
                    chat_id,
                    video=io.BytesIO(video_bytes),
                    filename=f"clip_{i}.mp4",
                    caption=f"Clip {i}/{total}",
                )
            except Exception as e:
                logger.exception("Failed to send clip %d to Telegram", i)
                await context.bot.send_message(
                    chat_id,
                    f"{label} Video was generated but failed to upload to Telegram: {e}"
                )
                continue

            generated += 1
            context.user_data["clips_done"] = generated

            # If cancelled, stop after uploading the finished clip
            if context.user_data.get("cancel"):
                await _edit_status(
                    status_msg,
                    f"Stopped. {generated}/{total} clips generated. (Cancelled by /stop)",
                )
                return

        # Done
        if generated == total:
            await _edit_status(status_msg, f"Done — all {total} clips generated.")
        elif generated > 0:
            await _edit_status(
                status_msg,
                f"Done — {generated}/{total} clips generated. "
                f"{total - generated} failed (see error messages above)."
            )
        else:
            await _edit_status(
                status_msg,
                f"All {total} clips failed. Check the error messages above.\n"
                f"This is likely a fal.ai issue — try again in a few minutes."
            )
    finally:
        context.user_data["generating"] = False
        context.user_data["cancel"] = False
        context.user_data.pop("current_clip", None)
        context.user_data.pop("total_clips", None)
        _clear_pending(context)
        _set_state(context, STATE_IDLE)


# ── Message handlers ─────────────────────────────────────────────────────────


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — state-aware dispatcher."""
    text = update.message.text.strip()
    if not text:
        return

    state = _get_state(context)

    # ── State: waiting for confirmation on segments ──
    if state == STATE_CONFIRMING:
        lower = text.lower().strip()

        if lower in _CONFIRM_WORDS:
            await _run_generation(update, context)
            return

        if lower in _CANCEL_WORDS:
            _clear_pending(context)
            _set_state(context, STATE_IDLE)
            await update.message.reply_text(
                "Cancelled. Segments discarded.\n"
                "Send a new script whenever you're ready."
            )
            return

        # Anything else while confirming — remind them
        segments = context.user_data.get("pending_segments", [])
        await update.message.reply_text(
            f"You have {len(segments)} segment(s) waiting for confirmation.\n\n"
            f"Reply OK to generate, NO to cancel, or /stop to discard."
        )
        return

    # ── State: currently generating ──
    if state == STATE_GENERATING:
        await update.message.reply_text(
            "Generation is in progress. Use /status to check progress or /stop to cancel."
        )
        return

    # ── State: idle — treat as a new script ──

    if len(text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"Script is too long ({len(text)} chars). Max is {MAX_TEXT_LENGTH}."
        )
        return

    # Quick heuristic checks — catch obvious non-scripts without spending tokens
    words = text.split()
    alpha_chars = sum(1 for c in text if c.isalpha())

    if len(words) < 3:
        await update.message.reply_text(
            "That's too short to be a script.\n\n"
            "Send me a proper script (at least a few sentences) and I'll "
            "split it into segments for talking-head video clips."
        )
        return

    if len(text) > 5 and alpha_chars < len(text) * 0.4:
        await update.message.reply_text(
            "That doesn't look like speakable text.\n\n"
            "Send me actual spoken text — sentences, a monologue, a script — "
            "and I'll segment it into clips."
        )
        return

    await _segment_and_confirm(update, context, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio messages — transcribe, then go through confirmation flow."""
    state = _get_state(context)

    if state == STATE_GENERATING:
        await update.message.reply_text(
            "Generation is in progress. Use /status to check or /stop to cancel."
        )
        return

    if state == STATE_CONFIRMING:
        await update.message.reply_text(
            "You have segments waiting for confirmation.\n"
            "Reply OK to generate, NO to cancel, or /stop to discard."
        )
        return

    voice_msg = update.message.voice or update.message.audio
    if not voice_msg:
        return

    duration = getattr(voice_msg, "duration", None)
    if duration and duration > MAX_AUDIO_DURATION_SECONDS:
        await update.message.reply_text(
            f"Audio is too long ({duration}s). Maximum is {MAX_AUDIO_DURATION_SECONDS}s."
        )
        return

    await update.message.reply_text("Step 1/2: Transcribing your audio...")
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    try:
        audio_bytes = await download_telegram_file(context.bot, voice_msg.file_id)
        transcript = await transcribe_audio(audio_bytes)
    except Exception as e:
        logger.exception("Failed to transcribe audio")
        await update.message.reply_text(
            f"Transcription failed: {type(e).__name__}: {e}\n"
            f"Try again or send the script as text instead."
        )
        return

    if not transcript.strip():
        await update.message.reply_text("I couldn't detect any speech in that audio.")
        return

    await update.message.reply_text(
        f"Transcript:\n\"{transcript}\"\n\n"
        f"Step 2/2: Segmenting..."
    )
    await _segment_and_confirm(update, context, transcript)


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
            "Starting frame set!\n\n"
            "Now send me a script and I'll segment it into clips for you to review."
        )
    except Exception as e:
        logger.exception("Failed to set avatar from photo")
        await update.message.reply_text(
            f"Failed to process that photo: {type(e).__name__}: {e}"
        )
