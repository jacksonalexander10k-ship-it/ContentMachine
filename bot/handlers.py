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
        "Send me a text message and I'll turn it into a talking-head video.\n\n"
        "You can also:\n"
        "  - Send a voice note and I'll transcribe + generate a video\n"
        "  - Send a photo to set your custom avatar\n\n"
        "Commands:\n"
        "  /voice — Choose your TTS voice\n"
        "  /avatar — View or reset your avatar\n"
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
            "You have a custom avatar set.\n"
            "Send a new photo to change it, or use /avatar_reset to clear it."
        )
    else:
        await update.message.reply_text(
            "You're using the default avatar.\n"
            "Send me a photo of a face to use as your custom avatar!"
        )


async def avatar_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /avatar_reset — clear custom avatar."""
    await clear_user_avatar(update.effective_user.id)
    await update.message.reply_text("Avatar reset to default.")


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


async def _generate_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    audio_bytes: bytes,
) -> None:
    """Shared logic for generating and sending a talking-head video."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    avatar_bytes = await _get_avatar_bytes(user_id)
    if not avatar_bytes:
        await update.message.reply_text(
            "You don't have an avatar set yet.\n"
            "Send me a photo of a face first, then try again!"
        )
        return

    status_msg = await update.message.reply_text("Starting video generation...")
    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)

    async def progress_cb(status: str, attempt: int, max_attempts: int) -> None:
        elapsed = attempt * KLING_POLL_INTERVAL_SECONDS
        try:
            await status_msg.edit_text(f"Generating video... ({elapsed}s elapsed)")
            await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)
        except Exception:
            pass  # Message edit can fail if text is unchanged or rate-limited

    try:
        video_bytes = await create_talking_head_video(
            audio_bytes, avatar_bytes, progress_callback=progress_cb
        )
        await status_msg.edit_text("Uploading video...")
        await update.message.reply_video(
            video=io.BytesIO(video_bytes),
            filename="talking_head.mp4",
            caption="Here's your talking-head video!",
        )
        await status_msg.delete()
    except KlingAPIError as e:
        logger.exception("Kling API error during video generation")
        await status_msg.edit_text(f"Error: {e.user_message}")
    except TimeoutError:
        await status_msg.edit_text(
            "Video generation timed out. The service may be overloaded — please try again later."
        )
    except Exception:
        logger.exception("Failed to generate video")
        await status_msg.edit_text(
            "Sorry, something went wrong generating your video. Please try again."
        )


# ── Message handlers ─────────────────────────────────────────────────────────


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — generate talking-head video from text."""
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text("Please send some text to generate a video.")
        return

    if len(text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"Text is too long ({len(text)} chars). Max is {MAX_TEXT_LENGTH}."
        )
        return

    voice = await get_user_voice(update.effective_user.id)

    try:
        audio_bytes = await text_to_speech(text, voice=voice)
    except Exception:
        logger.exception("TTS failed")
        await update.message.reply_text("Sorry, text-to-speech failed. Please try again.")
        return

    await _generate_video(update, context, audio_bytes)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio messages — transcribe, then generate video."""
    voice_msg = update.message.voice or update.message.audio
    if not voice_msg:
        return

    # Validate audio duration
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

    voice_name = await get_user_voice(update.effective_user.id)

    try:
        tts_audio = await text_to_speech(transcript, voice=voice_name)
    except Exception:
        logger.exception("TTS failed")
        await update.message.reply_text("Sorry, text-to-speech failed. Please try again.")
        return

    await _generate_video(update, context, tts_audio)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — download and store as custom avatar."""
    photos = update.message.photo
    if not photos:
        return

    photo = photos[-1]  # Largest resolution

    try:
        photo_bytes = await download_telegram_file(context.bot, photo.file_id)
        await set_user_avatar(update.effective_user.id, photo_bytes)
        await update.message.reply_text(
            "Avatar updated! I'll use this face for your next videos.\n"
            "Use /avatar_reset to go back to the default."
        )
    except Exception:
        logger.exception("Failed to set avatar from photo")
        await update.message.reply_text("Sorry, I couldn't process that photo.")
