import io
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from bot.config import DEFAULT_TTS_VOICE, MAX_TEXT_LENGTH, AVAILABLE_VOICES
from bot.keyboards import voice_selection_keyboard, VOICE_CALLBACK_PREFIX
from services.tts import text_to_speech
from services.transcribe import transcribe_audio
from services.talking_head import create_talking_head_video
from utils.media import download_telegram_file

logger = logging.getLogger(__name__)

# Per-user state keys
USER_VOICE = "tts_voice"
USER_AVATAR = "avatar_url"


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
    current = context.user_data.get(USER_VOICE, DEFAULT_TTS_VOICE)
    keyboard = voice_selection_keyboard(current)
    await update.message.reply_text(
        f"Current voice: *{current.capitalize()}*\n\nPick a new voice:",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


async def avatar_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /avatar — show current avatar info."""
    avatar = context.user_data.get(USER_AVATAR)
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
    context.user_data.pop(USER_AVATAR, None)
    await update.message.reply_text("Avatar reset to default.")


# ── Callback queries ─────────────────────────────────────────────────────────


async def voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice selection inline button."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith(VOICE_CALLBACK_PREFIX):
        return

    voice = data[len(VOICE_CALLBACK_PREFIX) :]
    if voice not in AVAILABLE_VOICES:
        return

    context.user_data[USER_VOICE] = voice
    keyboard = voice_selection_keyboard(voice)
    await query.edit_message_text(
        f"Voice set to *{voice.capitalize()}*!",
        reply_markup=keyboard,
        parse_mode="Markdown",
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

    voice = context.user_data.get(USER_VOICE, DEFAULT_TTS_VOICE)
    avatar_url = context.user_data.get(USER_AVATAR)

    await update.message.reply_text("Generating your talking-head video...")
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_VIDEO)

    try:
        # Step 1: TTS
        audio_bytes = await text_to_speech(text, voice=voice)

        # Step 2: Talking head
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_VIDEO)
        video_bytes = await create_talking_head_video(audio_bytes, avatar_url=avatar_url)

        # Step 3: Send video
        await update.message.reply_video(
            video=io.BytesIO(video_bytes),
            filename="talking_head.mp4",
            caption="Here's your talking-head video!",
        )
    except Exception:
        logger.exception("Failed to generate video for text message")
        await update.message.reply_text(
            "Sorry, something went wrong generating your video. Please try again."
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio messages — transcribe, then generate video."""
    voice_msg = update.message.voice or update.message.audio
    if not voice_msg:
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

    await update.message.reply_text(f'Transcript: "{transcript}"\n\nGenerating video...')
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_VIDEO)

    voice_name = context.user_data.get(USER_VOICE, DEFAULT_TTS_VOICE)
    avatar_url = context.user_data.get(USER_AVATAR)

    try:
        audio_bytes = await text_to_speech(transcript, voice=voice_name)
        video_bytes = await create_talking_head_video(audio_bytes, avatar_url=avatar_url)
        await update.message.reply_video(
            video=io.BytesIO(video_bytes),
            filename="talking_head.mp4",
            caption="Here's your talking-head video!",
        )
    except Exception:
        logger.exception("Failed to generate video from voice message")
        await update.message.reply_text(
            "Sorry, something went wrong generating your video. Please try again."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — set as custom avatar."""
    photos = update.message.photo
    if not photos:
        return

    # Use the largest resolution photo
    photo = photos[-1]

    try:
        tg_file = await context.bot.get_file(photo.file_id)
        # Store the Telegram file URL as the avatar
        # Note: Telegram file URLs expire, so for production you'd want to
        # re-upload to a persistent store. For MVP, we re-fetch each time.
        context.user_data[USER_AVATAR] = tg_file.file_path
        await update.message.reply_text(
            "Avatar updated! I'll use this face for your next videos.\n"
            "Use /avatar_reset to go back to the default."
        )
    except Exception:
        logger.exception("Failed to set avatar from photo")
        await update.message.reply_text("Sorry, I couldn't process that photo.")
