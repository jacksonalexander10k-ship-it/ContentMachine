from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bot.config import AVAILABLE_VOICES

VOICE_CALLBACK_PREFIX = "voice:"


def voice_selection_keyboard(current_voice: str) -> InlineKeyboardMarkup:
    """Build an inline keyboard for TTS voice selection."""
    buttons = []
    for voice in AVAILABLE_VOICES:
        label = f"{'> ' if voice == current_voice else ''}{voice.capitalize()}"
        buttons.append(
            InlineKeyboardButton(label, callback_data=f"{VOICE_CALLBACK_PREFIX}{voice}")
        )
    # Arrange in 2-column rows
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)
