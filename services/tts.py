import logging
from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY, DEFAULT_TTS_VOICE

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def text_to_speech(
    text: str,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = "tts-1",
    response_format: str = "mp3",
) -> bytes:
    """Convert text to speech audio bytes using OpenAI TTS."""
    logger.info("Generating TTS: voice=%s, model=%s, length=%d", voice, model, len(text))
    response = await _client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
    )
    audio_bytes = response.read()
    logger.info("TTS complete: %d bytes", len(audio_bytes))
    return audio_bytes
