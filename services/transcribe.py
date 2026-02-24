import io
import logging
from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribe audio bytes to text using OpenAI Whisper."""
    logger.info("Transcribing audio: %d bytes", len(audio_bytes))
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    transcript = await _client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
    )
    text = transcript.text
    logger.info("Transcription complete: %d chars", len(text))
    return text
