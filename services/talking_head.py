import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable

import httpx
import fal_client

from bot.config import FAL_MAX_CONCURRENT

logger = logging.getLogger(__name__)

# Concurrency limiter
_semaphore = asyncio.Semaphore(FAL_MAX_CONCURRENT)

# Type alias for progress callbacks
ProgressCallback = Callable[[str], Awaitable[None]]

FAL_MODEL = "fal-ai/kling-video/v3/pro/image-to-video"

# Default motion prompt — subtle realism
DEFAULT_MOTION_PROMPT = (
    "Subtle hand movements and gestures while speaking, "
    "very occasionally looks away from the camera for ultra-realism."
)


class VideoGenerationError(Exception):
    """Video generation failed."""

    def __init__(self, message: str):
        self.user_message = message
        super().__init__(message)


def _image_to_data_url(image_bytes: bytes) -> str:
    """Convert raw image bytes to a data URL for fal.ai."""
    b64 = base64.b64encode(image_bytes).decode()
    # Detect format from magic bytes
    if image_bytes[:8].startswith(b"\x89PNG"):
        mime = "image/png"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


async def generate_clip(
    image_bytes: bytes,
    segment_text: str,
    progress_callback: ProgressCallback | None = None,
) -> bytes:
    """Generate a single talking-head video clip using fal.ai Kling 3.0 Pro.

    The segment text is embedded in the prompt and Kling generates the speech
    audio natively — no separate TTS needed.

    Args:
        image_bytes: Raw image bytes for the starting frame.
        segment_text: The spoken text for this clip.
        progress_callback: Optional async callback(status_text) for progress updates.

    Returns:
        The generated MP4 video as bytes.
    """
    async with _semaphore:
        image_url = _image_to_data_url(image_bytes)

        # Build the prompt: the spoken text + motion direction
        prompt = (
            f'The person in the image speaks directly to camera and says: '
            f'"{segment_text}" '
            f'{DEFAULT_MOTION_PROMPT}'
        )

        logger.info("Submitting fal.ai clip: %s...", segment_text[:60])

        try:
            handle = await fal_client.submit_async(
                FAL_MODEL,
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "generate_audio": True,
                    "duration": "5",
                    "aspect_ratio": "9:16",
                },
            )

            # Poll for progress
            async for event in handle.iter_events(with_logs=True):
                if isinstance(event, fal_client.InProgress):
                    if progress_callback and event.logs:
                        latest = event.logs[-1].get("message", "Processing...")
                        try:
                            await progress_callback(latest)
                        except Exception:
                            pass

            result = await handle.get()

        except Exception as e:
            logger.exception("fal.ai request failed")
            raise VideoGenerationError(
                f"Video generation failed: {e}"
            ) from e

        video_info = result.get("video")
        if not video_info or not video_info.get("url"):
            raise VideoGenerationError("No video URL in response.")

        video_url = video_info["url"]
        logger.info("Clip ready: %s", video_url)

        return await _download_video(video_url)


async def _download_video(url: str) -> bytes:
    """Download the generated video."""
    logger.info("Downloading video from %s", url)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    logger.info("Video downloaded: %d bytes", len(resp.content))
    return resp.content
