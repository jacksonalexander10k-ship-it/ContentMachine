import asyncio
import base64
import logging
import time
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

# Timeout for a single clip generation (10 minutes)
CLIP_TIMEOUT_SECONDS = 600

# How often to send heartbeat updates when fal.ai is silent (seconds)
HEARTBEAT_INTERVAL = 15

# Voice direction — consistent East London accent across all clips
VOICE_PROMPT = (
    "The speaker has an East London UK accent. "
    "Well-spoken but distinctly East London — natural, confident, conversational tone."
)

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


def _format_elapsed(start: float) -> str:
    """Format elapsed time since start as a human-readable string."""
    elapsed = int(time.monotonic() - start)
    if elapsed < 60:
        return f"{elapsed}s"
    minutes, seconds = divmod(elapsed, 60)
    return f"{minutes}m {seconds}s"


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

        # Build the prompt: voice direction + spoken text + motion direction
        prompt = (
            f'{VOICE_PROMPT} '
            f'The person in the image speaks directly to camera and says: '
            f'"{segment_text}" '
            f'{DEFAULT_MOTION_PROMPT}'
        )

        logger.info("Submitting fal.ai clip: %s...", segment_text[:60])
        start_time = time.monotonic()

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

            request_id = getattr(handle, "request_id", None)
            logger.info("fal.ai job submitted (request_id=%s)", request_id)

            if progress_callback:
                elapsed = _format_elapsed(start_time)
                await _safe_callback(progress_callback, f"Submitted to fal.ai ({elapsed})")

            # Poll for progress with a heartbeat so the user always sees updates
            result = await asyncio.wait_for(
                _poll_with_heartbeat(handle, progress_callback, start_time),
                timeout=CLIP_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            elapsed = _format_elapsed(start_time)
            logger.error("fal.ai clip timed out after %s", elapsed)
            raise VideoGenerationError(
                f"Video generation timed out after {elapsed}. Please try again."
            )
        except VideoGenerationError:
            raise
        except Exception as e:
            elapsed = _format_elapsed(start_time)
            logger.exception("fal.ai request failed after %s", elapsed)
            raise VideoGenerationError(
                f"Video generation failed after {elapsed}: {e}"
            ) from e

        video_info = result.get("video")
        if not video_info or not video_info.get("url"):
            raise VideoGenerationError("No video URL in response.")

        video_url = video_info["url"]
        elapsed = _format_elapsed(start_time)
        logger.info("Clip ready after %s: %s", elapsed, video_url)

        if progress_callback:
            await _safe_callback(progress_callback, f"Downloading video ({elapsed})")

        return await _download_video(video_url)


async def _safe_callback(callback: ProgressCallback, text: str) -> None:
    """Call a progress callback, swallowing exceptions."""
    try:
        await callback(text)
    except Exception:
        pass


async def _poll_with_heartbeat(
    handle,
    progress_callback: ProgressCallback | None,
    start_time: float,
) -> dict:
    """Poll fal.ai for events while sending periodic heartbeat updates.

    Returns the final result dict.
    """
    last_status = "Processing..."
    last_update_time = time.monotonic()

    async def _heartbeat_loop():
        """Background task that sends elapsed-time updates when fal.ai is quiet."""
        nonlocal last_update_time
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if progress_callback:
                elapsed = _format_elapsed(start_time)
                await _safe_callback(
                    progress_callback,
                    f"{last_status} ({elapsed})",
                )
                last_update_time = time.monotonic()

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    try:
        async for event in handle.iter_events(with_logs=True):
            if isinstance(event, fal_client.InProgress):
                if event.logs:
                    for log_entry in event.logs:
                        # Handle both dict and object log entries
                        if isinstance(log_entry, dict):
                            msg = log_entry.get("message", "")
                        else:
                            msg = getattr(log_entry, "message", "")
                        if msg:
                            last_status = msg
                            logger.info("fal.ai progress: %s", msg)
                    if progress_callback:
                        elapsed = _format_elapsed(start_time)
                        await _safe_callback(
                            progress_callback,
                            f"{last_status} ({elapsed})",
                        )
                        last_update_time = time.monotonic()
                else:
                    # InProgress event but no logs — still show heartbeat
                    logger.debug("fal.ai InProgress event (no logs)")
                    if progress_callback:
                        elapsed = _format_elapsed(start_time)
                        await _safe_callback(
                            progress_callback,
                            f"Processing... ({elapsed})",
                        )
            else:
                logger.debug("fal.ai event type: %s", type(event).__name__)

        return await handle.get()
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def _download_video(url: str) -> bytes:
    """Download the generated video."""
    logger.info("Downloading video from %s", url)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    logger.info("Video downloaded: %d bytes", len(resp.content))
    return resp.content
