import asyncio
import base64
import logging
import time
from collections.abc import Awaitable, Callable

import httpx
import jwt

from bot.config import (
    KLING_ACCESS_KEY,
    KLING_API_BASE,
    KLING_MAX_CONCURRENT,
    KLING_POLL_INTERVAL_SECONDS,
    KLING_POLL_MAX_ATTEMPTS,
    KLING_SECRET_KEY,
)

logger = logging.getLogger(__name__)

# Concurrency limiter for Kling API
_semaphore = asyncio.Semaphore(KLING_MAX_CONCURRENT)

# Type alias for progress callbacks
ProgressCallback = Callable[[str, int, int], Awaitable[None]]

# Kling error code -> user-friendly message
_ERROR_MESSAGES: dict[int, str] = {
    1000: "Authentication failed. Please contact the bot admin.",
    1001: "Authentication failed. Please contact the bot admin.",
    1002: "Authentication failed. Please contact the bot admin.",
    1003: "Authentication failed. Please contact the bot admin.",
    1004: "Authentication token expired. Please try again.",
    1101: "API account issue. Please contact the bot admin.",
    1102: "API payment issue. Please contact the bot admin.",
    1200: "Invalid request parameters. Please try a different image or shorter text.",
    1201: "Invalid request parameters. Please try a different image or shorter text.",
    1203: "Invalid request parameters. Please try a different image or shorter text.",
    1301: "Content was flagged by the safety filter. Try different text or a different avatar photo.",
    1303: "The service is busy right now. Please try again in a minute.",
    5000: "Kling AI is experiencing server issues. Please try again later.",
    5001: "Kling AI is experiencing server issues. Please try again later.",
}


class KlingAPIError(Exception):
    """Kling API error with a user-facing message."""

    def __init__(self, code: int, api_message: str):
        self.code = code
        self.api_message = api_message
        self.user_message = _ERROR_MESSAGES.get(
            code,
            f"Video generation failed (error {code}). Please try again.",
        )
        super().__init__(f"Kling error {code}: {api_message}")


def _generate_jwt_token() -> str:
    """Generate a JWT token for Kling AI API authentication."""
    now = int(time.time())
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": now + 1800,
        "nbf": now - 5,
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256", headers=headers)


def _auth_headers() -> dict[str, str]:
    token = _generate_jwt_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def create_talking_head_video(
    audio_bytes: bytes,
    avatar_bytes: bytes,
    progress_callback: ProgressCallback | None = None,
) -> bytes:
    """Generate a talking head video using Kling AI's avatar API.

    Args:
        audio_bytes: Raw audio file bytes (MP3, WAV, etc.)
        avatar_bytes: Raw image bytes for the face (JPEG, PNG, etc.)
        progress_callback: Optional async callback(status, attempt, max_attempts)
            called periodically during polling.

    Returns:
        The generated MP4 video as bytes.
    """
    async with _semaphore:
        image_b64 = base64.b64encode(avatar_bytes).decode()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        task_id = await _create_avatar_task(image_b64, audio_b64)
        result_url = await _poll_task(task_id, progress_callback)
        return await _download_video(result_url)


async def _create_avatar_task(image_b64: str, audio_b64: str) -> str:
    """Create a Kling avatar task. Returns the task ID."""
    logger.info("Creating Kling avatar task...")
    payload = {
        "image": image_b64,
        "sound_file": audio_b64,
        "mode": "std",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{KLING_API_BASE}/v1/videos/avatar/image2video",
            headers=_auth_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        raise KlingAPIError(code, data.get("message", "Unknown error"))

    task_id = data["data"]["task_id"]
    logger.info("Avatar task created: %s", task_id)
    return task_id


async def _poll_task(
    task_id: str,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Poll a Kling avatar task until completion. Returns the video URL."""
    logger.info("Polling task %s...", task_id)
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, KLING_POLL_MAX_ATTEMPTS + 1):
            resp = await client.get(
                f"{KLING_API_BASE}/v1/videos/avatar/image2video/{task_id}",
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            code = data.get("code", -1)
            if code != 0:
                raise KlingAPIError(code, data.get("message", "Unknown error"))

            task_data = data["data"]
            status = task_data.get("task_status")

            if status == "succeed":
                videos = task_data["task_result"]["videos"]
                result_url = videos[0]["url"]
                logger.info("Task %s succeeded: %s", task_id, result_url)
                return result_url

            if status == "failed":
                msg = task_data.get("task_status_msg", "Unknown failure")
                raise KlingAPIError(0, msg)

            # Send progress updates every ~15 seconds (every 5 polls)
            if progress_callback and attempt % 5 == 0:
                try:
                    await progress_callback(status, attempt, KLING_POLL_MAX_ATTEMPTS)
                except Exception:
                    logger.debug("Progress callback failed", exc_info=True)

            logger.debug(
                "Task %s status=%s (attempt %d/%d)",
                task_id, status, attempt, KLING_POLL_MAX_ATTEMPTS,
            )
            await asyncio.sleep(KLING_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Task {task_id} did not complete within polling limit")


async def _download_video(url: str) -> bytes:
    """Download the generated video."""
    logger.info("Downloading video from %s", url)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    logger.info("Video downloaded: %d bytes", len(resp.content))
    return resp.content


async def check_kling_credentials() -> bool:
    """Verify Kling API credentials by calling the account endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{KLING_API_BASE}/account/costs",
                headers=_auth_headers(),
            )
            data = resp.json()
            return data.get("code") == 0
    except Exception:
        return False
