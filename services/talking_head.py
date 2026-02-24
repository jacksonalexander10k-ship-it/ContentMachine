import asyncio
import base64
import logging
import time

import httpx
import jwt

from bot.config import (
    KLING_ACCESS_KEY,
    KLING_API_BASE,
    KLING_POLL_INTERVAL_SECONDS,
    KLING_POLL_MAX_ATTEMPTS,
    KLING_SECRET_KEY,
    DEFAULT_AVATAR_URL,
)

logger = logging.getLogger(__name__)


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


async def _download_image_as_base64(url: str) -> str:
    """Download an image from a URL and return it as a base64 string."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return base64.b64encode(resp.content).decode()


async def create_talking_head_video(
    audio_bytes: bytes,
    avatar_url: str | None = None,
) -> bytes:
    """Generate a talking head video using Kling AI's avatar/lip-sync API.

    Takes audio bytes and a face image URL, returns the final MP4 video bytes.
    """
    avatar_url = avatar_url or DEFAULT_AVATAR_URL

    # Convert face image to base64
    logger.info("Downloading avatar image from %s", avatar_url)
    image_b64 = await _download_image_as_base64(avatar_url)

    # Convert audio to base64
    audio_b64 = base64.b64encode(audio_bytes).decode()

    # Create the avatar/lip-sync task
    task_id = await _create_avatar_task(image_b64, audio_b64)

    # Poll until done
    result_url = await _poll_task(task_id)

    # Download the result video
    return await _download_video(result_url)


async def _create_avatar_task(image_b64: str, audio_b64: str) -> str:
    """Create a Kling avatar lip-sync task. Returns the task ID."""
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

    if data.get("code") != 0:
        raise RuntimeError(f"Kling API error: {data.get('message', 'Unknown error')}")

    task_id = data["data"]["task_id"]
    logger.info("Avatar task created: %s", task_id)
    return task_id


async def _poll_task(task_id: str) -> str:
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

            if data.get("code") != 0:
                raise RuntimeError(
                    f"Kling poll error: {data.get('message', 'Unknown error')}"
                )

            task_data = data["data"]
            status = task_data.get("task_status")

            if status == "succeed":
                videos = task_data["task_result"]["videos"]
                result_url = videos[0]["url"]
                logger.info("Task %s succeeded: %s", task_id, result_url)
                return result_url

            if status == "failed":
                msg = task_data.get("task_status_msg", "Unknown failure")
                raise RuntimeError(f"Kling task failed: {msg}")

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
