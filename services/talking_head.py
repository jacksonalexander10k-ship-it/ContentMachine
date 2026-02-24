import asyncio
import base64
import logging

import httpx

from bot.config import (
    DID_API_BASE,
    DID_API_KEY,
    DID_POLL_INTERVAL_SECONDS,
    DID_POLL_MAX_ATTEMPTS,
    DEFAULT_AVATAR_URL,
)

logger = logging.getLogger(__name__)


def _auth_header() -> dict[str, str]:
    encoded = base64.b64encode(f"{DID_API_KEY}:".encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }


async def create_talking_head_video(
    audio_bytes: bytes,
    avatar_url: str | None = None,
) -> bytes:
    """Generate a talking head video from audio bytes and a face image URL.

    Returns the final MP4 video as bytes.
    """
    avatar_url = avatar_url or DEFAULT_AVATAR_URL

    # Upload audio to D-ID
    audio_url = await _upload_audio(audio_bytes)

    # Create the talk
    talk_id = await _create_talk(avatar_url, audio_url)

    # Poll until done
    result_url = await _poll_talk(talk_id)

    # Download the result video
    video_bytes = await _download_video(result_url)
    return video_bytes


async def _upload_audio(audio_bytes: bytes) -> str:
    """Upload audio to D-ID and return the URL."""
    logger.info("Uploading audio to D-ID: %d bytes", len(audio_bytes))
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{DID_API_BASE}/audios",
            headers={"Authorization": _auth_header()["Authorization"]},
            files={"audio": ("audio.mp3", audio_bytes, "audio/mpeg")},
        )
        resp.raise_for_status()
        data = resp.json()
    url = data["url"]
    logger.info("Audio uploaded: %s", url)
    return url


async def _create_talk(avatar_url: str, audio_url: str) -> str:
    """Create a D-ID talk and return the talk ID."""
    logger.info("Creating D-ID talk: avatar=%s", avatar_url)
    payload = {
        "source_url": avatar_url,
        "script": {
            "type": "audio",
            "audio_url": audio_url,
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{DID_API_BASE}/talks",
            headers=_auth_header(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    talk_id = data["id"]
    logger.info("Talk created: %s", talk_id)
    return talk_id


async def _poll_talk(talk_id: str) -> str:
    """Poll a D-ID talk until it's done. Returns the result video URL."""
    logger.info("Polling talk %s for completion...", talk_id)
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, DID_POLL_MAX_ATTEMPTS + 1):
            resp = await client.get(
                f"{DID_API_BASE}/talks/{talk_id}",
                headers=_auth_header(),
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "done":
                result_url = data["result_url"]
                logger.info("Talk %s done: %s", talk_id, result_url)
                return result_url

            if status in ("error", "rejected"):
                error_msg = data.get("error", {}).get("description", "Unknown error")
                raise RuntimeError(f"D-ID talk failed: {error_msg}")

            logger.debug(
                "Talk %s status=%s (attempt %d/%d)",
                talk_id, status, attempt, DID_POLL_MAX_ATTEMPTS,
            )
            await asyncio.sleep(DID_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Talk {talk_id} did not complete within polling limit")


async def _download_video(url: str) -> bytes:
    """Download the generated video."""
    logger.info("Downloading video from %s", url)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    logger.info("Video downloaded: %d bytes", len(resp.content))
    return resp.content
