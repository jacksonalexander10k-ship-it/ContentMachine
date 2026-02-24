import logging

import httpx

logger = logging.getLogger(__name__)


async def download_telegram_file(bot, file_id: str) -> bytes:
    """Download a file from Telegram by file_id."""
    tg_file = await bot.get_file(file_id)
    buf = bytearray()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(tg_file.file_path)
        resp.raise_for_status()
        buf.extend(resp.content)
    logger.info("Downloaded Telegram file %s: %d bytes", file_id, len(buf))
    return bytes(buf)
