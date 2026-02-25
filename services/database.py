import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "contentmachine.db"


async def init_db() -> None:
    """Create database tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                voice TEXT NOT NULL DEFAULT 'nova',
                avatar_data BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)


async def get_user_voice(user_id: int) -> str:
    """Get user's TTS voice preference. Returns 'nova' if not set."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT voice FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
    return row[0] if row else "nova"


async def set_user_voice(user_id: int, voice: str) -> None:
    """Set user's TTS voice preference."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO users (user_id, voice) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   voice = excluded.voice,
                   updated_at = datetime('now')""",
            (user_id, voice),
        )
        await db.commit()


async def get_user_avatar(user_id: int) -> bytes | None:
    """Get user's avatar image bytes, or None if not set."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT avatar_data FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
    return bytes(row[0]) if row and row[0] else None


async def set_user_avatar(user_id: int, avatar_data: bytes) -> None:
    """Store user's avatar image bytes."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO users (user_id, avatar_data) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   avatar_data = excluded.avatar_data,
                   updated_at = datetime('now')""",
            (user_id, avatar_data),
        )
        await db.commit()


async def clear_user_avatar(user_id: int) -> None:
    """Remove user's custom avatar."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET avatar_data = NULL, updated_at = datetime('now') WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
