import logging

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot.config import TELEGRAM_BOT_TOKEN, WEBHOOK_PORT, WEBHOOK_URL
from bot.handlers import (
    avatar_command,
    avatar_reset_command,
    handle_photo,
    handle_text,
    handle_voice,
    help_command,
    start_command,
    status_command,
    stop_command,
)
from services.database import init_db

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def post_init(application) -> None:
    """Run after bot initialization — database setup."""
    await init_db()

    me = await application.bot.get_me()
    logger.info("Bot connected as @%s (id=%d)", me.username, me.id)


def main() -> None:
    """Start the Telegram bot."""
    logger.info("Starting ContentMachine bot...")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("avatar", avatar_command))
    app.add_handler(CommandHandler("avatar_reset", avatar_reset_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("status", status_command))

    # Messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    if WEBHOOK_URL:
        logger.info("Running in webhook mode on port %d", WEBHOOK_PORT)
        app.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        )
    else:
        logger.info("Running in polling mode.")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
