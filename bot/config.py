import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# fal.ai API key (for Kling 3.0 Pro video generation)
# Set FAL_KEY env var — the fal_client reads it automatically
FAL_KEY = os.environ["FAL_KEY"]

# Default starting frame image URL (optional — if unset, users must upload a photo)
DEFAULT_AVATAR_URL = os.getenv("DEFAULT_AVATAR_URL", "")

# fal.ai concurrency
FAL_MAX_CONCURRENT = int(os.getenv("FAL_MAX_CONCURRENT", "2"))

# Limits
MAX_TEXT_LENGTH = 10000  # scripts can be long
MAX_AUDIO_DURATION_SECONDS = 300

# Webhook (optional — if set, runs in webhook mode instead of polling)
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("PORT", "8443"))
