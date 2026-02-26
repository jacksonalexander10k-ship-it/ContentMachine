import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Kling AI credentials
KLING_ACCESS_KEY = os.environ["KLING_ACCESS_KEY"]
KLING_SECRET_KEY = os.environ["KLING_SECRET_KEY"]

# Default avatar image URL (optional — if unset, users must upload a photo)
DEFAULT_AVATAR_URL = os.getenv("DEFAULT_AVATAR_URL", "")

# TTS defaults
DEFAULT_TTS_VOICE = "nova"
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Kling API settings
KLING_API_BASE = "https://api.klingai.com"
KLING_POLL_INTERVAL_SECONDS = 3
KLING_POLL_MAX_ATTEMPTS = 100  # ~5 minutes at 3s intervals
KLING_MAX_CONCURRENT = int(os.getenv("KLING_MAX_CONCURRENT", "2"))

# Limits
MAX_TEXT_LENGTH = 10000  # scripts can be long
MAX_AUDIO_DURATION_SECONDS = 300

# Webhook (optional — if set, runs in webhook mode instead of polling)
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("PORT", "8443"))
