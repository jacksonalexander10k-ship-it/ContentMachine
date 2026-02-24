import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Kling AI credentials
KLING_ACCESS_KEY = os.environ["KLING_ACCESS_KEY"]
KLING_SECRET_KEY = os.environ["KLING_SECRET_KEY"]

# Default avatar image URL (face photo for talking head generation)
DEFAULT_AVATAR_URL = os.getenv(
    "DEFAULT_AVATAR_URL",
    "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg",
)

# TTS defaults
DEFAULT_TTS_VOICE = "nova"
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Kling API settings
KLING_API_BASE = "https://api.klingai.com"
KLING_POLL_INTERVAL_SECONDS = 3
KLING_POLL_MAX_ATTEMPTS = 100  # ~5 minutes at 3s intervals

# Limits
MAX_TEXT_LENGTH = 2500
MAX_AUDIO_DURATION_SECONDS = 300
