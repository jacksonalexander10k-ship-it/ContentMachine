import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DID_API_KEY = os.environ["DID_API_KEY"]

DEFAULT_AVATAR_URL = os.getenv(
    "DEFAULT_AVATAR_URL",
    "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg",
)

# TTS defaults
DEFAULT_TTS_VOICE = "nova"
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# D-ID settings
DID_API_BASE = "https://api.d-id.com"
DID_POLL_INTERVAL_SECONDS = 3
DID_POLL_MAX_ATTEMPTS = 60

# Limits
MAX_TEXT_LENGTH = 2000
MAX_AUDIO_DURATION_SECONDS = 120
