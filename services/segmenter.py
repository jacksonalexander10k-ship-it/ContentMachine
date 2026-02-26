import json
import logging

from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_SEGMENTATION_PROMPT = """\
You are a script segmentation assistant. Your job is to split a spoken script \
into natural segments for talking-head video clip generation.

FIRST: Determine if the input is actually a speakable script. A valid script \
is text that a person could naturally speak to camera — a monologue, \
presentation, tutorial, story, pitch, narration, etc.

REJECT the input (return empty segments) if it is:
- Random gibberish, keyboard mashing, or nonsense words
- A question or command directed at a chatbot (e.g. "what can you do?", \
"generate me a script about X", "help me with Y")
- A request to CREATE content rather than actual content to be spoken
- Code, URLs, or non-speakable technical text
- Just a few random unrelated words that don't form coherent speech

If the input is NOT a valid speakable script, return:
{"segments": [], "rejected": true, "reason": "<brief explanation>"}

If the input IS a valid script, segment it:

Rules:
- Each segment should be roughly 10-30 words (approx 5-15 seconds of speech).
- NEVER cut mid-sentence or mid-phrase. Each segment must sound complete and \
natural when spoken on its own.
- Keep logical phrasing intact. A segment can be one sentence, a couple of \
short sentences, or a natural clause — but it must make sense when heard alone.
- "The man went to the shop to get some milk" = good segment.
- "The man went" / "to the shop to get some milk" = bad, unnatural split.
- Preserve the EXACT original wording. Do not rephrase, add, or remove words.
- If the script is very short (one sentence or under 30 words), return it as \
a single segment.

Return: {"segments": ["segment 1 text", "segment 2 text", ...], "rejected": false}"""


class ScriptRejected(Exception):
    """The input was not a valid speakable script."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


async def segment_script(script: str) -> list[str]:
    """Split a script into natural speech segments using OpenAI.

    Returns a list of segment strings preserving original wording.
    Raises ScriptRejected if the input is not a valid speakable script.
    """
    logger.info("Segmenting script (%d chars)...", len(script))

    response = await _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SEGMENTATION_PROMPT},
            {"role": "user", "content": script},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    result = json.loads(response.choices[0].message.content)

    # Check if the segmenter rejected the input
    if result.get("rejected"):
        reason = result.get("reason", "That doesn't look like a speakable script.")
        logger.info("Script rejected: %s", reason)
        raise ScriptRejected(reason)

    segments = result["segments"]
    if not segments:
        raise ScriptRejected("Could not extract any speakable segments from that text.")

    logger.info("Script segmented into %d parts: %s",
                len(segments),
                [f"{s[:40]}..." if len(s) > 40 else s for s in segments])

    return segments
