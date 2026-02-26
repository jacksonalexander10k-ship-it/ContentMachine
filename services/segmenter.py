import json
import logging

from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_SEGMENTATION_PROMPT = """\
You are a script segmentation assistant. Your job is to split a spoken script \
into natural segments for talking-head video clip generation.

Each segment will become a SEPARATE video clip with the same starting frame, \
so segments must sound natural and complete when spoken individually.

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

Return a JSON object: {"segments": ["segment 1 text", "segment 2 text", ...]}"""


async def segment_script(script: str) -> list[str]:
    """Split a script into natural speech segments using OpenAI.

    Returns a list of segment strings preserving original wording.
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
    segments = result["segments"]

    logger.info("Script segmented into %d parts: %s",
                len(segments),
                [f"{s[:40]}..." if len(s) > 40 else s for s in segments])

    return segments
