"""Episode analysis via LLM for TV Recap Engine.

Takes an episode summary and uses the configured LLM provider (reusing
MPT's existing ``app.services.llm``) to produce a structured recap script
with narration text and a shot-by-shot breakdown.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from app.services import llm
from app.tv_recap.models import (
    EpisodeInput,
    RecapScript,
    Shot,
    ShowConfig,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EPISODE_ANALYSIS_SYSTEM_PROMPT = """
# Role: TV Episode Recap Script Writer

## Goal
You are an expert TV recap scriptwriter. Given an episode summary, produce
a structured JSON recap script for a ~60-second AI-generated video recap.

## Rules
1. Output ONLY valid JSON — no markdown fences, no commentary.
2. The total duration of all shots must sum to approximately {target_duration} seconds.
3. Create {min_shots}–{max_shots} shots that tell the episode's story arc.
4. Each shot must specify which characters appear, the location, the action,
   a camera angle, and a duration (3–8 seconds).
5. Write a continuous narration (under "narration") that covers the full
   {target_duration} seconds. It should be dramatic and engaging — this
   narration will be read as voiceover over the video clips.
6. Do NOT reproduce copyrighted dialogue verbatim. Paraphrase the story.
7. Use the character descriptions provided to maintain visual consistency.
8. Respond in {language}.
""".strip()

EPISODE_ANALYSIS_USER_PROMPT = """
## Show Information
- Show: {show_name}
- Style: {style}
- Tone: {tone}

## Characters
{characters_block}

## Locations
{locations_block}

## Episode
- ID: {episode_id}
- Title: {episode_title}

## Episode Summary
{summary}

## Output Format
Return a single JSON object with this structure:
{{
  "narration": "Full narration text for the voiceover...",
  "shots": [
    {{
      "index": 1,
      "duration": 5,
      "characters": ["CharacterName"],
      "location": "LocationName",
      "action": "What happens in this shot",
      "camera": "medium"
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Analysis function
# ---------------------------------------------------------------------------

def _build_characters_block(show_config: ShowConfig | None) -> str:
    """Format the character bible for inclusion in the LLM prompt."""
    if not show_config or not show_config.characters:
        return "(No character descriptions provided — invent original characters.)"

    lines = []
    for char in show_config.characters:
        lines.append(f"- {char.to_prompt_description()}")
    return "\n".join(lines)


def _build_locations_block(show_config: ShowConfig | None) -> str:
    """Format the location list for inclusion in the LLM prompt."""
    if not show_config or not show_config.locations:
        return "(No specific locations provided — infer from the summary.)"

    lines = []
    for loc in show_config.locations:
        lines.append(f"- {loc.to_prompt_description()}")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_recap_response(
    response: str,
    episode_id: str,
    show_name: str,
    target_duration: int,
) -> RecapScript:
    """Parse the LLM JSON response into a RecapScript."""
    cleaned = _strip_code_fence(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM response as JSON: {cleaned[:200]}")
        data = json.loads(match.group())

    narration = data.get("narration", "")
    raw_shots = data.get("shots", [])

    shots = []
    for i, raw_shot in enumerate(raw_shots):
        shot = Shot(
            index=raw_shot.get("index", i + 1),
            duration=raw_shot.get("duration", 5),
            characters=raw_shot.get("characters", []),
            location=raw_shot.get("location", ""),
            action=raw_shot.get("action", ""),
            camera=raw_shot.get("camera", "medium"),
        )
        shots.append(shot)

    return RecapScript(
        episode_id=episode_id,
        show_name=show_name,
        total_duration=target_duration,
        narration=narration,
        shots=shots,
    )


def analyze_episode(
    episode: EpisodeInput,
    show_config: ShowConfig | None = None,
    target_duration: int = 60,
    min_shots: int = 8,
    max_shots: int = 15,
) -> RecapScript:
    """Analyze an episode and produce a structured recap script.

    Uses the currently configured LLM provider via ``app.services.llm``.

    Args:
        episode: The episode summary to analyze.
        show_config: Optional show configuration with character bible.
        target_duration: Target video duration in seconds.
        min_shots: Minimum number of shots to generate.
        max_shots: Maximum number of shots to generate.

    Returns:
        A validated RecapScript with narration and shot list.

    Raises:
        ValueError: If the LLM fails to produce a valid response.
    """
    logger.info(
        f"analyzing episode: {episode.show_name} {episode.episode_id}, "
        f"target_duration={target_duration}s, shots={min_shots}-{max_shots}"
    )

    style = (show_config.style if show_config else "") or "drama"
    tone = (show_config.tone if show_config else "") or "dramatic"

    system_prompt = EPISODE_ANALYSIS_SYSTEM_PROMPT.format(
        target_duration=target_duration,
        min_shots=min_shots,
        max_shots=max_shots,
        language=episode.language,
    )

    user_prompt = EPISODE_ANALYSIS_USER_PROMPT.format(
        show_name=episode.show_name,
        style=style,
        tone=tone,
        characters_block=_build_characters_block(show_config),
        locations_block=_build_locations_block(show_config),
        episode_id=episode.episode_id,
        episode_title=episode.title or "(untitled)",
        summary=episode.summary,
    )

    # Combine system + user prompt (MPT's llm uses single-prompt interface)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # Use MPT's existing LLM infrastructure with retries
    response = llm.generate_script(
        video_subject=f"{episode.show_name} {episode.episode_id} recap",
        language=episode.language,
        paragraph_number=1,
        custom_system_prompt=system_prompt,
        video_script_prompt=user_prompt,
    )

    if not response or "Error: " in response:
        error_msg = (
            response.removeprefix("Error: ").strip()
            if response and "Error: " in response
            else "LLM returned empty response"
        )
        raise ValueError(f"Episode analysis failed: {error_msg}")

    recap = _parse_recap_response(
        response=response,
        episode_id=episode.episode_id,
        show_name=episode.show_name,
        target_duration=target_duration,
    )

    logger.success(
        f"episode analysis complete: {episode.episode_id}, "
        f"shots={len(recap.shots)}, "
        f"planned_duration={recap.planned_duration}s, "
        f"narration_length={len(recap.narration)} chars"
    )
    return recap
