"""Scene planning — enriches recap shots with prompts and reference images.

Takes the raw RecapScript from the episode analyzer and produces fully
specified shots with:
  - Detailed video-generation prompts (incorporating character descriptions)
  - Reference image paths (from the character bible)
  - Duration adjustments to hit the target total
"""

from __future__ import annotations

from loguru import logger

from app.tv_recap.character_manager import (
    get_character_reference,
    get_location_reference,
)
from app.tv_recap.models import (
    RecapScript,
    Shot,
    ShowConfig,
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_shot_prompt(
    shot: Shot,
    show_config: ShowConfig | None,
    style: str = "",
) -> str:
    """Build a detailed video-generation prompt for a single shot.

    Incorporates character descriptions, location context, camera angle,
    and show style into a prompt suitable for text-to-video models.
    """
    parts = []

    # Camera and framing
    parts.append(f"{shot.camera} shot")

    # Characters with full descriptions
    if shot.characters and show_config:
        char_descriptions = []
        for char_name in shot.characters:
            char = show_config.get_character(char_name)
            if char:
                char_descriptions.append(char.to_prompt_description())
            else:
                char_descriptions.append(char_name)
        if len(char_descriptions) == 1:
            parts.append(f"of {char_descriptions[0]}")
        else:
            parts.append(f"of {' and '.join(char_descriptions)}")
    elif shot.characters:
        parts.append(f"of {' and '.join(shot.characters)}")

    # Location
    if shot.location:
        if show_config:
            loc = show_config.get_location(shot.location)
            if loc:
                parts.append(f"in {loc.to_prompt_description()}")
            else:
                parts.append(f"in {shot.location}")
        else:
            parts.append(f"in {shot.location}")

    # Action
    if shot.action:
        parts.append(f"— {shot.action}")

    # Style suffix
    if style:
        parts.append(f"Style: {style}")
    parts.append("cinematic lighting, high quality, 4K")

    return ". ".join(parts)


def _collect_reference_images(
    shot: Shot,
    show_config: ShowConfig | None,
    show_name: str,
) -> list[str]:
    """Collect all available reference images for a shot."""
    references = []

    if not show_config:
        return references

    # Character references
    for char_name in shot.characters:
        ref = get_character_reference(show_name, char_name)
        if ref:
            references.append(ref)

    # Location reference
    if shot.location:
        ref = get_location_reference(show_name, shot.location)
        if ref:
            references.append(ref)

    return references


# ---------------------------------------------------------------------------
# Duration adjustment
# ---------------------------------------------------------------------------

def _adjust_durations(shots: list[Shot], target_duration: int) -> list[Shot]:
    """Adjust shot durations so they sum to exactly the target.

    Uses proportional scaling: each shot keeps its relative weight but the
    total is forced to match.  Minimum shot duration is 3 seconds.
    """
    if not shots:
        return shots

    current_total = sum(s.duration for s in shots)
    if current_total == target_duration:
        return shots

    if current_total <= 0:
        # Fallback: distribute evenly
        per_shot = max(3, target_duration // len(shots))
        for i, shot in enumerate(shots):
            shot.duration = per_shot
        # Fix rounding by adjusting last shot
        remainder = target_duration - (per_shot * len(shots))
        shots[-1].duration += remainder
        return shots

    # Proportional scaling
    scale = target_duration / current_total
    adjusted_total = 0
    for shot in shots[:-1]:
        new_duration = max(3, round(shot.duration * scale))
        shot.duration = new_duration
        adjusted_total += new_duration

    # Last shot absorbs rounding error
    shots[-1].duration = max(3, target_duration - adjusted_total)

    return shots


# ---------------------------------------------------------------------------
# Main planning function
# ---------------------------------------------------------------------------

def plan_scenes(
    recap_script: RecapScript,
    show_config: ShowConfig | None = None,
) -> RecapScript:
    """Enrich a recap script with detailed prompts and references.

    This is the bridge between the episode analyzer (which produces abstract
    shot descriptions) and the shot generator (which needs concrete prompts
    and reference images).

    Args:
        recap_script: The raw recap script from episode analysis.
        show_config: Optional show config with character bible.

    Returns:
        The same RecapScript with enriched shots (prompts, references,
        adjusted durations).
    """
    show_name = recap_script.show_name
    style = (show_config.style if show_config else "") or ""

    logger.info(
        f"planning scenes: {recap_script.episode_id}, "
        f"shots={len(recap_script.shots)}, "
        f"target={recap_script.total_duration}s"
    )

    # Adjust durations to hit target
    recap_script.shots = _adjust_durations(
        recap_script.shots,
        recap_script.total_duration,
    )

    # Enrich each shot with prompt and references
    for shot in recap_script.shots:
        shot.prompt = _build_shot_prompt(shot, show_config, style)
        shot.reference_images = _collect_reference_images(
            shot, show_config, show_name,
        )

    logger.success(
        f"scene planning complete: {recap_script.episode_id}, "
        f"shots={len(recap_script.shots)}, "
        f"planned_duration={recap_script.planned_duration}s, "
        f"shots_with_refs="
        f"{sum(1 for s in recap_script.shots if s.reference_images)}"
    )

    return recap_script
