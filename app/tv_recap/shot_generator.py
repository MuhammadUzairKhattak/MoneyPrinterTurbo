"""Shot generation for the TV Recap Engine.

This module turns planned shots into local video clip files. Phase 1 reuses
the video providers already supported by MoneyPrinterTurbo instead of adding
new Kling/Veo adapters.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from loguru import logger

from app.models.schema import VideoAspect, VideoConcatMode
from app.services import material
from app.tv_recap.character_manager import episode_shots_dir
from app.tv_recap.models import RecapScript, Shot


SUPPORTED_GENERATION_PROVIDERS = frozenset(
    {
        "wavespeed",
        "volcengine_seedance",
        "ofox",
        "metaso_minimax",
        "muapi",
        "openai_image",
    }
)


def _safe_extension(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()
    return extension if extension else ".mp4"


def _copy_to_episode_shots_dir(
    *,
    source_path: str,
    show_name: str,
    episode_id: str,
    shot_index: int,
) -> str:
    """Copy a generated clip into the show's episode shots directory."""
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"generated shot clip does not exist: {source_path}")

    target_dir = episode_shots_dir(show_name, episode_id)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(
        target_dir,
        f"shot_{shot_index:03d}{_safe_extension(source_path)}",
    )
    if os.path.realpath(source_path) != os.path.realpath(target_path):
        shutil.copy2(source_path, target_path)
    return target_path


def _generate_single_shot(
    *,
    task_id: str,
    shot: Shot,
    provider: str,
    video_aspect: VideoAspect | str,
) -> str:
    prompt = (shot.prompt or shot.action or "").strip()
    if not prompt:
        raise ValueError(f"shot {shot.index} has no prompt or action")

    clip_duration = max(int(shot.duration or 1), 1)
    logger.info(
        "generating recap shot: "
        f"index={shot.index}, provider={provider}, duration={clip_duration}s"
    )
    generated_paths = material.download_videos(
        task_id=task_id,
        search_terms=[prompt],
        source=provider,
        video_aspect=VideoAspect(video_aspect),
        video_concat_mode=VideoConcatMode.sequential,
        audio_duration=clip_duration,
        max_clip_duration=clip_duration,
        match_script_order=True,
    )
    if not generated_paths:
        raise RuntimeError(f"provider returned no clip for shot {shot.index}")
    return generated_paths[0]


def generate_shots(
    recap_script: RecapScript,
    *,
    task_id: str,
    provider: str = "volcengine_seedance",
    video_aspect: VideoAspect | str = VideoAspect.portrait,
) -> RecapScript:
    """Generate local clips for every shot in a recap script.

    Args:
        recap_script: Planned recap script with prompts.
        task_id: MPT task identifier used for provider artifacts.
        provider: Existing MPT video source to use for shot generation.
        video_aspect: Target aspect ratio.

    Returns:
        The same recap script with ``Shot.clip_path`` populated.
    """
    provider = str(provider or "").strip()
    if provider not in SUPPORTED_GENERATION_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_GENERATION_PROVIDERS))
        raise ValueError(f"unsupported recap shot provider {provider!r}: {supported}")

    logger.info(
        f"generating recap shots: episode={recap_script.episode_id}, "
        f"provider={provider}, shots={len(recap_script.shots)}"
    )

    for shot in recap_script.shots:
        source_path = _generate_single_shot(
            task_id=task_id,
            shot=shot,
            provider=provider,
            video_aspect=video_aspect,
        )
        shot.clip_path = _copy_to_episode_shots_dir(
            source_path=source_path,
            show_name=recap_script.show_name,
            episode_id=recap_script.episode_id,
            shot_index=shot.index,
        )

    logger.success(
        f"recap shot generation complete: episode={recap_script.episode_id}, "
        f"clips={sum(1 for shot in recap_script.shots if shot.clip_path)}"
    )
    return recap_script
