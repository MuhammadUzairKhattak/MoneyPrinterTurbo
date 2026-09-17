"""MPT assembly bridge for TV recap clips.

The recap engine creates pre-generated shot clips. This module hands those clips
back to MoneyPrinterTurbo's normal local-material pipeline so voice, subtitles,
BGM, transitions, and FFmpeg assembly stay centralized.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from loguru import logger

from app.models.schema import MaterialInfo, VideoConcatMode, VideoParams
from app.services import task as task_service
from app.tv_recap.models import RecapScript
from app.utils import utils


def _copy_clip_to_local_materials(
    *,
    clip_path: str,
    task_id: str,
    index: int,
) -> str:
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"recap shot clip does not exist: {clip_path}")

    local_videos_dir = utils.storage_dir("local_videos", create=True)
    os.makedirs(local_videos_dir, exist_ok=True)
    extension = Path(clip_path).suffix.lower() or ".mp4"
    target_path = os.path.join(
        local_videos_dir,
        f"tv-recap-{task_id}-{index:03d}{extension}",
    )
    if os.path.realpath(clip_path) != os.path.realpath(target_path):
        shutil.copy2(clip_path, target_path)
    return target_path


def _build_local_materials(
    recap_script: RecapScript,
    *,
    task_id: str,
) -> list[MaterialInfo]:
    materials = []
    for index, shot in enumerate(recap_script.shots, start=1):
        if not shot.clip_path:
            raise ValueError(f"shot {shot.index} has no generated clip_path")
        material_path = _copy_clip_to_local_materials(
            clip_path=shot.clip_path,
            task_id=task_id,
            index=index,
        )
        materials.append(
            MaterialInfo(
                provider="local",
                url=material_path,
                duration=max(int(shot.duration or 0), 0),
            )
        )
    return materials


def build_video_params(
    recap_script: RecapScript,
    *,
    task_id: str,
    base_params: VideoParams | None = None,
) -> VideoParams:
    """Create MPT VideoParams for assembling a recap script."""
    materials = _build_local_materials(recap_script, task_id=task_id)
    max_clip_duration = max(
        [int(shot.duration or 1) for shot in recap_script.shots] or [5]
    )

    if base_params is None:
        params = VideoParams(
            video_subject=f"{recap_script.show_name} {recap_script.episode_id} recap",
            video_script=recap_script.narration,
        )
    else:
        params = base_params.model_copy(deep=True)

    params.video_subject = (
        params.video_subject
        or f"{recap_script.show_name} {recap_script.episode_id} recap"
    )
    params.video_script = recap_script.narration
    params.video_terms = []
    params.video_source = "local"
    params.video_materials = materials
    params.video_concat_mode = VideoConcatMode.sequential
    params.match_materials_to_script = True
    params.video_clip_duration = max_clip_duration
    return params


def assemble_recap(
    recap_script: RecapScript,
    *,
    task_id: str,
    base_params: VideoParams | None = None,
    stop_at: str = "video",
    allow_server_file_input: bool = True,
) -> dict:
    """Assemble generated recap clips through MPT's existing pipeline."""
    params = build_video_params(
        recap_script,
        task_id=task_id,
        base_params=base_params,
    )
    logger.info(
        f"assembling recap through MPT: task_id={task_id}, "
        f"episode={recap_script.episode_id}, clips={len(params.video_materials or [])}"
    )
    return task_service.start(
        task_id=task_id,
        params=params,
        stop_at=stop_at,
        allow_server_file_input=allow_server_file_input,
    )
