"""End-to-end TV recap pipeline orchestration."""

from __future__ import annotations

import json
import os

from loguru import logger

from app.models.schema import VideoAspect, VideoParams
from app.tv_recap import character_manager
from app.tv_recap.episode_analyzer import analyze_episode
from app.tv_recap.models import EpisodeInput, RecapScript, ShowConfig
from app.tv_recap.scene_planner import plan_scenes
from app.tv_recap.shot_generator import generate_shots
from app.tv_recap.video_assembler import assemble_recap


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_recap_script(recap_script: RecapScript) -> str:
    episode_dir = character_manager.episode_dir(
        recap_script.show_name,
        recap_script.episode_id,
    )
    path = os.path.join(episode_dir, "recap_script.json")
    _write_json(path, recap_script)
    return path


def save_episode_input(episode: EpisodeInput) -> str:
    episode_dir = character_manager.episode_dir(
        episode.show_name,
        episode.episode_id,
    )
    path = os.path.join(episode_dir, "input.json")
    _write_json(path, episode)
    return path


def run_recap_pipeline(
    *,
    task_id: str,
    episode: EpisodeInput,
    show_config: ShowConfig | None = None,
    base_video_params: VideoParams | None = None,
    shot_provider: str = "volcengine_seedance",
    video_aspect: VideoAspect | str = VideoAspect.portrait,
    target_duration: int = 60,
    min_shots: int = 8,
    max_shots: int = 15,
    stop_at: str = "video",
) -> dict:
    """Run the Phase 1 TV recap pipeline.

    The pipeline is episode summary to structured recap to planned shots to
    generated clips to MPT local-material assembly.
    """
    logger.info(
        f"start TV recap pipeline: task_id={task_id}, "
        f"episode={episode.show_name} {episode.episode_id}"
    )
    if show_config is None:
        show_config = character_manager.load_show_config(episode.show_name)

    save_episode_input(episode)

    recap_script = analyze_episode(
        episode=episode,
        show_config=show_config,
        target_duration=target_duration,
        min_shots=min_shots,
        max_shots=max_shots,
    )
    recap_script = plan_scenes(recap_script, show_config=show_config)
    save_recap_script(recap_script)

    recap_script = generate_shots(
        recap_script,
        task_id=task_id,
        provider=shot_provider,
        video_aspect=video_aspect,
    )
    save_recap_script(recap_script)

    result = assemble_recap(
        recap_script,
        task_id=task_id,
        base_params=base_video_params,
        stop_at=stop_at,
        allow_server_file_input=True,
    )
    logger.success(
        f"TV recap pipeline finished: task_id={task_id}, "
        f"episode={episode.episode_id}"
    )
    return {
        "task_id": task_id,
        "episode_id": episode.episode_id,
        "recap_script": recap_script.model_dump(mode="json"),
        "assembly_result": result,
    }
