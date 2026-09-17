import os
import shutil
import unittest
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

from app.models import const
from app.models.schema import VideoParams
from app.tv_recap.models import EpisodeInput, RecapScript, Shot
from app.tv_recap.pipeline import run_recap_pipeline
from app.tv_recap.shot_generator import generate_shots
from app.tv_recap.video_assembler import assemble_recap, build_video_params


class TestTvRecapEngine(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        root = Path("storage/test_tmp") / f"tv-recap-{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            yield str(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _recap_script(self, clip_path: str = "") -> RecapScript:
        return RecapScript(
            episode_id="S01E01",
            show_name="Demo Show",
            total_duration=8,
            narration="A concise recap narration.",
            shots=[
                Shot(
                    index=1,
                    duration=4,
                    characters=["Alex"],
                    location="Kitchen",
                    action="Alex finds the letter",
                    camera="close-up",
                    prompt="close-up of Alex finding a letter",
                    clip_path=clip_path or None,
                ),
                Shot(
                    index=2,
                    duration=4,
                    characters=["Blair"],
                    location="Street",
                    action="Blair runs away",
                    camera="tracking",
                    prompt="tracking shot of Blair running away",
                    clip_path=clip_path or None,
                ),
            ],
        )

    def test_generate_shots_uses_existing_provider_and_copies_episode_clips(self):
        with self._tempdir() as temp_dir:
            generated = Path(temp_dir) / "generated.mp4"
            generated.write_bytes(b"fake video")
            shots_dir = Path(temp_dir) / "shots"

            with patch(
                "app.tv_recap.shot_generator.material.download_videos",
                return_value=[str(generated)],
            ) as download_videos, patch(
                "app.tv_recap.shot_generator.episode_shots_dir",
                return_value=str(shots_dir),
            ):
                recap = generate_shots(
                    self._recap_script(),
                    task_id="task-1",
                    provider="volcengine_seedance",
                    video_aspect="9:16",
                )

            self.assertEqual(download_videos.call_count, 2)
            first_call = download_videos.call_args_list[0].kwargs
            self.assertEqual(first_call["source"], "volcengine_seedance")
            self.assertEqual(first_call["search_terms"], ["close-up of Alex finding a letter"])
            self.assertEqual(first_call["audio_duration"], 4)
            self.assertTrue(Path(recap.shots[0].clip_path).is_file())
            self.assertEqual(Path(recap.shots[0].clip_path).read_bytes(), b"fake video")

    def test_build_video_params_copies_clips_into_local_material_storage(self):
        with self._tempdir() as temp_dir:
            clip = Path(temp_dir) / "shot.mp4"
            clip.write_bytes(b"fake video")
            local_videos = Path(temp_dir) / "local_videos"
            recap = self._recap_script(str(clip))
            base_params = VideoParams(
                video_subject="custom subject",
                video_script="unused",
                voice_name="no-voice",
            )

            with patch(
                "app.tv_recap.video_assembler.utils.storage_dir",
                return_value=str(local_videos),
            ):
                params = build_video_params(
                    recap,
                    task_id="task-2",
                    base_params=base_params,
                )

            self.assertEqual(params.video_source, "local")
            self.assertEqual(params.video_subject, "custom subject")
            self.assertEqual(params.video_script, recap.narration)
            self.assertEqual(params.video_clip_duration, 4)
            self.assertEqual(len(params.video_materials), 2)
            for material in params.video_materials:
                self.assertEqual(material.provider, "local")
                self.assertTrue(os.path.realpath(material.url).startswith(os.path.realpath(str(local_videos))))
                self.assertTrue(Path(material.url).is_file())

    def test_assemble_recap_calls_mpt_task_pipeline_with_local_materials(self):
        with self._tempdir() as temp_dir:
            clip = Path(temp_dir) / "shot.mp4"
            clip.write_bytes(b"fake video")
            local_videos = Path(temp_dir) / "local_videos"
            recap = self._recap_script(str(clip))

            with patch(
                "app.tv_recap.video_assembler.utils.storage_dir",
                return_value=str(local_videos),
            ), patch(
                "app.tv_recap.video_assembler.task_service.start",
                return_value={"state": const.TASK_STATE_COMPLETE, "videos": ["final.mp4"]},
            ) as start:
                result = assemble_recap(recap, task_id="task-3")

            self.assertEqual(result["state"], const.TASK_STATE_COMPLETE)
            call = start.call_args.kwargs
            self.assertEqual(call["task_id"], "task-3")
            self.assertEqual(call["stop_at"], "video")
            self.assertEqual(call["params"].video_source, "local")
            self.assertTrue(call["allow_server_file_input"])

    def test_run_recap_pipeline_persists_inputs_and_calls_stages_in_order(self):
        with self._tempdir() as temp_dir:
            episode_dir = Path(temp_dir) / "episode"
            episode = EpisodeInput(
                show_name="Demo Show",
                season=1,
                episode=1,
                summary="Alex finds a clue and Blair flees.",
            )
            recap = self._recap_script("clip.mp4")

            with patch(
                "app.tv_recap.pipeline.character_manager.load_show_config",
                return_value=None,
            ), patch(
                "app.tv_recap.pipeline.character_manager.episode_dir",
                return_value=str(episode_dir),
            ), patch(
                "app.tv_recap.pipeline.analyze_episode",
                return_value=recap,
            ) as analyze, patch(
                "app.tv_recap.pipeline.plan_scenes",
                return_value=recap,
            ) as plan, patch(
                "app.tv_recap.pipeline.generate_shots",
                return_value=recap,
            ) as generate, patch(
                "app.tv_recap.pipeline.assemble_recap",
                return_value={"state": const.TASK_STATE_COMPLETE},
            ) as assemble:
                result = run_recap_pipeline(
                    task_id="task-4",
                    episode=episode,
                    shot_provider="wavespeed",
                    target_duration=8,
                    min_shots=2,
                    max_shots=2,
                )

            self.assertEqual(result["task_id"], "task-4")
            self.assertTrue((episode_dir / "input.json").is_file())
            self.assertTrue((episode_dir / "recap_script.json").is_file())
            analyze.assert_called_once()
            plan.assert_called_once_with(recap, show_config=None)
            generate.assert_called_once()
            self.assertEqual(generate.call_args.kwargs["provider"], "wavespeed")
            assemble.assert_called_once()


if __name__ == "__main__":
    unittest.main()
