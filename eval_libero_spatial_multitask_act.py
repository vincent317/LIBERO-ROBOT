#!/usr/bin/env python3
"""Evaluate one ACT checkpoint across all LIBERO-Spatial tasks."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from libero_spatial_tasks import get_libero_spatial_tasks
from local_lerobot_act import get_policy_classes_for_checkpoint, serialize_jsonable


DEFAULT_DATASET_ROOT = Path("/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act")
DEFAULT_CHECKPOINT = Path(
    "/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5/checkpoints/step_019000"
)
DEFAULT_OUTPUT_DIR = Path("/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="local/libero-spatial-all-tasks-act")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--video-scale", type=int, default=1)
    parser.add_argument(
        "--save-failure-videos",
        dest="save_failure_videos",
        action="store_true",
        default=True,
        help="Record and save videos for failed episodes. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-failure-videos",
        dest="save_failure_videos",
        action="store_false",
        help="Disable failure video recording.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _chunk_task_specs(task_specs: list[dict[str, object]], num_workers: int) -> list[list[dict[str, object]]]:
    worker_count = max(1, min(num_workers, len(task_specs)))
    shards: list[list[dict[str, object]]] = [[] for _ in range(worker_count)]
    for index, task_spec in enumerate(task_specs):
        shards[index % worker_count].append(task_spec)
    return [shard for shard in shards if shard]


def _evaluate_task_shard(job: dict[str, object]) -> list[dict[str, object]]:
    task_specs = list(job["task_specs"])
    checkpoint = Path(str(job["checkpoint"]))
    dataset_root = Path(str(job["dataset_root"]))
    output_dir = Path(str(job["output_dir"]))
    repo_id = str(job["repo_id"])
    num_episodes_per_task = int(job["num_episodes_per_task"])
    max_steps = int(job["max_steps"])
    save_failure_videos = bool(job["save_failure_videos"])
    render_height = int(job["render_height"])
    render_width = int(job["render_width"])
    video_scale = int(job["video_scale"])
    device_arg = job["device"]

    from eval_libero_act import (
        accumulate_timing_stats,
        empty_timing_stats,
        make_observation,
        maybe_sync_device,
        maybe_upscale_frame,
        now_s,
        prepare_env,
        timing_summary_line,
    )
    from local_lerobot_act import device_from_arg, get_policy_feature_utils
    import imageio.v2 as imageio
    import torch

    device = device_from_arg(device_arg)
    _, ACTPolicy = get_policy_classes_for_checkpoint(checkpoint)
    dataset_to_policy_features, NormalizerProcessorStep, UnnormalizerProcessorStep = get_policy_feature_utils()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.utils import prepare_observation_for_inference

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.reset()

    normalizer = NormalizerProcessorStep.from_lerobot_dataset(
        dataset,
        features=policy.config.input_features | policy.config.output_features,
        norm_map=policy.config.normalization_mapping,
        device=device,
    )
    unnormalizer = UnnormalizerProcessorStep.from_lerobot_dataset(
        dataset,
        features=policy.config.input_features | policy.config.output_features,
        norm_map=policy.config.normalization_mapping,
        device=device,
    )

    task_summaries: list[dict[str, object]] = []
    for task_spec in task_specs:
        task_id = int(task_spec["task_id"])
        task_output_dir = output_dir / f"task_{task_id:02d}_{task_spec['task_name']}"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        env, task, init_states = prepare_env(task_id, render_height, render_width)
        task_text = str(task_spec["language"])
        task_index_tensor = torch.tensor([task_id], device=device, dtype=torch.long)
        num_episodes = min(num_episodes_per_task, len(init_states))
        episode_results: list[dict[str, object]] = []
        task_timing = empty_timing_stats()

        for episode_idx in range(num_episodes):
            episode_start = now_s()
            episode_timing = empty_timing_stats()
            policy.reset()
            env.reset()
            obs = env.set_init_state(init_states[episode_idx])
            success = False
            video_frames: list[object] = []

            for step in range(max_steps):
                t0 = now_s()
                raw_observation = make_observation(env, obs)
                episode_timing["observation_s"] += now_s() - t0

                t0 = now_s()
                obs_tensors = prepare_observation_for_inference(
                    raw_observation, device, task_text, "libero_panda"
                )
                episode_timing["prepare_observation_s"] += now_s() - t0
                obs_tensors["task_index"] = task_index_tensor

                t0 = now_s()
                obs_tensors = normalizer._normalize_observation(obs_tensors, inverse=False)
                episode_timing["normalize_s"] += now_s() - t0

                maybe_sync_device(device)
                t0 = now_s()
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    normalized_action = policy.select_action(obs_tensors)
                maybe_sync_device(device)
                episode_timing["policy_inference_s"] += now_s() - t0

                t0 = now_s()
                action = unnormalizer._normalize_action(normalized_action, inverse=True)
                episode_timing["unnormalize_action_s"] += now_s() - t0

                action_np = action.squeeze(0).detach().cpu().numpy().astype("float32")

                t0 = now_s()
                obs, _, done, _ = env.step(action_np)
                episode_timing["env_step_s"] += now_s() - t0

                t0 = now_s()
                success = bool(done or env.check_success())
                episode_timing["success_check_s"] += now_s() - t0
                if save_failure_videos:
                    t0 = now_s()
                    video_frames.append(maybe_upscale_frame(obs["agentview_image"], video_scale))
                    episode_timing["video_frame_s"] += now_s() - t0
                if success:
                    break
            episode_timing["steps"] = float(step + 1)
            episode_timing["episode_total_s"] = now_s() - episode_start

            episode_results.append(
                {
                    "episode_index": episode_idx,
                    "success": success,
                    "steps": step + 1,
                    "timing": episode_timing,
                }
            )
            print(f"task={task_id} episode={episode_idx} success={int(success)} steps={step + 1}")
            print(timing_summary_line(f"task={task_id} episode={episode_idx} timing", episode_timing))

            if save_failure_videos and not success and video_frames:
                t0 = now_s()
                video_path = task_output_dir / f"episode_{episode_idx:03d}_failure_agentview.mp4"
                imageio.mimsave(video_path, video_frames, fps=20)
                episode_timing["video_encode_s"] += now_s() - t0
                print(timing_summary_line(f"task={task_id} episode={episode_idx} timing_post_video", episode_timing))

            accumulate_timing_stats(task_timing, episode_timing)

        env.close()

        success_rate = float(sum(int(item["success"]) for item in episode_results) / len(episode_results))
        task_metrics = {
            "task_id": task_id,
            "task_name": task.name,
            "task_language": task.language,
            "task_text_used": task_text,
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "num_episodes": len(episode_results),
            "success_rate": success_rate,
            "timing": task_timing,
            "episodes": episode_results,
        }
        (task_output_dir / "metrics.json").write_text(
            json.dumps(serialize_jsonable(task_metrics), indent=2),
            encoding="utf-8",
        )
        task_summaries.append(task_metrics)
        print(f"task={task_id} success_rate={success_rate:.4f}")
        print(timing_summary_line(f"task={task_id} timing", task_timing))

    return task_summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_specs = get_libero_spatial_tasks()
    if args.num_workers <= 1:
        task_summaries = _evaluate_task_shard(
            {
                "task_specs": task_specs,
                "checkpoint": str(args.checkpoint),
                "dataset_root": str(args.dataset_root),
                "output_dir": str(args.output_dir),
                "repo_id": args.repo_id,
                "num_episodes_per_task": args.num_episodes_per_task,
                "max_steps": args.max_steps,
                "save_failure_videos": args.save_failure_videos,
                "render_height": args.render_height,
                "render_width": args.render_width,
                "video_scale": args.video_scale,
                "device": args.device,
            }
        )
    else:
        jobs = [
            {
                "task_specs": shard,
                "checkpoint": str(args.checkpoint),
                "dataset_root": str(args.dataset_root),
                "output_dir": str(args.output_dir),
                "repo_id": args.repo_id,
                "num_episodes_per_task": args.num_episodes_per_task,
                "max_steps": args.max_steps,
                "save_failure_videos": args.save_failure_videos,
                "render_height": args.render_height,
                "render_width": args.render_width,
                "video_scale": args.video_scale,
                "device": args.device,
            }
            for shard in _chunk_task_specs(task_specs, args.num_workers)
        ]
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            worker_results = pool.map(_evaluate_task_shard, jobs)
        task_summaries = [item for shard in worker_results for item in shard]

    task_summaries.sort(key=lambda item: int(item["task_id"]))
    aggregate = {
        "benchmark": "libero_spatial",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_tasks": len(task_summaries),
        "mean_success_rate": float(sum(item["success_rate"] for item in task_summaries) / len(task_summaries)),
        "tasks": task_summaries,
    }
    (args.output_dir / "aggregate_metrics.json").write_text(
        json.dumps(serialize_jsonable(aggregate), indent=2),
        encoding="utf-8",
    )
    print(f"mean_success_rate={aggregate['mean_success_rate']:.4f}")


if __name__ == "__main__":
    main()
