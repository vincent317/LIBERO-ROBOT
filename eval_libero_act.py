#!/usr/bin/env python3
"""Evaluate a locally trained ACT checkpoint in LIBERO."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from local_lerobot_act import (
    device_from_arg,
    get_policy_classes,
    get_policy_classes_for_checkpoint,
    get_policy_feature_utils,
    serialize_jsonable,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act"
)
DEFAULT_CHECKPOINT = Path(
    "/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin/checkpoints/step_050000"
)
DEFAULT_OUTPUT_DIR = Path("/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="local/libero-black-bowl-plate-ramekin-act")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-episode", type=int, default=0)
    parser.add_argument("--video-camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--video-scale", type=int, default=1)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument(
        "--initial-noop-steps",
        type=int,
        default=0,
        help="Use an all-zero action for the first N environment steps of each episode.",
    )
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
    parser.add_argument(
        "--task-text",
        default=None,
        help="Optional language instruction override. Defaults to the selected LIBERO task language.",
    )
    parser.add_argument("--use-task-id-conditioning", action="store_true")
    parser.add_argument("--task-index", type=int, default=0)
    return parser.parse_args()


def prepare_env(task_id: int, render_height: int, render_width: int):
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark("libero_spatial")()
    task = benchmark.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=render_height,
        camera_widths=render_width,
    )
    return env, task, init_states


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def make_observation(env, obs: dict[str, np.ndarray], policy_image_size: int = 128) -> dict[str, np.ndarray]:
    return {
        "observation.images.agentview": resize_image(obs["agentview_image"], policy_image_size, policy_image_size),
        "observation.images.wrist": resize_image(
            obs["robot0_eye_in_hand_image"], policy_image_size, policy_image_size
        ),
        "observation.state": env.env.get_robot_state_vector(obs).astype(np.float32),
    }


def get_video_frame(obs: dict[str, np.ndarray], camera: str) -> np.ndarray:
    key = "agentview_image" if camera == "agentview" else "robot0_eye_in_hand_image"
    return np.asarray(obs[key], dtype=np.uint8)


def maybe_upscale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def now_s() -> float:
    return time.perf_counter()


def maybe_sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_timing_stats() -> dict[str, float]:
    return {
        "episode_total_s": 0.0,
        "observation_s": 0.0,
        "prepare_observation_s": 0.0,
        "normalize_s": 0.0,
        "policy_inference_s": 0.0,
        "unnormalize_action_s": 0.0,
        "env_step_s": 0.0,
        "success_check_s": 0.0,
        "video_frame_s": 0.0,
        "video_encode_s": 0.0,
        "steps": 0.0,
    }


def accumulate_timing_stats(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def timing_summary_line(prefix: str, timing_stats: dict[str, float]) -> str:
    steps = max(1.0, timing_stats.get("steps", 0.0))
    total = timing_stats.get("episode_total_s", 0.0)
    per_step_ms = total * 1000.0 / steps
    return (
        f"{prefix} total_s={total:.3f} steps={int(timing_stats.get('steps', 0.0))} "
        f"per_step_ms={per_step_ms:.2f} "
        f"obs_ms={timing_stats.get('observation_s', 0.0) * 1000.0 / steps:.2f} "
        f"prep_ms={timing_stats.get('prepare_observation_s', 0.0) * 1000.0 / steps:.2f} "
        f"norm_ms={timing_stats.get('normalize_s', 0.0) * 1000.0 / steps:.2f} "
        f"infer_ms={timing_stats.get('policy_inference_s', 0.0) * 1000.0 / steps:.2f} "
        f"unnorm_ms={timing_stats.get('unnormalize_action_s', 0.0) * 1000.0 / steps:.2f} "
        f"env_ms={timing_stats.get('env_step_s', 0.0) * 1000.0 / steps:.2f} "
        f"success_ms={timing_stats.get('success_check_s', 0.0) * 1000.0 / steps:.2f} "
        f"video_frame_ms={timing_stats.get('video_frame_s', 0.0) * 1000.0 / steps:.2f} "
        f"video_encode_s={timing_stats.get('video_encode_s', 0.0):.3f}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint.exists():
        ACTConfig, ACTPolicy = get_policy_classes_for_checkpoint(args.checkpoint)
    else:
        ACTConfig, ACTPolicy = get_policy_classes(args.use_task_id_conditioning)
    dataset_to_policy_features, NormalizerProcessorStep, UnnormalizerProcessorStep = get_policy_feature_utils()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.utils import prepare_observation_for_inference

    device = device_from_arg(args.device)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    policy_features = dataset_to_policy_features(dataset.features)

    policy = ACTPolicy.from_pretrained(args.checkpoint)
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

    env, task, init_states = prepare_env(args.task_id, args.render_height, args.render_width)
    task_text = args.task_text or task.language
    num_episodes = min(args.num_episodes, len(init_states))
    episode_results: list[dict[str, object]] = []
    aggregate_timing = empty_timing_stats()

    for episode_idx in range(num_episodes):
        episode_start = now_s()
        episode_timing = empty_timing_stats()
        policy.reset()
        env.reset()
        obs = env.set_init_state(init_states[episode_idx])
        done = False
        success = False
        should_record = args.save_failure_videos or (
            args.save_video and episode_idx == args.video_episode
        )
        video_frames: list[np.ndarray] = []
        if should_record:
            video_frames = [maybe_upscale_frame(get_video_frame(obs, args.video_camera), args.video_scale)]

        for step in range(args.max_steps):
            t0 = now_s()
            raw_observation = make_observation(env, obs)
            episode_timing["observation_s"] += now_s() - t0

            t0 = now_s()
            obs_tensors = prepare_observation_for_inference(
                raw_observation, device, task_text, "libero_panda"
            )
            episode_timing["prepare_observation_s"] += now_s() - t0
            if args.use_task_id_conditioning:
                obs_tensors["task_index"] = torch.tensor([args.task_index], device=device, dtype=torch.long)

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

            action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if step < args.initial_noop_steps:
                action_np = np.zeros_like(action_np)

            t0 = now_s()
            obs, reward, done, info = env.step(action_np)
            episode_timing["env_step_s"] += now_s() - t0

            if should_record:
                t0 = now_s()
                video_frames.append(maybe_upscale_frame(get_video_frame(obs, args.video_camera), args.video_scale))
                episode_timing["video_frame_s"] += now_s() - t0

            t0 = now_s()
            success = bool(done or env.check_success())
            episode_timing["success_check_s"] += now_s() - t0
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
        print(f"episode={episode_idx} success={int(success)} steps={step + 1}")
        print(timing_summary_line(f"episode={episode_idx} timing", episode_timing))

        should_save_video = (
            (args.save_video and episode_idx == args.video_episode)
            or (args.save_failure_videos and not success)
        )
        if should_record and should_save_video and video_frames:
            t0 = now_s()
            suffix = "failure" if not success else args.video_camera
            video_path = args.output_dir / f"episode_{episode_idx:03d}_{suffix}_{args.video_camera}.mp4"
            imageio.mimsave(video_path, video_frames, fps=20)
            episode_timing["video_encode_s"] += now_s() - t0
            print(f"saved_video={video_path}")
            print(timing_summary_line(f"episode={episode_idx} timing_post_video", episode_timing))

        accumulate_timing_stats(aggregate_timing, episode_timing)

    env.close()

    success_rate = float(sum(int(item["success"]) for item in episode_results) / len(episode_results))
    results = {
        "task_name": task.name,
        "task_language": task.language,
        "task_text_used": task_text,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_episodes": len(episode_results),
        "initial_noop_steps": args.initial_noop_steps,
        "success_rate": success_rate,
        "timing": aggregate_timing,
        "episodes": episode_results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(serialize_jsonable(results), indent=2),
        encoding="utf-8",
    )
    print(f"success_rate={success_rate:.4f}")
    print(timing_summary_line("eval_timing", aggregate_timing))


if __name__ == "__main__":
    main()
