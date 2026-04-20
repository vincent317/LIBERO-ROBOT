#!/usr/bin/env python3
"""Convert all LIBERO-Spatial demos into one combined local LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from libero_spatial_tasks import get_libero_spatial_tasks
from local_lerobot_act import save_json


DEFAULT_OUTPUT_ROOT = Path("/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act")
DEFAULT_REPO_ID = "local/libero-spatial-all-tasks-act"
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
STATE_NAMES = [
    "gripper_qpos_left",
    "gripper_qpos_right",
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_quat_w",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--train-episodes-per-task", type=int, default=45)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_features() -> dict[str, dict]:
    return {
        "observation.images.agentview": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
    }


def main() -> None:
    args = parse_args()
    tasks = get_libero_spatial_tasks()
    if not tasks:
        raise SystemExit("No LIBERO-Spatial tasks found.")

    for task in tasks:
        if not Path(task["demo_hdf5"]).exists():
            raise SystemExit(f"Missing demo HDF5: {task['demo_hdf5']}")
        if not Path(task["init_states_path"]).exists():
            raise SystemExit(f"Missing init states file: {task['init_states_path']}")

    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output dataset already exists: {args.output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        fps=args.fps,
        features=make_features(),
        robot_type="libero_panda",
        use_videos=False,
    )

    global_episode_index = 0
    total_frames = 0
    train_episode_indices: list[int] = []
    val_episode_indices: list[int] = []
    task_manifests: list[dict[str, object]] = []

    for task in tasks:
        demo_hdf5 = Path(task["demo_hdf5"])
        with h5py.File(demo_hdf5, "r") as h5_file:
            data_group = h5_file["data"]
            demo_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
            if args.max_episodes_per_task is not None:
                demo_names = demo_names[: args.max_episodes_per_task]

            num_train = min(args.train_episodes_per_task, len(demo_names))
            num_val = min(args.val_episodes_per_task, max(0, len(demo_names) - num_train))
            selected_demo_names = demo_names[: num_train + num_val]

            task_episode_indices: list[int] = []
            task_episode_lengths: list[int] = []
            task_train_indices: list[int] = []
            task_val_indices: list[int] = []

            for local_episode_idx, demo_name in enumerate(selected_demo_names):
                demo_group = data_group[demo_name]
                actions = np.asarray(demo_group["actions"], dtype=np.float32)
                robot_states = np.asarray(demo_group["robot_states"], dtype=np.float32)
                agentview_rgb = np.asarray(demo_group["obs"]["agentview_rgb"], dtype=np.uint8)
                wrist_rgb = np.asarray(demo_group["obs"]["eye_in_hand_rgb"], dtype=np.uint8)

                length = int(actions.shape[0])
                total_frames += length
                task_episode_indices.append(global_episode_index)
                task_episode_lengths.append(length)

                if local_episode_idx < num_train:
                    train_episode_indices.append(global_episode_index)
                    task_train_indices.append(global_episode_index)
                else:
                    val_episode_indices.append(global_episode_index)
                    task_val_indices.append(global_episode_index)

                for frame_idx in range(length):
                    dataset.add_frame(
                        {
                            "observation.images.agentview": agentview_rgb[frame_idx],
                            "observation.images.wrist": wrist_rgb[frame_idx],
                            "observation.state": robot_states[frame_idx],
                            "action": actions[frame_idx],
                            "task": task["language"],
                        }
                    )

                dataset.save_episode()
                global_episode_index += 1

        task_manifests.append(
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "language": task["language"],
                "demo_hdf5": str(task["demo_hdf5"]),
                "init_states_path": str(task["init_states_path"]),
                "episode_indices": task_episode_indices,
                "train_episode_indices": task_train_indices,
                "val_episode_indices": task_val_indices,
                "episode_lengths": task_episode_lengths,
                "num_episodes": len(task_episode_indices),
                "num_frames": int(sum(task_episode_lengths)),
            }
        )

    dataset.finalize()

    manifest = {
        "benchmark": "libero_spatial",
        "output_root": str(args.output_root),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "train_episodes_per_task": args.train_episodes_per_task,
        "val_episodes_per_task": args.val_episodes_per_task,
        "train_episode_indices": train_episode_indices,
        "val_episode_indices": val_episode_indices,
        "num_tasks": len(task_manifests),
        "num_episodes": global_episode_index,
        "total_frames": total_frames,
        "tasks": task_manifests,
    }
    save_json(manifest, args.output_root / "conversion_manifest.json")

    reloaded = LeRobotDataset(repo_id=args.repo_id, root=args.output_root)
    print(f"Converted {len(task_manifests)} tasks to {args.output_root}")
    print(f"Episodes: {global_episode_index}")
    print(f"Frames: {reloaded.num_frames}")
    print(f"Tasks: {reloaded.meta.info['total_tasks']}")


if __name__ == "__main__":
    main()
