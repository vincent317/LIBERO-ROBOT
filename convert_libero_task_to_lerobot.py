#!/usr/bin/env python3
"""Convert one LIBERO demo HDF5 into a local LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from local_lerobot_act import save_json


DEFAULT_SOURCE_HDF5 = Path(
    "/home/ec2-user/LIBERO/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act"
)
TASK_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
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
    parser.add_argument("--source-hdf5", type=Path, default=DEFAULT_SOURCE_HDF5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default="local/libero-black-bowl-plate-ramekin-act")
    parser.add_argument("--task-text", default=TASK_TEXT)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None)
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
    if not args.source_hdf5.exists():
        raise SystemExit(f"Source demo file not found: {args.source_hdf5}")

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

    episode_lengths: list[int] = []

    with h5py.File(args.source_hdf5, "r") as h5_file:
        data_group = h5_file["data"]
        demo_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
        if args.max_episodes is not None:
            demo_names = demo_names[: args.max_episodes]

        for demo_name in demo_names:
            demo_group = data_group[demo_name]
            actions = np.asarray(demo_group["actions"], dtype=np.float32)
            robot_states = np.asarray(demo_group["robot_states"], dtype=np.float32)
            agentview_rgb = np.asarray(demo_group["obs"]["agentview_rgb"], dtype=np.uint8)
            wrist_rgb = np.asarray(demo_group["obs"]["eye_in_hand_rgb"], dtype=np.uint8)

            length = int(actions.shape[0])
            episode_lengths.append(length)

            for frame_idx in range(length):
                frame = {
                    "observation.images.agentview": agentview_rgb[frame_idx],
                    "observation.images.wrist": wrist_rgb[frame_idx],
                    "observation.state": robot_states[frame_idx],
                    "action": actions[frame_idx],
                    "task": args.task_text,
                }
                dataset.add_frame(frame)

            dataset.save_episode()

    dataset.finalize()

    manifest = {
        "source_hdf5": str(args.source_hdf5),
        "output_root": str(args.output_root),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task_text": args.task_text,
        "num_episodes": len(episode_lengths),
        "total_frames": int(sum(episode_lengths)),
        "episode_lengths": episode_lengths,
    }
    save_json(manifest, args.output_root / "conversion_manifest.json")

    reloaded = LeRobotDataset(repo_id=args.repo_id, root=args.output_root)
    print(f"Converted {len(episode_lengths)} episodes to {args.output_root}")
    print(f"Frames: {reloaded.num_frames}")
    print(f"Features: {list(reloaded.features.keys())}")


if __name__ == "__main__":
    main()
