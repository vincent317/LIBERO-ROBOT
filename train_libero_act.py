#!/usr/bin/env python3
"""Train LeRobot ACT on the converted LIBERO dataset."""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from local_lerobot_act import (
    device_from_arg,
    get_policy_classes,
    get_policy_feature_utils,
    save_json,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="local/libero-black-bowl-plate-ramekin-act")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-episodes", default="0:45")
    parser.add_argument("--val-episodes", default="45:50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--pretrained-backbone-weights", default=None)
    parser.add_argument("--run-name", default="act_resnet18")
    parser.add_argument("--gripper-oversample-weight", type=float, default=2.0)
    parser.add_argument("--gripper-transition-window", type=int, default=5)
    parser.add_argument(
        "--task-oversample-weights",
        default="4:4,5:8,9:4",
        help="Comma-separated task_index:weight multipliers applied in the WeightedRandomSampler.",
    )
    parser.add_argument("--use-task-id-conditioning", action="store_true")
    parser.add_argument("--num-task-ids", type=int, default=1)
    return parser.parse_args()


def parse_episode_spec(spec: str, dataset_root: Path) -> list[int]:
    spec = spec.strip()
    if spec.startswith("manifest:"):
        split_name = spec.split(":", 1)[1]
        manifest_path = dataset_root / "conversion_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Manifest split requested but file not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = f"{split_name}_episode_indices"
        if key not in manifest:
            raise SystemExit(f"Manifest split not found: {key}")
        return [int(idx) for idx in manifest[key]]

    episode_indices: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start, end = chunk.split(":", 1)
            episode_indices.extend(range(int(start), int(end)))
        else:
            episode_indices.append(int(chunk))
    return episode_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_datasets(args: argparse.Namespace):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hf_transform_to_torch

    train_eps = parse_episode_spec(args.train_episodes, args.dataset_root)
    val_eps = parse_episode_spec(args.val_episodes, args.dataset_root)
    delta_timestamps = {"action": [i / 20.0 for i in range(args.chunk_size)]}

    @contextlib.contextmanager
    def raw_load_during_init():
        original_load_hf_dataset = LeRobotDataset.load_hf_dataset

        def load_hf_dataset_without_transform(self):
            hf_dataset = original_load_hf_dataset(self)
            hf_dataset.reset_format()
            return hf_dataset

        LeRobotDataset.load_hf_dataset = load_hf_dataset_without_transform
        try:
            yield
        finally:
            LeRobotDataset.load_hf_dataset = original_load_hf_dataset

    with raw_load_during_init():
        train_ds = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.dataset_root,
            episodes=train_eps,
            delta_timestamps=delta_timestamps,
        )
        val_ds = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.dataset_root,
            episodes=val_eps,
            delta_timestamps=delta_timestamps,
        )

    train_ds.hf_dataset.set_transform(hf_transform_to_torch)
    val_ds.hf_dataset.set_transform(hf_transform_to_torch)
    return train_ds, val_ds, train_eps, val_eps


def normalize_batch(normalizer, batch: dict[str, torch.Tensor], input_keys: list[str]) -> dict[str, torch.Tensor]:
    observation = {key: batch[key] for key in input_keys}
    normalized_observation = normalizer._normalize_observation(observation, inverse=False)
    action = batch["action"]
    normalized_action = normalizer._normalize_action(action, inverse=False)
    normalized_batch = {
        **normalized_observation,
        "action": normalized_action,
        "action_is_pad": batch["action_is_pad"],
    }
    if "task_index" in batch:
        normalized_batch["task_index"] = batch["task_index"]
    return normalized_batch


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_gripper_oversample_weights(
    dataset,
    oversample_weight: float,
    transition_window: int,
) -> torch.Tensor:
    weights = torch.ones(len(dataset), dtype=torch.double)
    if oversample_weight <= 1.0:
        return weights

    dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = dataset.hf_dataset.with_format(None)
    episode_indices = raw_hf_dataset["episode_index"]
    actions_column = raw_hf_dataset["action"]
    gripper_values = torch.as_tensor(np.stack(actions_column)[:, -1])

    transition_indices: set[int] = set()
    for rel_idx in range(len(dataset) - 1):
        if int(episode_indices[rel_idx]) != int(episode_indices[rel_idx + 1]):
            continue
        current_gripper = gripper_values[rel_idx].item()
        next_gripper = gripper_values[rel_idx + 1].item()
        if current_gripper == next_gripper:
            continue
        # Oversample both grasp and release events: open (-1) <-> close (1).
        if not (
            (current_gripper == -1.0 and next_gripper == 1.0)
            or (current_gripper == 1.0 and next_gripper == -1.0)
        ):
            continue
        episode_idx = int(episode_indices[rel_idx])
        start = max(0, rel_idx - transition_window)
        end = min(len(dataset), rel_idx + 2 + transition_window)
        for sample_idx in range(start, end):
            if int(episode_indices[sample_idx]) == episode_idx:
                transition_indices.add(sample_idx)

    matched = len(transition_indices)
    if matched > 0:
        weights[list(transition_indices)] = oversample_weight

    print(
        "gripper transition oversampling:"
        f" matched_samples={matched}/{len(dataset)}"
        f" transition_window={transition_window}"
        f" weight={oversample_weight}"
    )

    return weights


def parse_task_oversample_weights(spec: str) -> dict[int, float]:
    spec = spec.strip()
    if not spec:
        return {}

    weights: dict[int, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(
                f"Invalid task oversample entry '{chunk}'. Expected comma-separated task_id:weight pairs."
            )
        task_id_str, weight_str = chunk.split(":", 1)
        task_id = int(task_id_str.strip())
        weight = float(weight_str.strip())
        if weight <= 0:
            raise SystemExit(f"Task oversample weight must be > 0 for task {task_id}, got {weight}.")
        weights[task_id] = weight
    return weights


def apply_task_oversample_weights(dataset, sample_weights: torch.Tensor, task_weight_map: dict[int, float]) -> torch.Tensor:
    if not task_weight_map:
        return sample_weights

    dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = dataset.hf_dataset.with_format(None)
    task_indices = raw_hf_dataset["task_index"]

    matched_counts = {task_id: 0 for task_id in task_weight_map}
    updated = sample_weights.clone()
    for sample_idx, task_index in enumerate(task_indices):
        task_id = int(task_index)
        task_weight = task_weight_map.get(task_id)
        if task_weight is None:
            continue
        updated[sample_idx] *= task_weight
        matched_counts[task_id] += 1

    print("task oversampling:")
    for task_id in sorted(task_weight_map):
        print(
            f"  task_id={task_id} weight={task_weight_map[task_id]}"
            f" matched_samples={matched_counts[task_id]}/{len(updated)}"
        )

    return updated


@torch.no_grad()
def evaluate_loss(policy, normalizer, loader, device: torch.device, input_keys: list[str], limit: int = 20) -> float:
    policy.train()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= limit:
            break
        batch = move_to_device(batch, device)
        normalized_batch = normalize_batch(normalizer, batch, input_keys)
        loss, _ = policy.forward(normalized_batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


def save_checkpoint(policy, output_dir: Path, step: int, metadata: dict) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    save_json(metadata, checkpoint_dir / "run_metadata.json")


def write_loss_svg(
    out_path: Path,
    title: str,
    subtitle: str,
    series: list[tuple[str, list[int], list[float], str, float]],
) -> None:
    all_x = [x for _, xs, _, _, _ in series for x in xs]
    all_y = [max(y, 1e-6) for _, _, ys, _, _ in series for y in ys if not math.isnan(y)]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    log_min_y, log_max_y = math.log10(min_y), math.log10(max_y)

    width, height = 1200, 760
    left, right, top, bottom = 90, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    bg = "#f7f4ea"
    axis = "#1f2933"
    grid = "#d9d2bf"
    text = "#222222"

    def sx(x: int) -> float:
        return left + (0 if max_x == min_x else (x - min_x) / (max_x - min_x) * plot_w)

    def sy(y: float) -> float:
        y = max(y, 1e-6)
        log_y = math.log10(y)
        return top + plot_h - (0 if log_max_y == log_min_y else (log_y - log_min_y) / (log_max_y - log_min_y) * plot_h)

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'<rect width="100%" height="100%" fill="{bg}"/>')
    parts.append(f'<text x="{left}" y="38" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="{text}">{title}</text>')
    parts.append(f'<text x="{left}" y="58" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="{text}">{subtitle}</text>')

    for i in range(6):
        t = i / 5
        log_y = log_min_y + (log_max_y - log_min_y) * t
        yv = 10 ** log_y
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{text}">{yv:.3g}</text>')
    for i in range(7):
        xv = min_x + (max_x - min_x) * i / 6
        x = sx(int(round(xv)))
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{text}">{int(round(xv))}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{axis}" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{axis}" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="{text}">Training Step</text>')
    parts.append(f'<text x="24" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="{text}" transform="rotate(-90 24 {top + plot_h / 2:.2f})">Loss (log scale)</text>')

    for label, xs, ys, color, stroke_w in series:
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys) if not math.isnan(y))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_w}" points="{pts}"/>')
        if "Validation" in label:
            for x, y in zip(xs, ys):
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.3" fill="{color}"/>')

    legend_x = left + plot_w - 230
    legend_y = top + 16
    legend_h = 36 + 24 * len(series)
    parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 18}" width="240" height="{legend_h}" rx="10" fill="#fffdf7" stroke="{grid}"/>')
    for idx, (label, _, _, color, _) in enumerate(series):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 40}" y="{y + 5}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="{text}">{label}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def smooth_series(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values[:]

    smoothed: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        smoothed.append(running_sum / min(idx + 1, window))
    return smoothed


def generate_loss_diagrams(output_dir: Path) -> None:
    metrics_path = output_dir / "training_metrics.txt"
    checkpoint_root = output_dir / "checkpoints"
    if not metrics_path.exists() or not checkpoint_root.exists():
        return

    steps: list[int] = []
    train_loss: list[float] = []
    l1_loss: list[float] = []
    kl_loss: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_loss.append(float(row["total_loss"]))
            l1_loss.append(float(row["l1_loss"]))
            kl_loss.append(float(row["kl_loss"]))

    val_steps: list[int] = []
    val_loss: list[float] = []
    for metadata_path in sorted(checkpoint_root.glob("step_*/run_metadata.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        val_steps.append(int(data["step"]))
        val_loss.append(float(data["val_loss"]))

    smoothing_window = max(25, min(250, len(steps) // 100))
    train_loss_smoothed = smooth_series(train_loss, smoothing_window)
    l1_loss_smoothed = smooth_series(l1_loss, smoothing_window)
    kl_loss_smoothed = smooth_series(kl_loss, smoothing_window)

    write_loss_svg(
        output_dir / "train_vs_val_loss_log.svg",
        "Training vs Validation Loss",
        f"Log-scale y-axis, plotted against step. Training curve smoothed with trailing window={smoothing_window}.",
        [
            ("Training Loss", steps, train_loss_smoothed, "#264653", 1.8),
            ("Validation Loss", val_steps, val_loss, "#e76f51", 2.4),
        ],
    )
    write_loss_svg(
        output_dir / "l1_vs_kl_loss_log.svg",
        "L1 vs KL Loss",
        f"Log-scale y-axis, plotted against step. Curves smoothed with trailing window={smoothing_window}.",
        [
            ("L1 Loss", steps, l1_loss_smoothed, "#2a9d8f", 1.8),
            ("KL Loss", steps, kl_loss_smoothed, "#e9c46a", 1.8),
        ],
    )


def get_aux_metric(aux: dict, *keys: str) -> float:
    for key in keys:
        value = aux.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return math.nan


def serialize_aux_metrics(aux: dict) -> dict[str, float]:
    serialized: dict[str, float] = {}
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = float(value.detach().item())
            continue
        try:
            serialized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return serialized


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    use_amp = args.use_amp or device.type == "cuda"

    ACTConfig, ACTPolicy = get_policy_classes(args.use_task_id_conditioning)
    dataset_to_policy_features, NormalizerProcessorStep, _ = get_policy_feature_utils()

    train_ds, val_ds, train_eps, val_eps = build_datasets(args)
    policy_features = dataset_to_policy_features(train_ds.features)
    input_keys = [
        "observation.images.agentview",
        "observation.images.wrist",
        "observation.state",
    ]

    config_kwargs = dict(
        input_features={key: policy_features[key] for key in input_keys},
        output_features={"action": policy_features["action"]},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone="resnet18",
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        device=str(device),
        use_amp=use_amp,
        push_to_hub=False,
        kl_weight=args.kl_weight,
        optimizer_lr=args.learning_rate,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.learning_rate,
    )
    if args.use_task_id_conditioning:
        config_kwargs["num_task_ids"] = args.num_task_ids
        config_kwargs["use_task_id_conditioning"] = True
    config = ACTConfig(**config_kwargs)
    policy = ACTPolicy(config).to(device)

    normalizer = NormalizerProcessorStep.from_lerobot_dataset(
        train_ds,
        features=config.input_features | config.output_features,
        norm_map=config.normalization_mapping,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    task_oversample_weights = parse_task_oversample_weights(args.task_oversample_weights)
    train_sample_weights = build_gripper_oversample_weights(
        train_ds,
        args.gripper_oversample_weight,
        args.gripper_transition_window,
    )
    train_sample_weights = apply_task_oversample_weights(
        train_ds,
        train_sample_weights,
        task_oversample_weights,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=WeightedRandomSampler(
            train_sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_iter = cycle(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    save_json(
        {
            "dataset_root": str(args.dataset_root),
            "repo_id": args.repo_id,
            "train_episodes": train_eps,
            "val_episodes": val_eps,
            "device": str(device),
            "use_amp": use_amp,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "n_action_steps": args.n_action_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "kl_weight": args.kl_weight,
            "seed": args.seed,
            "input_keys": input_keys,
            "pretrained_backbone_weights": args.pretrained_backbone_weights,
            "use_task_id_conditioning": args.use_task_id_conditioning,
            "num_task_ids": args.num_task_ids,
            "gripper_oversample_weight": args.gripper_oversample_weight,
            "gripper_transition_window": args.gripper_transition_window,
            "task_oversample_strategy": "OVERSAMPLE_BAD_TASK" if task_oversample_weights else "NONE",
            "task_oversample_weights": task_oversample_weights,
        },
        args.output_dir / "run_config.json",
    )
    save_json(train_ds.meta.stats or {}, args.output_dir / "normalization_stats.json")
    metrics_log_path = args.output_dir / "training_metrics.txt"
    with metrics_log_path.open("w", encoding="utf-8") as metrics_log:
        metrics_log.write(
            "step,total_loss,mean_recent_loss,l1_loss,kl_loss,elapsed_s,data_wait_s,host_prep_s,gpu_step_s\n"
        )

        start_time = time.time()
        recent_losses: list[float] = []
        recent_data_wait_s: list[float] = []
        recent_host_prep_s: list[float] = []
        recent_gpu_step_s: list[float] = []

        for step in range(1, args.steps + 1):
            data_wait_start = time.perf_counter()
            batch = next(train_iter)
            data_wait_s = time.perf_counter() - data_wait_start

            host_prep_start = time.perf_counter()
            batch = move_to_device(batch, device)
            normalized_batch = normalize_batch(normalizer, batch, input_keys)
            maybe_synchronize(device)
            host_prep_s = time.perf_counter() - host_prep_start

            gpu_step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                loss, aux = policy.forward(normalized_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            maybe_synchronize(device)
            gpu_step_s = time.perf_counter() - gpu_step_start

            loss_value = float(loss.item())
            aux_metrics = serialize_aux_metrics(aux)
            l1_loss_value = get_aux_metric(aux_metrics, "l1_loss", "l1")
            kl_loss_value = get_aux_metric(aux_metrics, "kl_loss", "kld_loss", "kld", "kl")
            recent_losses.append(loss_value)
            recent_data_wait_s.append(data_wait_s)
            recent_host_prep_s.append(host_prep_s)
            recent_gpu_step_s.append(gpu_step_s)
            if len(recent_losses) > args.log_freq:
                recent_losses.pop(0)
                recent_data_wait_s.pop(0)
                recent_host_prep_s.pop(0)
                recent_gpu_step_s.pop(0)

            mean_loss = float(np.mean(recent_losses))
            elapsed = time.time() - start_time
            mean_data_wait = float(np.mean(recent_data_wait_s))
            mean_host_prep = float(np.mean(recent_host_prep_s))
            mean_gpu_step = float(np.mean(recent_gpu_step_s))
            metrics_log.write(
                f"{step},{loss_value:.6f},{mean_loss:.6f},{l1_loss_value:.6f},"
                f"{kl_loss_value:.6f},{elapsed:.1f},{mean_data_wait:.4f},"
                f"{mean_host_prep:.4f},{mean_gpu_step:.4f}\n"
            )
            metrics_log.flush()

            if step % args.log_freq == 0 or step == 1:
                print(
                    f"step={step} loss={loss_value:.6f} mean_loss={mean_loss:.6f} "
                    f"l1={l1_loss_value:.6f} kl={kl_loss_value:.6f} elapsed_s={elapsed:.1f} "
                    f"data_wait_s={mean_data_wait:.4f} "
                    f"host_prep_s={mean_host_prep:.4f} "
                    f"gpu_step_s={mean_gpu_step:.4f}"
                )

            if step % args.save_freq == 0 or step == args.steps:
                val_loss = evaluate_loss(policy, normalizer, val_loader, device, input_keys)
                metadata = {
                    "step": step,
                    "train_loss": loss_value,
                    "mean_recent_train_loss": mean_loss,
                    "val_loss": val_loss,
                    "elapsed_s": elapsed,
                    "aux_metrics": aux_metrics,
                    "l1_loss": l1_loss_value,
                    "kl_loss": kl_loss_value,
                }
                save_checkpoint(policy, args.output_dir, step, metadata)
                print(f"saved checkpoint step={step} val_loss={val_loss:.6f}")

    generate_loss_diagrams(args.output_dir)
    print(f"saved loss diagrams to {args.output_dir}")


if __name__ == "__main__":
    main()
