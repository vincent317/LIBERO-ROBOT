#!/usr/bin/env python3
"""Train the baseline multitask ACT task-id model with balanced per-task batches."""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler

from local_lerobot_act import device_from_arg, get_policy_feature_utils, save_json
from local_multitask_act import ACTTaskIDConfig, ACTTaskIDPolicy
from train_libero_act import (
    evaluate_loss,
    generate_loss_diagrams,
    get_aux_metric,
    maybe_synchronize,
    move_to_device,
    normalize_batch,
    parse_episode_spec,
    save_checkpoint,
    serialize_aux_metrics,
    set_seed,
)


class BalancedTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        task_to_indices: dict[int, list[int]],
        sample_weights: torch.Tensor,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
    ):
        self.task_ids = sorted(task_to_indices)
        if not self.task_ids:
            raise ValueError("Expected at least one task for balanced batching.")
        if batch_size % len(self.task_ids) != 0:
            raise ValueError(
                f"`batch_size` ({batch_size}) must be divisible by num tasks ({len(self.task_ids)}) "
                "for even per-task sampling."
            )

        self.per_task_batch_size = batch_size // len(self.task_ids)
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self._epoch = 0

        self.task_to_indices = {
            task_id: torch.tensor(indices, dtype=torch.long) for task_id, indices in task_to_indices.items()
        }
        self.task_to_weights = {
            task_id: sample_weights[self.task_to_indices[task_id]].to(dtype=torch.double)
            for task_id in self.task_ids
        }
        for task_id, weights in self.task_to_weights.items():
            if len(weights) == 0:
                raise ValueError(f"Task {task_id} has no samples.")
            if not torch.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError(f"Task {task_id} has invalid sampling weights.")

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.batches_per_epoch):
            batch_parts: list[torch.Tensor] = []
            for task_id in self.task_ids:
                choice = torch.multinomial(
                    self.task_to_weights[task_id],
                    self.per_task_batch_size,
                    replacement=True,
                    generator=generator,
                )
                batch_parts.append(self.task_to_indices[task_id][choice])
            batch = torch.cat(batch_parts)
            batch = batch[torch.randperm(len(batch), generator=generator)]
            yield batch.tolist()

    def __len__(self) -> int:
        return self.batches_per_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act"),
    )
    parser.add_argument("--repo-id", default="local/libero-spatial-all-tasks-act")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_taskid"),
    )
    parser.add_argument("--train-episodes", default="manifest:train")
    parser.add_argument("--val-episodes", default="manifest:val")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument("--run-name", default="act_taskbalanced_taskid")
    parser.add_argument("--gripper-oversample-weight", type=float, default=2.0)
    parser.add_argument("--gripper-transition-window", type=int, default=5)
    parser.add_argument("--num-task-ids", type=int, default=10)
    return parser.parse_args()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


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

    if transition_indices:
        weights[list(transition_indices)] = oversample_weight

    print(
        "gripper transition oversampling:"
        f" matched_samples={len(transition_indices)}/{len(dataset)}"
        f" transition_window={transition_window}"
        f" weight={oversample_weight}"
    )
    return weights


def build_task_index_map(dataset) -> dict[int, list[int]]:
    dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = dataset.hf_dataset.with_format(None)
    task_indices = raw_hf_dataset["task_index"]
    task_to_indices: dict[int, list[int]] = {}
    for sample_idx, task_index in enumerate(task_indices):
        task_to_indices.setdefault(int(task_index), []).append(sample_idx)
    return task_to_indices


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"starting training: output_dir={args.output_dir}", flush=True)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    use_amp = args.use_amp or device.type == "cuda"
    dataset_to_policy_features, NormalizerProcessorStep, _ = get_policy_feature_utils()

    print("loading datasets...", flush=True)
    train_ds, val_ds, train_eps, val_eps = build_datasets(args)
    print(f"datasets loaded: train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    policy_features = dataset_to_policy_features(train_ds.features)
    input_keys = [
        "observation.images.agentview",
        "observation.images.wrist",
        "observation.state",
    ]

    task_to_indices = build_task_index_map(train_ds)
    num_tasks_in_train = len(task_to_indices)
    if args.num_task_ids < num_tasks_in_train:
        raise SystemExit(
            f"`num_task_ids` ({args.num_task_ids}) is smaller than the number of train tasks ({num_tasks_in_train})."
        )

    config = ACTTaskIDConfig(
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
        num_task_ids=args.num_task_ids,
        use_task_id_conditioning=True,
    )
    print("initializing policy...", flush=True)
    policy = ACTTaskIDPolicy(config).to(device)
    print(f"policy initialized: device={device} use_amp={use_amp}", flush=True)

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

    sample_weights = build_gripper_oversample_weights(
        train_ds,
        args.gripper_oversample_weight,
        args.gripper_transition_window,
    )
    batches_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    train_batch_sampler = BalancedTaskBatchSampler(
        task_to_indices=task_to_indices,
        sample_weights=sample_weights,
        batch_size=args.batch_size,
        batches_per_epoch=batches_per_epoch,
        seed=args.seed,
    )
    print(
        "balanced sampler ready:"
        f" num_tasks={num_tasks_in_train}"
        f" per_task_batch={args.batch_size // num_tasks_in_train}"
        f" batches_per_epoch={batches_per_epoch}",
        flush=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
    print("entering training loop...", flush=True)

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
            "use_task_id_conditioning": True,
            "num_task_ids": args.num_task_ids,
            "num_tasks_in_train": num_tasks_in_train,
            "balanced_task_batches": True,
            "per_task_samples_per_batch": args.batch_size // num_tasks_in_train,
            "gripper_oversample_weight": args.gripper_oversample_weight,
            "gripper_transition_window": args.gripper_transition_window,
            "policy_type": ACTTaskIDPolicy.name,
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
                    f"gpu_step_s={mean_gpu_step:.4f}",
                    flush=True,
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
                print(f"saved checkpoint step={step} val_loss={val_loss:.6f}", flush=True)

    generate_loss_diagrams(args.output_dir)
    print(f"saved loss diagrams to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
