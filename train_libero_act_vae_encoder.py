#!/usr/bin/env python3
"""Train the baseline multitask ACT task-id model with task-id added only to the VAE encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from local_lerobot_act import device_from_arg, get_policy_feature_utils, save_json
from local_multitask_act_vae_encoder import ACTTaskIDVAEEncoderConfig, ACTTaskIDVAEEncoderPolicy
from train_libero_act import (
    apply_task_oversample_weights,
    build_datasets,
    build_gripper_oversample_weights,
    cycle,
    evaluate_loss,
    generate_loss_diagrams,
    get_aux_metric,
    maybe_synchronize,
    move_to_device,
    normalize_batch,
    save_checkpoint,
    serialize_aux_metrics,
    set_seed,
)


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
        default=Path("/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid_vae_encoder"),
    )
    parser.add_argument("--train-episodes", default="manifest:train")
    parser.add_argument("--val-episodes", default="manifest:val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=19000)
    parser.add_argument("--save-freq", type=int, default=100)
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
    parser.add_argument("--gripper-oversample-weight", type=float, default=2.0)
    parser.add_argument("--gripper-transition-window", type=int, default=5)
    parser.add_argument("--num-task-ids", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    use_amp = args.use_amp or device.type == "cuda"

    dataset_to_policy_features, NormalizerProcessorStep, _ = get_policy_feature_utils()
    train_ds, val_ds, train_eps, val_eps = build_datasets(args)
    policy_features = dataset_to_policy_features(train_ds.features)
    input_keys = [
        "observation.images.agentview",
        "observation.images.wrist",
        "observation.state",
    ]

    config = ACTTaskIDVAEEncoderConfig(
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
    policy = ACTTaskIDVAEEncoderPolicy(config).to(device)

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

    train_sample_weights = build_gripper_oversample_weights(
        train_ds,
        args.gripper_oversample_weight,
        args.gripper_transition_window,
    )
    train_sample_weights = apply_task_oversample_weights(train_ds, train_sample_weights, {})

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=torch.utils.data.WeightedRandomSampler(
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
            "use_task_id_conditioning": True,
            "task_id_in_vae_encoder": True,
            "num_task_ids": args.num_task_ids,
            "gripper_oversample_weight": args.gripper_oversample_weight,
            "gripper_transition_window": args.gripper_transition_window,
            "task_oversample_strategy": "NONE",
            "task_oversample_weights": {},
            "policy_type": ACTTaskIDVAEEncoderPolicy.name,
        },
        args.output_dir / "run_config.json",
    )
    save_json(train_ds.meta.stats or {}, args.output_dir / "normalization_stats.json")

    metrics_log_path = args.output_dir / "training_metrics.txt"
    with metrics_log_path.open("w", encoding="utf-8") as metrics_log:
        metrics_log.write(
            "step,total_loss,mean_recent_loss,l1_loss,kl_loss,elapsed_s,data_wait_s,host_prep_s,gpu_step_s\n"
        )

        import time
        import numpy as np

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
