# LIBERO ACT Worklog

## Goal

Train LeRobot ACT on LIBERO task:

`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`

## What Was Implemented

- `convert_libero_task_to_lerobot.py`
  - Converts the shipped LIBERO HDF5 demos into a local LeRobot dataset.
  - Uses:
    - `obs/agentview_rgb` -> `observation.images.agentview`
    - `obs/eye_in_hand_rgb` -> `observation.images.wrist`
    - `robot_states` -> `observation.state`
    - `actions` -> `action`
  - Writes image-backed parquet data instead of video-backed data.

- `train_libero_act.py`
  - Local ACT-only trainer.
  - Avoids the broken top-level LeRobot trainer import path under Python 3.12.

- `eval_libero_act.py`
  - Evaluates a local ACT checkpoint in LIBERO.
  - Saves metrics and optional rollout videos.
  - Supports higher-resolution rendering for video via `--render-height` and `--render-width`.
  - Resizes rendered frames back to `128x128` for policy input so eval stays compatible with training.

- `local_lerobot_act.py`
  - Shared helpers for ACT-only imports and local writable cache paths.

- `train_libero_act.sh`
  - Original convenience launcher for the baseline `chunk_size=50` training setup.

## Dataset

- Dataset root:
  - `lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act`
- Total episodes:
  - `50`
- Total frames:
  - `5068`
- FPS:
  - `20`
- Train split used in training runs here:
  - episodes `0:45`
- Validation split used in training runs here:
  - episodes `45:50`

Each sample contains:

- `observation.images.agentview`
- `observation.images.wrist`
- `observation.state`
- `action`

For ACT training, a sample becomes:

- current observation
- plus a future action chunk of length `chunk_size`
- plus `action_is_pad` for end-of-episode masking

## Evaluation Init States

- `eval_libero_act.py` uses LIBERO's `.pruned_init` file for rollout evaluation, not the HDF5 demo start states.
- For this task, the `50` states in:
  - `/home/ec2-user/LIBERO/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init`
  are not the same as the `50` HDF5 demo initial states in:
  - `/home/ec2-user/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5`
- That means evaluation from `.pruned_init` is the correct LIBERO benchmark protocol, but it is not equivalent to replaying policies from the recorded demo starts.
- To verify this directly, run:

```bash
cd /home/ec2-user/libero_test
conda run --no-capture-output -n lerobot python compare_libero_init_states.py
```

## Key Results

- Full dataset conversion passed.
  - `50` episodes
  - `5068` frames

- Smoke training passed on CPU.
  - Output:
    - `outputs/smoke_act`

- One-epoch ACT run with larger batch size completed.
  - Config:
    - `batch_size=64`
    - `chunk_size=50`
    - `n_action_steps=50`
    - `steps=73`
  - Output:
    - `outputs/libero_act_black_bowl_plate_ramekin_bs64_1epoch`
  - Final checkpoint:
    - `outputs/libero_act_black_bowl_plate_ramekin_bs64_1epoch/checkpoints/step_000073`
  - Final validation loss:
    - `3.5120292603969574`

- Main ACT run with shorter chunks completed far enough to produce a successful policy.
  - Config:
    - `batch_size=256`
    - `chunk_size=5`
    - `n_action_steps=5`
    - `100` epochs
    - `1900` total steps
  - Output:
    - `outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch`
  - Best checked checkpoint so far:
    - `outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch/checkpoints/step_001800`
  - Validation loss at step `1800`:
    - `0.5870377123355865`

- LIBERO evaluation succeeded for the step `1800` checkpoint.
  - Eval output:
    - `eval_outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_step1800`
  - Result on 1 episode:
    - `success_rate=1.0`
    - `episode=0 success=1 steps=98`

## Useful Artifacts

- Main successful checkpoint:
  - `outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch/checkpoints/step_001800`

- Successful rollout metrics:
  - `eval_outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_step1800/metrics.json`

- Successful rollout video:
  - `eval_outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_step1800/episode_000_agentview.mp4`

- Higher-resolution rendered rollout video:
  - `eval_outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_step1800/episode_000_agentview.mp4`

- One golden demo export:
  - `eval_outputs/golden_demos/episode_000_agentview_demo_fixed.mp4`

- Action histograms:
  - `eval_outputs/action_histograms/action_histograms.png`

- Action time-series plots:
  - `eval_outputs/action_time_series/action_time_series_concatenated.png`
  - `eval_outputs/action_time_series/action_time_series_normalized.png`

## Important Environment Notes

- The machine has an NVIDIA L4.
- CUDA is available only when training/eval are run unsandboxed.
- After delaying the HF image transform until after `LeRobotDataset` initialization, `num_workers=4` worked for the current image-backed training setup and reduced batch `data_wait` substantially versus `num_workers=2`.
- Stable runs here used:
  - `conda run --no-capture-output -n lerobot ...`
  - `--num-workers 4`

## Known Workarounds

- LeRobot trainer import bug:
  - Avoided by the local ACT-only trainer.

- Read-only cache locations:
  - Avoided by redirecting caches in `local_lerobot_act.py`.

- LIBERO init-state load under PyTorch 2.6:
  - Avoided in `eval_libero_act.py` with `torch.load(..., weights_only=False)`.

- `robosuite` numba cache issue:
  - Avoided in `eval_libero_act.py` with `NUMBA_DISABLE_JIT=1`.

## Recommended Commands

Train with the better-performing short-chunk setup:

```bash
cd /home/ec2-user/libero_test
conda run --no-capture-output -n lerobot python -u train_libero_act.py \
  --dataset-root /home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act \
  --output-dir /home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch \
  --repo-id local/libero-black-bowl-plate-ramekin-act \
  --train-episodes 0:45 \
  --val-episodes 45:50 \
  --batch-size 256 \
  --steps 1900 \
  --save-freq 100 \
  --log-freq 10 \
  --num-workers 4 \
  --chunk-size 5 \
  --n-action-steps 5 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --seed 1000 \
  --use-amp
```

Evaluate the step `1800` checkpoint with video:

```bash
cd /home/ec2-user/libero_test
conda run --no-capture-output -n lerobot python -u eval_libero_act.py \
  --checkpoint /home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch/checkpoints/step_001800 \
  --dataset-root /home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act \
  --output-dir /home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_step1800 \
  --num-episodes 1 \
  --max-steps 300 \
  --save-video \
  --video-camera agentview \
  --render-height 512 \
  --render-width 512
```

## Files In This Workspace

- `README.md`
- `local_lerobot_act.py`
- `convert_libero_task_to_lerobot.py`
- `train_libero_act.py`
- `train_libero_act.sh`
- `eval_libero_act.py`
