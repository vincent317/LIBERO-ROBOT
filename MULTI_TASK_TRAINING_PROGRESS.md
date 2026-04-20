# Multi-Task Training Progress

## Scope

- Benchmark: `libero_spatial`
- Training mode: one shared ACT policy across all `10` LIBERO-Spatial tasks
- Current dataset root:
  - `/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act`
- Current run output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid`

## Combined Dataset

- Total tasks: `10`
- Total episodes: `500`
- Total frames: `62250`
- FPS: `20`
- Robot type: `libero_panda`

Source:
- [lerobot_datasets/libero_spatial_all_tasks_act/meta/info.json](/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act/meta/info.json:1)

## Split

- Per-task train episodes: `45`
- Per-task validation episodes: `5`
- Total train episodes: `450`
- Total validation episodes: `50`

Source:
- [lerobot_datasets/libero_spatial_all_tasks_act/conversion_manifest.json](/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_all_tasks_act/conversion_manifest.json:1)

## Observation And Action

- Inputs:
  - `observation.images.agentview`
  - `observation.images.wrist`
  - `observation.state`
  - `task_index`
- Output:
  - `action`

State contents:
- `gripper_qpos_left`
- `gripper_qpos_right`
- `eef_pos_x`
- `eef_pos_y`
- `eef_pos_z`
- `eef_quat_w`
- `eef_quat_x`
- `eef_quat_y`
- `eef_quat_z`

## Task Conditioning

- Conditioning type: learned discrete `task_id` embedding
- Number of task IDs: `10`
- One shared learnable embedding table is used in both:
  - transformer encoder
  - transformer decoder
- The VAE encoder is not task-conditioned
- Task text is stored in the dataset per frame, but the current ACT model path does not consume text

Current token structure:
- Main transformer encoder inputs:
  - latent `z` token
  - `task_id` token
  - robot-state token
  - image tokens from `agentview`
  - image tokens from `wrist`
- Transformer decoder inputs:
  - learned decoder query tokens
  - plus the same learned `task_id` embedding added to every decoder query

## Current Training Recipe

- Policy family: LeRobot `ACT`
- Vision backbone: `resnet18`
- Vision initialization:
  - `ResNet18_Weights.IMAGENET1K_V1`
- Batch size: `32`
- Chunk size: `5`
- Action horizon: `5`
- Steps: `19000`
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- KL weight: `5.0`
- Seed: `1000`
- AMP: enabled
- Requested DataLoader workers: `4`

Oversampling:
- Strategy: gripper transition oversampling
- Match rule:
  - grasp `(-1 -> 1)`
  - release `(1 -> -1)`
- Transition window: `5`
- Oversample weight: `2.0`
- Matched train samples in the multitask dataset: `11491 / 55981`

Source:
- [outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/run_config.json](/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/run_config.json:1)

## Status

- Combined multitask dataset conversion: complete
- Multitask training run: complete
- Training metrics file:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/training_metrics.txt`
- Final completed training step:
  - `19000`
- Final checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/checkpoints/step_019000`
- Checkpoints written:
  - every `100` steps through `step_019000`

## Final Evaluation Setup

- Evaluation script:
  - `/home/ec2-user/libero_test/eval_libero_spatial_multitask_act.py`
- Evaluated checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/checkpoints/step_019000`
- Benchmark:
  - `libero_spatial`
- Episodes per task:
  - `50`
- Max rollout steps per episode:
  - `300`
- Task conditioning at eval:
  - discrete `task_index`
- Final eval output dir:
  - `/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask_step19000_taskid_parallel2`
- Parallel eval configuration:
  - `2` worker processes
- Eval speedup changes used:
  - skip image resize when frames are already `128x128`
  - shard tasks across worker processes, one policy/env stack per worker

Source:
- [eval_outputs/libero_act_libero_spatial_multitask_step19000_taskid_parallel2/aggregate_metrics.json](/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask_step19000_taskid_parallel2/aggregate_metrics.json:1)

## Final Evaluation Results

- Mean success rate across all `10` tasks:
  - `0.694`

Per-task success rates:
- `task_00` bowl between plate and ramekin -> plate:
  - `0.82`
- `task_01` bowl next to ramekin -> plate:
  - `0.64`
- `task_02` bowl from table center -> plate:
  - `0.94`
- `task_03` bowl on cookie box -> plate:
  - `0.92`
- `task_04` bowl in top drawer of wooden cabinet -> plate:
  - `0.48`
- `task_05` bowl on ramekin -> plate:
  - `0.16`
- `task_06` bowl next to cookie box -> plate:
  - `0.78`
- `task_07` bowl on stove -> plate:
  - `0.54`
- `task_08` bowl next to plate -> plate:
  - `0.80`
- `task_09` bowl on wooden cabinet -> plate:
  - `0.86`

Best tasks:
- `task_02`:
  - `0.94`
- `task_03`:
  - `0.92`

Weakest tasks:
- `task_05`:
  - `0.16`
- `task_04`:
  - `0.48`

## Current Issue

- The original sandboxed launch hit a multiprocessing IPC permission failure with `num_workers=4`
- Final training and final benchmark evaluation were run non-sandboxed
- The main remaining issue is uneven task performance across the multitask benchmark:
  - strong on easier tabletop placements
  - weak on the ramekin and top-drawer variants

## OVERSAMPLE_BAD_TASK

- Applies to:
  - the original multitask architecture
  - the original non-balanced `WeightedRandomSampler` training path in `/home/ec2-user/libero_test/train_libero_act.py`
- Purpose:
  - increase sampling pressure on the weakest multitask benchmark tasks without switching to balanced per-task batches
- Task-level oversample multipliers:
  - `task_04 = 4`
  - `task_05 = 8`
  - `task_09 = 4`
- Combination rule:
  - these task weights multiply with the existing gripper-transition oversampling weights
- Trainer metadata label:
  - `task_oversample_strategy = OVERSAMPLE_BAD_TASK`

Oversample run used here:
- Output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid_oversample_bad_task`
- Batch size:
  - `32`
- Chunk size:
  - `5`
- Action horizon:
  - `5`
- Steps:
  - `19000`
- Learning rate:
  - `1e-5`
- Weight decay:
  - `1e-4`
- KL weight:
  - `5.0`
- Seed:
  - `1000`
- AMP:
  - enabled

## OVERSAMPLE_BAD_TASK Benchmark Evaluation

- Evaluated checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid_oversample_bad_task/checkpoints/step_019000`
- Eval output dir:
  - `/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask_step19000_taskid_oversample_bad_task_parallel4`
- Episodes per task:
  - `50`
- Max rollout steps per episode:
  - `300`
- Parallel eval workers used:
  - `4`

Final benchmark result at `step_019000`:
- Mean success rate across all `10` tasks:
  - `0.220`

Per-task success rates:
- `task_00` bowl between plate and ramekin -> plate:
  - `0.10`
- `task_01` bowl next to ramekin -> plate:
  - `0.10`
- `task_02` bowl from table center -> plate:
  - `0.10`
- `task_03` bowl on cookie box -> plate:
  - `0.36`
- `task_04` bowl in top drawer of wooden cabinet -> plate:
  - `0.24`
- `task_05` bowl on ramekin -> plate:
  - `0.00`
- `task_06` bowl next to cookie box -> plate:
  - `0.78`
- `task_07` bowl on stove -> plate:
  - `0.26`
- `task_08` bowl next to plate -> plate:
  - `0.12`
- `task_09` bowl on wooden cabinet -> plate:
  - `0.14`

Comparison against the earlier multitask task-id baseline:
- Earlier multitask baseline mean success:
  - `0.694`
- `OVERSAMPLE_BAD_TASK` mean success:
  - `0.220`
- Difference:
  - `-0.474`

Notes:
- This strategy materially worsened benchmark performance instead of improving the weakest tasks.
- `task_06` held steady at `0.78`, but every other task was worse than the baseline.
- `task_05`, one of the intended rescue targets, dropped from `0.16` to `0.00`.

## Shared Encoder Variant

- New model path:
  - `/home/ec2-user/libero_test/local_multitask_act_shared_task_encoder.py`
- New train path:
  - `/home/ec2-user/libero_test/train_libero_act_taskbalanced_sharedencoder.py`
- Purpose:
  - add the same shared learnable `task_id` embedding to the ACT/VAE encoder
  - keep one shared learnable task embedding table across:
    - ACT/VAE encoder
    - transformer encoder
    - transformer decoder
  - keep the original architecture and training entrypoints unchanged

Conditioning layout in this variant:
- ACT/VAE encoder inputs:
  - CLS token
  - learned `task_id` token
  - action sequence tokens
  - robot-state token
- Main transformer encoder inputs:
  - latent `z` token
  - learned `task_id` token
  - robot-state token
  - image tokens from `agentview`
  - image tokens from `wrist`
- Transformer decoder inputs:
  - learned decoder query tokens
  - plus the same learned `task_id` embedding added to every decoder query

## Balanced Batch Sampling Variant

- Batch construction rule:
  - each batch is sampled evenly across all `10` tasks
- Sampler implementation:
  - `BalancedTaskBatchSampler` in `/home/ec2-user/libero_test/train_libero_act_taskbalanced_sharedencoder.py`
- Constraint:
  - `batch_size` must be divisible by `10`

Balanced run used here:
- Output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30`
- Batch size:
  - `30`
- Per-task samples per batch:
  - `3`
- Chunk size:
  - `5`
- Action horizon:
  - `5`
- Steps:
  - `50000`
- Learning rate:
  - `1e-5`
- Weight decay:
  - `1e-4`
- KL weight:
  - `5.0`
- Seed:
  - `1000`
- AMP:
  - enabled
- Requested DataLoader workers:
  - `4`

## Shared Encoder Run Status

- Run status:
  - complete
- Launch mode:
  - non-sandboxed conda shell with live `tee` logging
- Avoid `nohup` for this environment:
  - see `/home/ec2-user/libero_test/NOHUP_NOTES.md`
- Training log:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30/train.log`
- Training metrics:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30/training_metrics.txt`
- Run config:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30/run_config.json`
- Final completed training step:
  - `50000`
- Final checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30/checkpoints/step_050000`

Validation checkpoints observed:
- `step_005000`:
  - `val_loss=0.440182`
- `step_010000`:
  - `val_loss=0.404158`
- `step_025000`:
  - `val_loss=0.411527`
- `step_050000`:
  - `val_loss=0.407398`

Final training metrics at `step=50000`:
- `loss=0.142091`
- `mean_recent_loss=0.133525`
- `l1=0.140307`
- `kl=0.000357`
- elapsed:
  - `4277.8s` (`~71.3` minutes)

Comparison versus the previous multitask task-id architecture:
- The new run trains stably and reaches lower training loss over time.
- This is not an apples-to-apples architecture comparison because both the conditioning path and the batch sampling scheme changed.
- The earlier run used:
  - no task conditioning in the ACT/VAE encoder
  - uneven task composition within a batch
  - `batch_size=32`
- The new run used:
  - shared task conditioning in the ACT/VAE encoder, transformer encoder, and decoder
  - even per-task sampling within each batch
  - `batch_size=30`

## Shared Encoder Benchmark Evaluation

- Evaluated checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30/checkpoints/step_020000`
- Eval output dir:
  - `/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder_bs30_step20000_failure_videos`
- Episodes per task:
  - `50`
- Max rollout steps per episode:
  - `300`
- Parallel eval workers used:
  - `2`
- Failure videos:
  - enabled

Final benchmark result at `step_020000`:
- Mean success rate across all `10` tasks:
  - `0.658`

Per-task success rates:
- `task_00` bowl between plate and ramekin -> plate:
  - `0.92`
- `task_01` bowl next to ramekin -> plate:
  - `0.66`
- `task_02` bowl from table center -> plate:
  - `0.90`
- `task_03` bowl on cookie box -> plate:
  - `0.90`
- `task_04` bowl in top drawer of wooden cabinet -> plate:
  - `0.52`
- `task_05` bowl on ramekin -> plate:
  - `0.06`
- `task_06` bowl next to cookie box -> plate:
  - `0.88`
- `task_07` bowl on stove -> plate:
  - `0.70`
- `task_08` bowl next to plate -> plate:
  - `0.58`
- `task_09` bowl on wooden cabinet -> plate:
  - `0.46`

Comparison against the earlier multitask task-id baseline:
- Earlier multitask baseline checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid/checkpoints/step_019000`
- Earlier multitask baseline mean success:
  - `0.694`
- Shared-encoder balanced run at the comparable checkpoint:
  - `0.658`
- Difference:
  - `-0.036`

Notes:
- This comparison is closer to apples-to-apples than using the shared-encoder `step_050000` checkpoint because the earlier multitask baseline was evaluated at `step_019000`.
- A full benchmark evaluation for the shared-encoder `step_050000` checkpoint was started but not completed.

## Balanced Batch Baseline Variant

- New train path:
  - `/home/ec2-user/libero_test/train_libero_act_taskbalanced_taskid.py`
- Purpose:
  - keep the original multitask `act_task_id` architecture unchanged
  - keep gripper-transition oversampling unchanged
  - replace only the original non-balanced `WeightedRandomSampler` with `BalancedTaskBatchSampler`

Balanced baseline run used here:
- Output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_taskid_bs30`
- Batch size:
  - `30`
- Per-task samples per batch:
  - `3`
- Chunk size:
  - `5`
- Action horizon:
  - `5`
- Requested steps:
  - `50000`
- Actual run handling:
  - the run was allowed to go past the original baseline budget by mistake
  - it was stopped manually
  - the comparison checkpoint chosen for eval was the closest saved checkpoint to the baseline budget: `step_020000`

## Balanced Batch Baseline Benchmark Evaluation

- Evaluated checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_taskid_bs30/checkpoints/step_020000`
- Eval output dir:
  - `/home/ec2-user/libero_test/eval_outputs/libero_act_libero_spatial_multitask_taskbalanced_taskid_bs30_step20000`
- Episodes per task:
  - `50`
- Max rollout steps per episode:
  - `300`
- Parallel eval workers used:
  - `2`
- Failure videos:
  - disabled

Final benchmark result at `step_020000`:
- Mean success rate across all `10` tasks:
  - `0.614`

Per-task success rates:
- `task_00` bowl between plate and ramekin -> plate:
  - `0.78`
- `task_01` bowl next to ramekin -> plate:
  - `0.62`
- `task_02` bowl from table center -> plate:
  - `0.92`
- `task_03` bowl on cookie box -> plate:
  - `0.82`
- `task_04` bowl in top drawer of wooden cabinet -> plate:
  - `0.36`
- `task_05` bowl on ramekin -> plate:
  - `0.02`
- `task_06` bowl next to cookie box -> plate:
  - `0.92`
- `task_07` bowl on stove -> plate:
  - `0.40`
- `task_08` bowl next to plate -> plate:
  - `0.72`
- `task_09` bowl on wooden cabinet -> plate:
  - `0.58`

Comparison against earlier multitask runs:
- Original multitask task-id baseline mean success:
  - `0.694`
- Balanced batch baseline mean success:
  - `0.614`
- Difference vs original baseline:
  - `-0.080`
- Shared-encoder balanced run mean success at `step_020000`:
  - `0.658`
- Difference vs shared-encoder balanced run:
  - `-0.044`

Notes:
- This isolates balanced per-task batch construction more cleanly than the shared-encoder balanced run because the policy architecture stayed at the original `act_task_id` design.
- On this benchmark, balanced per-task batches alone did not improve over the original baseline.

## VAE-Only Task-ID Variant

- New model path:
  - `/home/ec2-user/libero_test/local_multitask_act_vae_encoder.py`
- New train path:
  - `/home/ec2-user/libero_test/train_libero_act_vae_encoder.py`
- Purpose:
  - keep the original multitask `act_task_id` encoder and decoder conditioning path
  - add `task_id` only to the ACT/VAE encoder
  - keep the original non-balanced `WeightedRandomSampler` and gripper-transition oversampling

VAE-only run used here:
- Output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_bs32_chunk5_19000step_kl5_taskid_vae_encoder`
- Batch size:
  - `32`
- Chunk size:
  - `5`
- Action horizon:
  - `5`
- Steps:
  - `19000`
- Learning rate:
  - `1e-5`
- Weight decay:
  - `1e-4`
- KL weight:
  - `5.0`
- Seed:
  - `1000`
- AMP:
  - enabled
- Task oversampling:
  - disabled

Final training metrics at `step=19000`:
- `train_loss=0.186870`
- `mean_recent_train_loss=0.207675`
- `val_loss=0.383771`
- `l1_loss=0.186920`
- `kl_loss=-0.000010`

Comparison versus the original multitask task-id baseline at `step=19000`:
- Original baseline `train_loss`:
  - `0.188671`
- VAE-only `train_loss`:
  - `0.186870`
- Original baseline `mean_recent_train_loss`:
  - `0.211492`
- VAE-only `mean_recent_train_loss`:
  - `0.207675`
- Original baseline `val_loss`:
  - `0.383343`
- VAE-only `val_loss`:
  - `0.383771`

Notes:
- This run finished essentially tied with the original baseline.
- Training loss was slightly better than baseline, but validation loss was slightly worse.
- A full LIBERO benchmark evaluation for this VAE-only checkpoint has not been run yet.

## Task 05 Single-Task Check

Goal:
- test whether the best verified single-task ACT recipe can rescue the weak multitask `task_05` case

Task:
- `task_05` bowl on ramekin -> plate:
  - language: `pick up the black bowl on the ramekin and place it on the plate`

Dataset conversion:
- Source HDF5:
  - `/home/ec2-user/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5`
- Output dataset:
  - `/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_task05_black_bowl_on_ramekin_act`
- Repo id:
  - `local/libero-task05-black-bowl-on-ramekin-act`
- Converted episodes:
  - `50`
- Converted frames:
  - `5796`

Training recipe used:
- copied from the best verified single-task recipe in `TRAINIG_PROGRESS.md`
- output dir:
  - `/home/ec2-user/libero_test/outputs/libero_act_task05_black_bowl_on_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl5`
- batch size:
  - `32`
- chunk size:
  - `5`
- action horizon:
  - `5`
- steps:
  - `1900`
- learning rate:
  - `1e-5`
- weight decay:
  - `1e-4`
- KL weight:
  - `5.0`
- seed:
  - `1000`
- AMP:
  - enabled
- oversampling:
  - gripper-transition oversampling only
  - transition window `5`
  - oversample weight `2.0`

Training result:
- Final checkpoint:
  - `/home/ec2-user/libero_test/outputs/libero_act_task05_black_bowl_on_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl5/checkpoints/step_001900`
- Final validation loss at `step_001900`:
  - `0.759951`

Evaluation:
- Eval output dir:
  - `/home/ec2-user/libero_test/eval_outputs/libero_act_task05_black_bowl_on_ramekin_bs32_chunk5_step1900_failures_kl5`
- Episodes:
  - `50`
- Success rate:
  - `0.00`
- Success count:
  - `0 / 50`
- Failure videos saved:
  - `50`

Comparison against the multitask `task_05` baseline:
- Multitask task-id baseline on `task_05`:
  - `0.16`
- Multitask task-id baseline with eval-only `10`-step no-op:
  - `0.12`
- New single-task `task_05` run:
  - `0.00`

Conclusion:
- Porting the best known single-task recipe to a freshly converted `task_05` dataset did not help.
- The result is much worse than the multitask baseline on this task.
- The most likely explanation is a train/eval mismatch or dataset-conversion mismatch specific to this new single-task `task_05` setup, not simple undertraining.

## Strategy Summary

- `Task-id baseline`:
  - transformer encoder/decoder conditioned with a learned discrete `task_id`
  - original non-balanced `WeightedRandomSampler`
  - gripper-transition oversampling only
  - best completed benchmark result so far: mean success `0.694`
- `OVERSAMPLE_BAD_TASK`:
  - same task-id architecture and original sampler
  - added task-level oversampling multipliers for weak tasks `04`, `05`, and `09`
  - result: mean success `0.220`
  - conclusion: not a viable training strategy
- `Shared encoder + balanced batches`:
  - added the same learned `task_id` embedding to the ACT/VAE encoder, transformer encoder, and decoder
  - replaced the original sampler with even per-task batch construction
  - comparable benchmark result at `step_020000`: mean success `0.658`
  - conclusion: optimization was stable, but the completed benchmark result is still below the task-id baseline
- `Balanced batches only`:
  - kept the original `act_task_id` architecture
  - changed only the sampler to even per-task batches
  - benchmark result at `step_020000`: mean success `0.614`
  - conclusion: balanced per-task batching alone was worse than both the original baseline and the shared-encoder balanced run
- `VAE-only task_id`:
  - kept the original sampler and encoder/decoder conditioning
  - added `task_id` only inside the ACT/VAE encoder
  - final training result at `step_019000`: nearly identical to baseline validation loss
  - conclusion: promising as a clean architectural ablation, but benchmark eval is still pending

Current takeaways:
- The plain multitask task-id baseline remains the strongest verified strategy.
- Targeted bad-task oversampling was clearly harmful.
- Balanced per-task batching alone did not help.
- The shared-encoder balanced approach remains unresolved at its final checkpoint because the full `step_050000` benchmark evaluation was not completed.
- The VAE-only task-id variant is the cleanest next benchmark candidate because it stayed closest to the original baseline in training while changing only one conditioning location.
