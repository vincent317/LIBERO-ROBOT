# Training Progress

## Current Task

- Benchmark: `libero_spatial`
- Task: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- Language instruction: `pick up the black bowl between the plate and the ramekin and place it on the plate`

## Dataset

- Source demos: `50` episodes, `5068` total frames
- Local dataset root: `/home/ec2-user/libero_test/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act`
- Train split: episodes `0:45`
- Validation split: episodes `45:50`
- Observation keys:
  - `observation.images.agentview`
  - `observation.images.wrist`
  - `observation.state`
- Action key:
  - `action`

## Model

- Policy family: LeRobot `ACT`
- Implementation: `/home/ec2-user/libero_test/train_libero_act.py`
- Vision backbone: `resnet18`
- Inputs:
  - `agentview` RGB image
  - `wrist` RGB image
  - `robot state`
- Output:
  - chunked action prediction

## Training Recipe

- Optimizer: `AdamW`
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- Seed: `1000`
- Device used in recorded main runs: `cuda`
- AMP: enabled
- Practical environment settings used here:
  - `conda run --no-capture-output -n lerobot`
  - current recommended `--num-workers 4`
  - grasp-only oversampling: `open (-1) -> close (1)` transitions with `transition_window=5`, `oversample_weight=2.0`

## Main Short-Chunk Run

- Output dir: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch`
- Batch size: `256`
- Steps: `1900`
- Chunk size: `5`
- Action horizon: `5`
- Checkpoint `step_001800`:
  - Train loss: `0.39307767152786255`
  - Mean recent train loss: `0.3987330436706543`
  - Validation loss: `0.5870377123355865`
- Checkpoint `step_001900`:
  - Train loss: `0.3652629554271698`
  - Mean recent train loss: `0.3698241710662842`
  - Validation loss: `0.568953812122345`

## Grasp-Oversampling Run

- Output dir: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_100epoch_grasp_os_nw2_single`
- Batch size: `128`
- Steps: `1900`
- Chunk size: `5`
- Action horizon: `5`
- Num workers: `2`
- Gripper oversampling:
  - Match rule: grasp transitions only (`-1 -> 1`)
  - Transition window: `5`
  - Oversample weight: `2.0`
- Final checkpoint `step_001900`:
  - Train loss: `0.4366857707500458`
  - Mean recent train loss: `0.44243946969509124`
  - Validation loss: `0.6426221877336502`
  - Elapsed time: `1292.94s`

## Gripper-Transition Oversampling Runs

- Match rule:
  - both grasp (`-1 -> 1`) and release (`1 -> -1`) transitions
  - transition window: `5`
  - oversample weight: `2.0`
  - matched samples: `1196 / 4613`
  - effective sample-mass increase: about `25.9%`

- `batch_size=32`, `num_workers=4`
  - Output dir: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_100epoch_gripper_transitions_nw4`
  - Final checkpoint `step_001900`:
    - Train loss: `0.864547848701477`
    - Mean recent train loss: `0.8289335012435913`
    - Validation loss: `0.9195051749547323`
    - Elapsed time: `196.34s`

- `batch_size=128`, `num_workers=4`
  - Output dir: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_100epoch_gripper_transitions_nw4`
  - Final checkpoint `step_001900`:
    - Train loss: `0.43420690298080444`
    - Mean recent train loss: `0.44467719495296476`
    - Validation loss: `0.6358998268842697`
    - Elapsed time: `555.24s`

## Evaluation Results

- Main benchmark eval
  - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs256_chunk5_100epoch/checkpoints/step_001800`
  - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_step1800_failures`
  - Episodes: `50`
  - Success rate: `0.44`
  - Success count: `22 / 50`

- Grasp-oversampling benchmark eval on LIBERO `.pruned_init`
  - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_100epoch_grasp_os_nw2_single/checkpoints/step_001900`
  - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_step1900_pruned_init_failures`
  - Episodes: `50`
  - Success rate: `0.46`
  - Success count: `23 / 50`
  - Failure videos saved: `27`

- Gripper-transition benchmark eval on LIBERO `.pruned_init`
  - `batch_size=32`
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_100epoch_gripper_transitions_nw4/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures`
    - Episodes: `50`
    - Success rate: `0.62`
    - Success count: `31 / 50`
    - Failure videos saved: `19`
  - `batch_size=128`
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_100epoch_gripper_transitions_nw4/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs128_chunk5_step1900_pruned_init_gripper_transitions_failures`
    - Episodes: `50`
    - Success rate: `0.40`
    - Success count: `20 / 50`
    - Failure videos saved: `30`

- Reduced-KL gripper-transition benchmark eval on LIBERO `.pruned_init`
  - `batch_size=32`, `kl_weight=5.0`
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl5/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures_kl5`
    - Episodes: `50`
    - Success rate: `0.82`
    - Success count: `41 / 50`
    - Failure videos saved: `9`

- Additional ablation benchmark evals on LIBERO `.pruned_init`
  - `batch_size=32`, `kl_weight=2.0`, transition oversampling enabled
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl2/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures_kl2`
    - Episodes: `50`
    - Success rate: `0.76`
    - Success count: `38 / 50`
    - Failure videos saved: `12`
  - `batch_size=32`, `kl_weight=5.0`, no oversampling
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_1900step_no_oversampling_kl5/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures_no_oversampling_kl5`
    - Episodes: `50`
    - Success rate: `0.52`
    - Success count: `26 / 50`
    - Failure videos saved: `24`
  - `batch_size=32`, `kl_weight=5.0`, transition oversampling enabled, pretrained `resnet18`
    - Checkpoint: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl5_pretrained_resnet18/checkpoints/step_001900`
    - Output dir: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures_kl5_pretrained_resnet18`
    - Episodes: `50`
    - Success rate: `0.78`
    - Success count: `39 / 50`

## Benchmark Summary Table

- Approximate epochs below use `145` steps per epoch, since the train split has `4613` samples and the benchmark runs here use `batch_size <= 32` or checkpointed step counts directly for comparison.

| Run | Batch Size | Chunk Size | Oversampling Strategy | Steps | Approx. Epochs | Best / Eval Checkpoint | Eval Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main short-chunk baseline | `256` | `5` | `none` | `1900` | `13.1` | `step_001800` | `0.44` (`22 / 50`) |
| Grasp-only oversampling | `128` | `5` | `grasp only`, window `5`, weight `2.0` | `1900` | `13.1` | `step_001900` | `0.46` (`23 / 50`) |
| Gripper-transition oversampling | `32` | `5` | `grasp + release`, window `5`, weight `2.0` | `1900` | `13.1` | `step_001900` | `0.62` (`31 / 50`) |
| Gripper-transition oversampling | `128` | `5` | `grasp + release`, window `5`, weight `2.0` | `1900` | `13.1` | `step_001900` | `0.40` (`20 / 50`) |
| Gripper-transition oversampling | `16` | `5` | `grasp + release`, window `5`, weight `2.0` | `1900` | `13.1` | `step_001900` | `0.46` (`23 / 50`) |
| Gripper-transition oversampling | `32` | `5` | `grasp + release`, window `5`, weight `2.0` | `3800` | `26.2` | `step_003700` | `0.50` (`25 / 50`) |
| Gripper-transition oversampling, reduced KL | `32` | `5` | `grasp + release`, window `5`, weight `2.0`, `kl_weight=5.0` | `1900` | `13.1` | `step_001900` | `0.82` (`41 / 50`) |
| Gripper-transition oversampling, reduced KL | `32` | `5` | `grasp + release`, window `5`, weight `2.0`, `kl_weight=2.0` | `1900` | `13.1` | `step_001900` | `0.76` (`38 / 50`) |
| Gripper-transition oversampling, reduced KL, pretrained `resnet18` | `32` | `5` | `grasp + release`, window `5`, weight `2.0`, `kl_weight=5.0` | `1900` | `13.1` | `step_001900` | `0.78` (`39 / 50`) |
| No oversampling, reduced KL | `32` | `5` | `none`, `kl_weight=5.0` | `1900` | `13.1` | `step_001900` | `0.52` (`26 / 50`) |

## Manual Failure Breakdown

- Source: manual review of the `28` recorded failure videos from the `50`-episode benchmark eval
- Does not even touch the bowl: `5`
  - `002, 018, 021, 038, 047`
- Touched the bowl but was unable to grasp it: `16`
  - `006, 008, 010, 012, 013, 014, 024, 026, 027, 030, 035, 036, 041, 044, 046, 048`
- Bowl was placed on the plate but slightly tilted: `7`
  - `005, 023, 025, 028, 034, 039, 049`

## Failure Mode Summary

- Main failure mode: grasp acquisition after contact (`16 / 28` failures)
- Secondary failure mode: near-success placement rejected because the bowl was slightly tilted on the plate (`7 / 28`)
- Smaller failure mode: approach failure where the robot does not even touch the bowl (`5 / 28`)

## Current Best Result

- Best successful checkpoint so far: `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_1900step_gripper_transitions_nw4_kl5/checkpoints/step_001900`
- Best benchmark success rate so far: `0.82` on `50` LIBERO evaluation episodes (`41 / 50`)
- Best benchmark eval output dir so far: `/home/ec2-user/libero_test/eval_outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_step1900_pruned_init_failures_kl5`
- Best recorded validation loss among the strong benchmarked transition-oversampling runs kept here: `0.5584837158521017` at `step_003700` from `/home/ec2-user/libero_test/outputs/libero_act_black_bowl_plate_ramekin_bs32_chunk5_200epoch_gripper_transitions_nw4`
- Key finding so far:
  - reducing `kl_weight` from `10.0` to `5.0` on the prior best `batch_size=32`, `chunk_size=5`, transition-oversampling setup improved benchmark success from `0.62` to `0.82`
  - pushing `kl_weight` further down to `2.0` reduced success to `0.76`
  - switching that same `kl_weight=5.0` setup to pretrained ImageNet `resnet18` reached a slightly better final validation loss (`0.667927` vs `0.697599`) but lower benchmark success (`0.78` vs `0.82`)
  - removing oversampling while keeping `kl_weight=5.0` reduced success to `0.52`

## Data Pipeline Findings

- The original startup stalls were not caused by the original HDF5 file being read in the loop.
- Training reads from the converted LeRobot parquet/image dataset, but the HF transform was being attached too early and triggered unnecessary image decoding during dataset initialization.
- `train_libero_act.py` was updated to:
  - delay the HF image transform until after `LeRobotDataset` initialization
  - read raw columns with `with_format(None)` while building gripper oversampling weights

## Timing Findings

- A timing-instrumented run was added to report:
  - `data_wait_s`
  - `host_prep_s`
  - `gpu_step_s`
- At `num_workers=2`, `batch_size=128`, `chunk_size=5`, step `100`:
  - `data_wait_s=0.3357`
  - `host_prep_s=0.0050`
  - `gpu_step_s=0.1525`
- At `num_workers=4`, `batch_size=128`, `chunk_size=5`, step `100`:
  - `data_wait_s=0.1640`
  - `host_prep_s=0.0044`
  - `gpu_step_s=0.1489`
- Interpretation:
  - host-side prep after batch arrival is negligible
  - the main source of GPU idle time is waiting for the next batch
  - increasing `num_workers` from `2` to `4` reduced `data_wait_s` by about half while leaving `gpu_step_s` nearly unchanged

## Evaluation Protocol Note

- `eval_libero_act.py` evaluates on LIBERO `.pruned_init` states.
- For this task, the `.pruned_init` states are not the same as the HDF5 demo initial states.
- So the reported success rates here are benchmark-style LIBERO eval results, not demo-start replay results.
