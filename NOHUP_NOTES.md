# Nohup Notes

Do not use `nohup` for long-running training jobs in this workspace.

## Why

- `nohup` plus `conda run` was unreliable here for real-time logging.
- Background launches appeared to start, but `train.log` often stayed empty.
- In practice, this made it hard to tell whether the job had failed, stalled, or was still initializing.

## Preferred Approach

Use a persistent shell session with the conda environment activated, and mirror stdout/stderr into a log file with `tee`.

Example:

```bash
/bin/bash -lc 'source /home/ec2-user/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
PYTHONUNBUFFERED=1 python /home/ec2-user/libero_test/train_libero_act_taskbalanced_sharedencoder.py ... 2>&1 | \
tee /home/ec2-user/libero_test/outputs/libero_act_libero_spatial_multitask_taskbalanced_sharedencoder/train.log'
```

## Rule

- Avoid `nohup` for training jobs.
- Prefer a persistent PTY/session for anything that needs visible progress.
- Use `python -u` or `PYTHONUNBUFFERED=1` when live logs matter.
