#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/libero_test"
DATASET_ROOT="${ROOT}/lerobot_datasets/libero_spatial_black_bowl_plate_ramekin_act"
OUTPUT_DIR="${ROOT}/outputs/libero_act_black_bowl_plate_ramekin"

cd "${ROOT}"

conda run -n lerobot python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
conda run -n lerobot python "${ROOT}/train_libero_act.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-id "local/libero-black-bowl-plate-ramekin-act" \
  --train-episodes "0:45" \
  --val-episodes "45:50" \
  --batch-size 16 \
  --steps 50000 \
  --save-freq 5000 \
  --log-freq 100 \
  --num-workers 4 \
  --chunk-size 50 \
  --n-action-steps 50 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --seed 1000 \
  --use-amp
