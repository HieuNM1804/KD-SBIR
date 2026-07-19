#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/content/sketchy/Sketchy}"
ADAPTER_EPOCHS="${2:?Usage: $0 <dataset_root> <adapter_epochs>}"
ADAPTER_DIR="${ADAPTER_DIR:-teacher_adapter_runs/dfn5b_sketchy1_full_seen}"
EXP_NAME="${EXP_NAME:-sketchy1_pretrained_adapter_distill}"
WORKERS="${WORKERS:-8}"

python -m src.pretrain_teacher_adapter \
    --root "$ROOT" \
    --dataset sketchy_1 \
    --epochs "$ADAPTER_EPOCHS" \
    --batch_size 128 \
    --workers "$WORKERS" \
    --bottleneck_dim 64 \
    --lr 1e-4 \
    --temperature 0.07 \
    --triplet_margin 0.2 \
    --lambda_contrastive 1 \
    --lambda_retrieval 1 \
    --lambda_semantic 1 \
    --fp16_backbone \
    --seed 42 \
    --output_dir "$ADAPTER_DIR"

python -m src.train \
    --root "$ROOT" \
    --dataset sketchy_1 \
    --epochs 3 \
    --teacher_adapter_ckpt "$ADAPTER_DIR/last.pt" \
    --no_joint_teacher_adapter \
    --workers "$WORKERS" \
    --batch_size 64 \
    --progress \
    --lr 4e-5 \
    --quantize_fp16 \
    --seed 42 \
    --lambda_kd 3 \
    --kd_temperature 0.07 \
    --n_ctx 2 \
    --lambda_cls 1 \
    --lambda_triplet 1 \
    --exp_name "$EXP_NAME"
