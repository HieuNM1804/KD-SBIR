# CLIP-KD Feature Distillation for ZS-SBIR

This experiment compares two direct feature-distillation objectives while
keeping the teacher, student, data split, prompts, optimizer, and evaluation
protocol fixed.

- Teacher: frozen DFN5B ViT-H/14 image features (1024 dimensions).
- Student: frozen OpenAI CLIP ViT-B/32 with independent trainable photo and
  sketch deep visual prompts (512 dimensions).
- Alignment: one trainable `Linear(512, 1024)` shared by photo and sketch.
- Inference: raw 512-dimensional student embeddings; the projector is not used.

Both sides are L2-normalized before matching. A run uses exactly one objective:

```text
MSE:    0.5 * (MSE(student_photo, teacher_photo)
             + MSE(student_sketch, teacher_sketch))

Cosine: 0.5 * ((1 - cosine(student_photo, teacher_photo))
             + (1 - cosine(student_sketch, teacher_sketch)))
```

The teacher cache remains byte-layout compatible with `main`, including its
text tensors, although this branch does not use text, relational, or modality
distillation during student training.

## MSE run

```bash
python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 7 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --feature_loss mse \
    --lambda_fd 1.0 \
    --lr 1e-3 \
    --momentum 0.94 \
    --weight_decay 1e-3 \
    --teacher_pretrain_epochs 4 \
    --teacher_pretrain_batch_size 64 \
    --teacher_n_ctx_visual 3 \
    --teacher_prompt_depth 12 \
    --teacher_prompt_std 0.02 \
    --teacher_prompt_lr 3e-5 \
    --teacher_prompt_seed 42 \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --lambda_teacher_retrieval 1.5 \
    --teacher_triplet_margin 0.2 \
    --seed 42 \
    --exp_name clip_kd_fd_mse \
    --progress
```

## Cosine run

Use the same command and change only:

```bash
    --feature_loss cosine \
    --lambda_fd 1.0 \
    --exp_name clip_kd_fd_cosine
```

The logged training metrics are `FD_PHOTO`, `FD_SKETCH`, `FD`, and
`train_loss`. Retrieval evaluation remains `mAP@200` and `P@200` for
`sketchy_2`, with the existing metrics for the other configured splits.
