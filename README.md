# CLIP-KD Gradient Distillation for SBIR

This branch adapts Gradient Distillation (GD) from CLIP-KD to cross-domain
sketch-based image retrieval.

- Teacher: frozen DFN5B ViT-H/14 photo and sketch embeddings (1024D).
- Student: frozen OpenAI CLIP ViT-B/32 with separate trainable photo/sketch
  deep visual prompts (512D).
- Alignment: separate train-only photo and sketch Linear(512, 1024)
  projectors.
- Inference: unprojected 512D student features, so the GD heads add no serving
  cost.
- Positives: every same-class item in the batch, with probability distributed
  uniformly across them. This prevents repeated Sketchy classes becoming false
  negatives.

## Objective

For normalized anchor matrix P, candidate matrix K, temperature tau, and
row-normalized multi-positive target Y:

```text
Q = softmax(P K^T / tau)
dL/dP = (Q - Y) K   / (B tau)
dL/dK = (Q - Y)^T P / (B tau)
```

The formula is evaluated for both retrieval directions:

1. sketch anchors -> photo candidates;
2. photo anchors -> sketch candidates.

Student and teacher therefore each produce four gradient roles: sketch-anchor,
photo-key, photo-anchor, and sketch-key. GD matches corresponding roles with
the paper's batch mean squared L2 distance:

```text
L_GD = sum_role mean_batch ||g_student(role) - g_teacher(role)||_2^2
```

Teacher gradients are detached targets. The student formula remains
differentiable, so a normal backward pass updates both projectors and both
visual prompt learners. The implementation uses the closed-form derivative
instead of building a nested `autograd.grad(create_graph=True)` graph.

The complete training loss also retains the raw 512D student's bidirectional
multi-positive SBIR contrastive objective:

```text
L_total = lambda_task * L_task + lambda_gd * L_GD
```

This task term is the optimization anchor used by CLIP-KD's GD experiment. Set
`--lambda_task 0` only for a deliberate GD-only ablation.

## Kaggle run

```bash
python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 20 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --teacher_pretrain_epochs 1 \
    --teacher_pretrain_batch_size 64 \
    --teacher_n_ctx_visual 10 \
    --teacher_prompt_depth 12 \
    --teacher_prompt_std 0.02 \
    --teacher_prompt_lr 3e-2 \
    --teacher_prompt_seed 42 \
    --teacher_prompt_gradient_checkpointing \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --lambda_teacher_retrieval 1.5 \
    --teacher_triplet_margin 0.2 \
    --task_temperature 0.07 \
    --gd_temperature 0.07 \
    --lambda_task 1.0 \
    --lambda_gd 1.0 \
    --lr 1e-5 \
    --momentum 0.95 \
    --weight_decay 5e-4 \
    --seed 42 \
    --exp_name clip_kd_gd_l1_sketchy2 \
    --progress
```

The useful training logs are `TASK`, `GD`, `GD_SK_A`, `GD_PH_K`, `GD_PH_A`,
`GD_SK_K`, and `train_loss`. Retrieval evaluation remains mAP@200 and P@200
for `sketchy_2`.

For a controlled weight sweep, keep every other setting fixed and try
`--lambda_gd 0.1`, `1.0`, and `10.0`. This implementation follows the
paper's squared-L2 sum over embedding dimensions, not PyTorch's element-mean
MSE; therefore the very large coefficient used by some public CLIP-KD scripts
must not be copied directly.
