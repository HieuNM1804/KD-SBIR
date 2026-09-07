# CLIP-KD Feature Distillation with Dual Projectors

This branch is an ablation of `experiment/clip-kd-feature-distillation` in
which photo and sketch no longer share the train-time projection head.
Everything else remains unchanged.

- Teacher: frozen DFN5B ViT-H/14 image features, 1024 dimensions.
- Student: frozen OpenAI CLIP ViT-B/32 with independent trainable photo and
  sketch deep visual prompts, 512 dimensions.
- Photo alignment: trainable `Linear(512, 1024)` used only for photos.
- Sketch alignment: a different trainable `Linear(512, 1024)` used only for
  sketches.
- Inference: raw 512-dimensional student embeddings, exactly as in the shared
  projector branch; neither projector is used.

The two projectors have independent parameters and default PyTorch linear
initialization. Each contains 525,312 parameters, for 1,050,624 projector
parameters in total. The shared-projector baseline contains 525,312.

Both projected student features and detached teacher features are
L2-normalized by the selected feature loss:

```text
student_photo_1024 = photo_projector(student_photo_512)
student_sketch_1024 = sketch_projector(student_sketch_512)

loss = lambda_fd * 0.5 * (
    feature_loss(student_photo_1024, teacher_photo_1024)
  + feature_loss(student_sketch_1024, teacher_sketch_1024)
)
```

`feature_loss` is either MSE or cosine, with exactly one active per run. No
Domain KD, Modality KD, text KD, relational KD, PCA, or patch-to-text prompt is
used. The teacher cache remains compatible with `main` and the shared-projector
branch.

## What this ablation measures

Use exactly the same seed and hyperparameters as the shared-projector run.

- Better dual-projector results suggest that photo and sketch need different
  mappings into the teacher space.
- Similar results suggest that sharing the projector is sufficient and more
  parameter-efficient.
- Worse results suggest that the shared mapping acts as useful cross-modal
  regularization.

## MSE run

```bash
python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 20 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 1 \
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
    --feature_loss mse \
    --lambda_fd 1.0 \
    --lr 1e-5 \
    --momentum 0.95 \
    --weight_decay 5e-4 \
    --seed 42 \
    --exp_name clip_kd_fd_dual_projector_mse_l1_sketchy2 \
    --progress
```

## Cosine run

Use the same command and change only:

```bash
    --feature_loss cosine \
    --lambda_fd 1.0 \
    --exp_name clip_kd_fd_dual_projector_cosine_l1_sketchy2
```

Training logs remain `FD_PHOTO`, `FD_SKETCH`, `FD`, and `train_loss`.
Retrieval evaluation remains `mAP@200` and `P@200` for `sketchy_2`.
