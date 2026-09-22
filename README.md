# Dual-Axis Augmented Feature Distillation for ZS-SBIR

This branch is an isolated CLIP-KD AFD experiment based on main commit
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`. Main's student domain KD and
modality KD do not participate in training: both corresponding CLI weights must
be zero and the parser rejects nonzero values.

The frozen DFN5B teacher produces 1024-dimensional image and class-text
features. The frozen CLIP ViT-B/32 student plus its trainable visual prompts
produces 512-dimensional features. Two shared projections resolve this dimension
mismatch:

- one image projection is shared by sketch and photo;
- one text projection is shared by sketch and photo class templates.

Each projection receives the concatenation of normalized student and detached
teacher features and maps it to normalized 512-dimensional augmented features.
The only student objective is

```text
L = lambda_afd_sp * L_AFD(sketch, photo)
  + lambda_afd_it * 0.5 * (L_AFD(sketch, sketch-text)
                           + L_AFD(photo, photo-text)).
```

The sketch-photo term is bidirectional multi-positive InfoNCE. The image-text
term is symmetric image-to-class-text and class-text-to-image contrast. Native
student image features are used for validation and inference, so the teacher and
fusion projections are absent at deployment.

Controls make the mechanism testable:

- `student_only` removes teacher inputs while keeping the same fusion/loss code;
- `teacher_only` removes the student input and therefore should not improve the
  deployable student prompts;
- `shuffled_image`, `shuffled_text`, and `shuffled_both` preserve teacher
  marginals while breaking correspondence.

Gradient norms for prompts and fusion projections, the cosine between the two
axis gradients, branch weight norms, component losses, and native retrieval
metrics are logged to TensorBoard.

Run one configuration after the offline setup:

```bash
!python -u -m src.train \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
  --dataset sketchy_1 \
  --epochs 5 \
  --workers 8 \
  --batch_size 64 \
  --test_batch_size 1024 \
  --n_ctx_visual 3 \
  --prompt_depth 12 \
  --teacher_pretrain_epochs 2 \
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
  --teacher_cache_path /kaggle/working/teacher_cache/sketchy_1_main_teacher2_afd_v6.pt \
  --lambda_domain 0 \
  --lambda_modality 0 \
  --lambda_afd_sp 0.3 \
  --lambda_afd_it 0.3 \
  --afd_temperature_sp 0.07 \
  --afd_temperature_it 0.07 \
  --afd_fusion_lr 1e-3 \
  --afd_control verified \
  --afd_init student_identity \
  --lr 1e-2 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --seed 42 \
  --exp_name afd_full_sketchy1_s42 \
  --progress
```

Run the 25-condition one-seed mechanism and hyperparameter study with:

```bash
!python -u test/kaggle_afd_sweep.py --dataset sketchy_1
```

The sweep creates one ZIP containing logs, scalar curves, a CSV, and an analysis
manifest. It does not save model checkpoints. This sweep is exploratory; a final
claim requires rerunning selected configurations across multiple seeds.
