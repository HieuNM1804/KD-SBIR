# KD-SBIR: Across-Dataset Zero-Shot SBIR

This branch trains the existing teacher/student distillation pipeline on one
source dataset and evaluates the source-selected student checkpoint on a
different target dataset. Target sketches/photos are never used by teacher
pretraining, student training, or checkpoint selection.

The target query/gallery uses the fixed subsets published by the
SpLIP/ZSE-SBIR protocol: Sketchy-1 -> TU-Berlin has 21 classes and Sketchy-1 ->
QuickDraw has 11 classes. They are stored in `src/data_config.py`. Evaluation
reports mAP@all and P@100. The code prints the exact selected class list and
fails if a target directory is missing or a target class appears in source
training. A custom direction remains possible through `--target_classes_file`.

This branch keeps the DFN5B teacher visual-prompt pretraining pipeline from
`experiment/teacher-visual-prompt-tuning`. The teacher has separate photo and
sketch deep visual prompts, learns only from retrieval triplet loss, and is
validated on the unseen retrieval split after each pretraining epoch.

The student CLIP backbone is fully frozen. Its only trainable parameters are
independent photo and sketch deep visual prompts. The student text encoder has
no learnable prompt tokens and remains fully frozen. Image-text distillation
uses fixed CLIP features from these modality-specific templates:

- `a photo of a {class}.`
- `a sketch of a {class}.`

The student objectives are sketch-photo relational KD, photo-text KD, and
sketch-text KD. Setting an objective weight to zero disables that objective.

```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_1 \
    --target_root /kaggle/input/datasets/b20dccn616nguynhutun/tuberlin \
    --target_dataset tuberlin \
    --epochs 7 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --lambda_domain 1.0 \
    --lambda_modality 1.0 \
    --photo_text_kd_temperature 0.2 \
    --sketch_text_kd_temperature 0.02 \
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
    --exp_name sketchy1_to_tuberlin \
    --progress
```

To run Sketchy-1 to QuickDraw, only change `--target_root`,
`--target_dataset quickdraw`, and `--exp_name`. Teacher caches depend solely on
the source-side teacher configuration, so the same tuned teacher cache is
reused across target datasets and student-only ablations.

Teacher caches are named from the dataset and complete teacher configuration.
Changing only student prompts, losses, or optimizer settings reuses a compatible
teacher cache. Use `--rebuild_teacher_cache` only when intentionally replacing
that cache.

Teacher prompt pretraining keeps the epoch with the highest unseen P@K, restores
that prompt state, and materializes the distillation cache from it. Student
checkpoints are also ranked by unseen P@K instead of mAP. Ties keep the earlier
teacher epoch. Neither state cloning nor checkpoint serialization consumes RNG.
Because unseen labels determine both selections, this setting has test-set
model-selection leakage and is not a strict inductive ZS-SBIR protocol.
