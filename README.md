# KD-SBIR: Teacher Visual Prompts, Student Visual-Only Prompts

## Bidirectional teacher triplet with symmetric student KD

Branch: `experiment/teacher-triplet-bidirectional-symmetric-kd`, based directly on
`a8d22f4f95b6f853ff164a744ca9c4b111f54aa1` (one-way teacher triplet with symmetric KD).

Teacher pretraining uses batch-hard same-class positives and different-class
negatives for both sketch anchors and photo anchors:
`0.5 * (triplet(sketch -> photo) + triplet(photo -> sketch))`.
This restores main's teacher triplet objective. With `lambda_teacher_retrieval=1.5`,
each directional term has coefficient 0.75.

Each student KD term (domain, photo-text, sketch-text) uses
`0.5 * (KL(teacher || student) + KL(student || teacher))`.
Both KL terms update only the student. Domain distributions remain sketch queries
over photo candidates; the reverse KL does not transpose the retrieval matrix.
The text losses retain main's symmetric formulation over class candidates.
Temperatures and outer objective weights remain unchanged:
`lambda_domain * domain + lambda_modality * (photo_text + sketch_text)`.

Teacher architecture and optimizer settings stay unchanged. Cache metadata now
records `teacher_triplet_direction="bidirectional"`, so one-way teacher caches
are rejected and automatic cache names differ. Legacy main caches without this
explicit direction marker are also rejected; create a new cache for this branch.

Run the focused CPU checks with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Kaggle setup

Online: paste all of `test/kaggle_online.py` into one cell, run with Internet on,
and save the notebook output. The builder clones this published branch and pins
source commit `908e482f7a99cb16f023dd3527fdfe24cfef8ca4`; it includes dependencies and both model checkpoints.
The later setup-script commit intentionally differs from the source pin.

Offline: attach the saved online output and Sketchy dataset, then use exactly two cells:

1. Paste all of `test/kaggle_offline.py` to restore the attached bundle and install offline dependencies.
2. Paste `test/kaggle_train.ipy` (the `%cd` and `!python` command below).

Teacher caches and student outputs use separate experiment paths. This starts a fresh student run, without resuming
another experiment's checkpoint.

```ipython
%cd /kaggle/working/KD-SBIR

!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
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
    --teacher_prompt_lr 0.03 \
    --teacher_prompt_seed 42 \
    --teacher_prompt_gradient_checkpointing \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 0.001 \
    --lambda_teacher_retrieval 1.5 \
    --teacher_triplet_margin 0.2 \
    --lambda_domain 3 \
    --lambda_modality 1 \
    --kd_temperature 0.07 \
    --photo_text_kd_temperature 0.15 \
    --sketch_text_kd_temperature 0.02 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --seed 42 \
    --teacher_cache_dir /kaggle/working/teacher_cache_bidirectional \
    --exp_name teacher_triplet_bidirectional_symmetric_kd_sketchy1 \
    --progress
```

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
