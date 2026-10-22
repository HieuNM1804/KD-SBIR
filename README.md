# KD-SBIR: Generalized Zero-Shot SBIR

This experimental branch evaluates the existing distillation model with the
subset-GZS-SBIR protocol. Training still uses only seen categories. Every
validation query is an unseen-category sketch, while the retrieval gallery
contains every unseen photo plus photos from the fixed seen-category subset in
`GENERALIZED_CLASSES`. Relevant photos are determined by category; cosine
values remain a continuous ranking score.

The fixed seen distractors are nine Sketchy-2 classes (`teapot`, `harp`,
`piano`, `trumpet`, `saxophone`, `hourglass`, `mushroom`, `pretzel`, `bell`)
and six TU-Berlin classes (`blimp`, `tablelamp`, `telephone`,
`human-skeleton`, `pickup truck`, `megaphone`). `airplane` is not a seen
distractor because it belongs to the fixed TU-Berlin unseen split.

The branch defaults to `--eval_protocol gzs`. Use `--eval_protocol zs` only as
an unseen-gallery ablation. Sketchy-2 reports mAP@200/P@200; TU-Berlin reports
mAP@all/P@100, matching the paper's dataset-specific conventions.

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

Training matches main: teacher triplet uses only sketch anchors (sketch -> photo),
without a half factor. Domain, photo-text, and sketch-text KD each use
`0.5 * (KL(T || S) + KL(S || T))`; both terms update only the student.
SGD and default hyperparameters match main. Teacher caches record
`teacher_triplet_direction=sketch_to_photo`; old bidirectional caches are incompatible.
The fixed GZS gallery classes remain unchanged.

```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --eval_protocol gzs \
    --epochs 1 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --lambda_domain 2 \
    --lambda_modality 2 \
    --kd_temperature 0.07 \
    --photo_text_kd_temperature 0.15 \
    --sketch_text_kd_temperature 0.02 \
    --lr 0.11 \
    --momentum 0.95 \
    --weight_decay 1e-4 \
    --teacher_pretrain_epochs 1 \
    --teacher_pretrain_batch_size 64 \
    --teacher_n_ctx_visual 9 \
    --teacher_prompt_depth 12 \
    --teacher_prompt_std 0.02 \
    --teacher_prompt_lr 0.03 \
    --teacher_prompt_seed 42 \
    --teacher_prompt_gradient_checkpointing \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --lambda_teacher_retrieval 1.5 \
    --teacher_triplet_margin 0.3 \
    --seed 42 \
    --exp_name gzs_sketchy2 \
    --progress
```

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

For Kaggle, run `test/kaggle_gzs_online.py` as one online cell and save its
output. Attach the output bundle and dataset in the offline notebook, then use:

1. All of `test/kaggle_gzs_offline.py` as the setup cell.
2. `test/kaggle_train.ipy` as the `%cd` + `!python` training cell.

For TU-Berlin, change root, dataset to `tuberlin`, and experiment name.
The bundle includes non-CUDA wheels for Python 3.12/3.13, source, and model weights.
Setup does not block differing Python versions; pip selects compatible wheels.
Both scripts pin source commit `f941cc82c51e4915b78982c4a177a42ea80c2cf4`.
Rebuild the online bundle; old bundles still contain the old training losses.
