# KD-SBIR: Teacher Visual Prompts, Student Visual-Only Prompts

## Teacher sketch-to-photo triplet experiment

Branch: `experiment/teacher-triplet-sketch-photo`, based on main commit
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.

Teacher pretraining uses one-direction batch-hard triplet loss: each sketch is
an anchor, its hardest positive is the farthest same-class photo in the batch,
and its hardest negative is the nearest different-class photo. The loss is
`mean(relu(d_positive - d_negative + teacher_triplet_margin))`, with normalized
cosine distance `d = 1 - cosine`. Photo features receive gradients through these
triplets. There is no photo-anchor term or factor of one half.

The teacher cache configuration records
`teacher_triplet_direction="sketch_to_photo"`, producing a different cache key
and rejecting caches from main's symmetric teacher objective. Student objectives
and all optimizer settings follow main. Use the training command below;
no additional direction argument is needed.

Run the focused CPU checks with:

```bash
python -m unittest discover -s tests -p test_teacher_triplet.py -v
```

## Kaggle setup

The setup follows the online/offline scripts from `experiment/clip-kd-ctkd`.
The bundle pins source commit `040515e0bba195f7301d1bc8552c33a3eb6cc495`;
the later setup-script commit intentionally differs. This source commit already
contains the one-direction loss, teacher cache identity and focused tests.

In a notebook with Internet enabled, paste the whole
[`test/kaggle_online.py`](test/kaggle_online.py) into one cell and run it.
It clones this published branch, downloads the non-CUDA dependency wheels,
DFN5B and ViT-B/32 weights, checks their hashes, and saves an `offline_bundle`.
Use Save Version -> Save & Run All -> Always save output.

For training, attach that notebook's output and the Sketchy dataset to a GPU
notebook, disable Internet, and use exactly two code cells:

1. **Setup from attached output:** paste the whole
   [`test/kaggle_offline.py`](test/kaggle_offline.py). It finds this experiment's
   manifest, installs dependencies from attached wheels, restores the source
   and pretrained weights, and sets offline cache environment variables.
2. **Train:** paste [`test/kaggle_train.ipy`](test/kaggle_train.ipy), containing
   `%cd /kaggle/working/KD-SBIR` followed by `!python -m src.train`.

Use the same Kaggle Python image for online and offline notebooks where
possible so the saved wheels remain compatible. Setup includes tests and
`src.train --help`; those execute inside the setup cell.

The training cell uses your Sketchy-1 configuration: student depth 12, three
visual prompt tokens, teacher depth 12 with ten tokens and one pretraining
epoch, teacher LR 0.03, student LR 0.01, and domain/modality weights 3/1.
Teacher caches are saved under `/kaggle/working/teacher_cache_sketch_photo`.

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

```python
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
    --teacher_cache_dir /kaggle/working/teacher_cache_sketch_photo \
    --exp_name teacher_triplet_sketch_photo_sketchy1 \
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
