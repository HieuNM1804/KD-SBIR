# KD-SBIR: Baseline A — Photo-only Representation Self-Challenging

Branch: `experiment/student-photo-rsc`, based on
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.

During student training, compute the existing weighted KD objective on detached
FP32 copies of the unmasked embeddings. Rank photo coordinates by
`abs(d L_KD / d photo_embedding)`. For each independently selected photo, mute
up to `floor(embedding_dim * student_rsc_drop)` highest-gradient coordinates,
renormalize, then optimize the **same existing KD objective** on the challenged
photo and unchanged sketch. The unmasked probe is not added to the training loss.
The mask is detached and no second-order derivative or encoder probe backward
is needed. Both relational KD and photo-text KD use the challenged photo.

This is a KD-loss-gradient adaptation of the self-challenging idea, not an exact
reproduction of classification RSC (which uses a correct-class score). High
gradient measures KD sensitivity; it does not establish that a coordinate
represents texture, background, or a domain shortcut. No cross-modal gradient
subtraction, teacher change, additional student parameters, or inference masking
is introduced. The mask acts on the final shared embedding coordinates, not
individual patches or transformer layers. The pretrained CLIP backbone stays
frozen; only the existing visual prompts learn.

`--student_rsc_prob 0.5` selects each photo with probability 0.5, and
`--student_rsc_drop 0.1` removes at most 51 of 512 coordinates in each selected
photo. Zero-gradient coordinates are retained. Masks that would leave a zero
vector are skipped. `rsc_drop_fraction` logs the actual fraction removed across
the batch. These are initial experimental settings, not measured optima.
Set either flag to zero for the original training computation, with no extra
random draws. RSC is disabled for validation and retrieval.

The extra training cost is one embedding-level loss/gradient probe and another
loss evaluation, without another encoder forward. Compare `prob=0` and `prob=0.5`
with identical teacher cache, seed, and all other hyperparameters before tuning.
The teacher cache format and identity remain compatible with the base commit.

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
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --lambda_domain 3.0 \
    --lambda_modality 1.0 \
    --photo_text_kd_temperature 0.15 \
    --sketch_text_kd_temperature 0.02 \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
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
    --seed 42 \
    --student_rsc_prob 0.5 \
    --student_rsc_drop 0.1 \
    --exp_name student_photo_rsc_sketchy2 \
    --progress
```

Run `python -m pytest tests -q` for numerical and CPU/CUDA training checks without
downloading the real student/teacher checkpoints. Kaggle bundle scripts are
local workflow files and are not versioned on this experiment branch.

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
