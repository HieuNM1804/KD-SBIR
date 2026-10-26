# Prompted encoder cost: DFN5B, CLIP ViT-B/32, TinyCLIP-61M

Branch: `experiment/prompted-encoder-cost`, based on main `741ea27`.
Two independent modes share the same Kaggle online/offline bundle:

- **Inference:** image-encoder params, GFLOPs, batch-1 GPU/CPU latency at 224×224
  for DFN5B, CLIP-B/32 and TinyCLIP-61M with deep prompts. No training or dataset needed.
- **Training cost:** main's DFN5B -> cache -> CLIP-B/32 pipeline on Sketchy-Ext-2,
  timing teacher adaptation, feature-cache creation and student distillation.

Kaggle steps (paste each file's complete contents into a cell):

1. Online: [`test/kaggle_encoder_cost_online.py`](test/kaggle_encoder_cost_online.py).
   Or paste [`test/kaggle_encoder_cost_online_launcher.py`](test/kaggle_encoder_cost_online_launcher.py)
   to fetch and execute the current online cell directly from this branch.
   Save the notebook version with output. Optional `REUSE_WEIGHTS` avoids
   downloading backbones you already attached. Completed weights in previous
   working bundles are automatically reused after checksum verification, even
   after a failed online cell. CLIP download supports origin/CDN retries and curl.
2. Offline GPU notebook: attach that output, then run
   [`test/kaggle_encoder_cost_offline.py`](test/kaggle_encoder_cost_offline.py).
3. Choose a run cell:
   - Inference: [`test/kaggle_encoder_cost_run.ipy`](test/kaggle_encoder_cost_run.ipy).
   - Training cost: attach Sketchy and run
     [`test/kaggle_training_cost_run.ipy`](test/kaggle_training_cost_run.ipy).
4. For inference:
   Defaults: teacher 9 tokens, students 3 tokens, depth 12, GPU FP16 / CPU FP16,
   30 warm-up + 100 measured calls; all are explicit adjustable arguments.
   Return `table4.csv`, `measurements.csv`, `summary.json` under
   `/kaggle/working/encoder_cost/<timestamp>/` for analysis.
   For training: return `training_cost.csv` and `training_cost.json` under
   `/kaggle/working/training_cost/<timestamp>/`.

See [ENCODER_COST.md](ENCODER_COST.md) for the counting convention, optional trained
prompt checkpoints, CPU-only/FP32 runs, and timing scope. See
[TRAINING_COST.md](TRAINING_COST.md) for stage boundaries and cache handling.
Training reuses main without changing its losses or optimizers. Local tests use
tiny random models; full pretrained/CUDA measurements run on Kaggle.

The source pin in online/offline cells intentionally points to the runtime commit
immediately before the bundle-script pin update, avoiding a self-referential SHA.

---

## Inherited KD-SBIR training documentation

## One-way teacher triplet with symmetric student KD

Branch: `experiment/teacher-triplet-sketch-photo-symmetric-kd`, based directly on
`040515e0bba195f7301d1bc8552c33a3eb6cc495` (one-way teacher triplet).

Teacher pretraining keeps sketch anchors with batch-hard same-class positive
photos and different-class negative photos. Its triplet loss has no photo-anchor
term and no factor of one half.

Each student KD term (domain, photo-text, sketch-text) uses
`0.5 * (KL(teacher || student) + KL(student || teacher))`.
Both KL terms update only the student. Domain distributions remain sketch queries
over photo candidates; the reverse KL does not transpose the retrieval matrix.
The text losses retain main's symmetric formulation over class candidates.
Temperatures and outer objective weights remain unchanged:
`lambda_domain * domain + lambda_modality * (photo_text + sketch_text)`.

Teacher architecture, pretraining, cache metadata, and optimizer settings match
the base. Existing caches from `experiment/teacher-triplet-sketch-photo` (including
the forward-KL student branch) can be reused. Main's two-way teacher caches remain
incompatible because `teacher_triplet_direction` differs.

Run the focused CPU checks with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
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

```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_1 \
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
    --exp_name teacher_visual_student_visual_only \
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
