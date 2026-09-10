# KD-SBIR: Joint Cross-Modal Geometry Distillation

Branch: `experiment/joint-cross-modal-geometry-kd`.
Baseline: `main` at `b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.
This experiment transfers the teacher's final-embedding sketch/photo geometry.
It supports training the student using **only this objective**, with both
original student losses disabled. This is a relational KD baseline to evaluate,
not a claim of a novel method or guaranteed improvement over main.

Use `src.train` and the category-level Sketchy dataset, not `src.train_fg`.
Teacher and student use their normalized final CLS embeddings after the visual
projection: width 1024 in DFN5B and 512 in ViT-B/32. The student backbone and
text encoder remain frozen; its independent visual prompts are the only
trainable parameters. All 12 prompt layers lie on the new loss's gradient path.
There are no patch hooks, projectors, learned alignment modules, or extra
inference operations. The teacher's pretraining and checkpoint selection are
unchanged from main.

For a batch of B sketches and B same-class sampled photos, define:

```text
G_SP = normalize(sketch) @ normalize(photo).T
G_SS = normalize(sketch) @ normalize(sketch).T
G_PP = normalize(photo)  @ normalize(photo).T

L_SP = mean((G_SP_student - stopgrad(G_SP_teacher))^2)
L_SS = mean_off_diagonal((G_SS_student - stopgrad(G_SS_teacher))^2)
L_PP = mean_off_diagonal((G_PP_student - stopgrad(G_PP_teacher))^2)

L_joint = alpha * L_SP + (1 - alpha) * (L_SS + L_PP) / 2
L_total = lambda_domain * L_domain_KL
        + lambda_modality * (L_photo_text + L_sketch_text)
        + lambda_joint_geometry * L_joint
```

The default `alpha = joint_cross_weight = 0.5` balances cross-modal geometry
against intra-modal geometry (SP: 50%, SS: 25%, PP: 25%). Each block is averaged
separately; self-similarities in SS and PP are excluded. The SP diagonal is
retained because its entries are different images sampled from the same class.
PS is exactly SP transposed and is not counted twice. Signed cosine values
are preserved without softmax, temperature, absolute values, or row centering.
The teacher's actual similarities supply the targets; labels are not used to
force all same-class embeddings together. Main's class pairing is retained.

Different teacher/student embedding widths are supported because the compared
objects are B-by-B relation matrices. This does not imply a lower-dimensional,
prompt-only student can exactly realize every teacher geometry. Matching only
SS/PP is insufficient for cross-modal alignment: independently rotating each
modality preserves these blocks while changing SP. The cross block prevents
that ambiguity. This loss still does not guarantee generalization or rule out
all bad stationary points; standalone retrieval performance must be measured.

Run **joint KD alone**, keeping the old teacher and student optimizer settings:

```python
%cd /kaggle/working/KD-SBIR
!python -m src.train \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
  --dataset sketchy_2 \
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
  --lambda_domain 0 \
  --lambda_modality 0 \
  --lambda_joint_geometry 1.0 \
  --joint_cross_weight 0.5 \
  --geometry_diagnostics \
  --photo_text_kd_temperature 0.15 \
  --sketch_text_kd_temperature 0.02 \
  --lr 1e-2 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --seed 42 \
  --exp_name joint_geometry_only_sketchy2 \
  --progress
```

`lambda_joint_geometry` defaults to **0**, so old main commands keep their old
objective. Weight 1.0 in this command is an experimental starting point, not a
tuned optimum or a scale matched to KL. The two text temperatures above have
no effect while `lambda_modality=0`; they are retained for reproducible main
comparisons. Do not reuse any `patch_*` or `lambda_patch_relation` flags.

Suggested controlled ablations (change exp_name for every run):

| Run | lambda_domain | lambda_modality | lambda_joint_geometry | joint_cross_weight |
| --- | ---: | ---: | ---: | ---: |
| Original main | 3 | 1 | 0 | ignored |
| Original SP KL alone | 3 | 0 | 0 | ignored |
| New joint geometry alone | 0 | 0 | 1 | 0.5 |
| SP cosine MSE alone | 0 | 0 | 1 | 1.0 |
| Main + joint geometry | 3 | 1 | 1 | 0.5 |

The SP-MSE-only ablation separates the change of distance function from the
addition of SS/PP knowledge. `joint_cross_weight=0` is also available as a
within-modality-only diagnostic, not the proposed cross-modal method. Test
against unprompted pretrained CLIP and the initialized prompted student when
interpreting degradation; frozen CLIP weights themselves are never updated.

The existing global teacher cache and its naming/metadata remain compatible
with main. Student objective settings do not change the cache identity. A
compatible cache skips teacher pretraining and DFN5B loading. On the first
run, teacher prompts are tuned as before, seen-image global features are
materialized, and DFN5B is released before student training. No patch bank or
online patch teacher is needed. The baseline's existing unseen validation
selection protocol is retained; this change does not establish a new protocol.

Progress shows `JOINT`, while TensorBoard records `joint_geometry`, `joint_sp`,
`joint_ss`, and `joint_pp`. With `--geometry_diagnostics`, it also records under
`geometry/`, for both student and teacher:

- Sketch/photo variance: summed coordinate variances of normalized embeddings.
- SP cosine standard deviation and sketch/photo centroid distance.
- Mean same-class and different-class SP cosine, with pair counts.

These are epoch averages of batch statistics, not full-dataset statistics.
Zero different-class count means the batch had no negative pairs; its reported
negative cosine placeholder is zero. Small variance/std can suggest collapse;
small centroid distance alone cannot establish successful domain alignment.

Validation without weights or dataset downloads:

```bash
python -m pytest tests/test_joint_geometry.py -q
python -c "import runpy; runpy.run_path('tests/geometry_smoke.py', run_name='__main__')"
```

The smoke executes the real 12-layer prompted student training_step with ONLY
joint KD on CPU/CUDA, checks gradients and updates for all 24 prompt tensors,
checks frozen backbone/teacher targets, and verifies unchanged inference and
checkpoint structure. Synthetic optimization checks are numerical tests;
they do not demonstrate Sketchy retrieval improvement.

## Original main documentation

The retained examples below do not enable joint geometry KD.

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
