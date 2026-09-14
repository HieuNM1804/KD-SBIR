# Semantic region attention distillation for ZS-SBIR

Branch: `experiment/semantic-region-attention-kd`. Baseline: `main` at
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.
The original main losses/descriptor remain available with `--retrieval_head main`.
This branch tests whether semantic crop targets plus region attention can improve
category-level sketch-photo retrieval without an additional student ranking loss.
It is a research prototype; improvement and novelty require experiments.

## Descriptor and supervision

Teacher global/crop embeddings use the original tuned DFN5B teacher. Each image
is split into a 2x2 grid **after** its ordinary deterministic transform. Each
teacher crop is resized to the full input size and encoded as an image. Targets
are semantic crop embeddings, rather than pooled AV vectors or raw CLS attention.

A fixed semi-orthogonal matrix Q (1024x512) is fitted by uncentered orthogonal
Procrustes between teacher globals and the initial seeded student native globals.
Calibration uses up to 32 images per **seen** category per modality. Q is frozen;
student retrieval stays in its 512-dimensional space. Alignment can lose useful
teacher geometry, so preparation measures full unseen retrieval both before and
after Q. It is not assumed to be harmless.

The student reads dense tokens by taking the input to the last visual block and
applying its local V/output projection, residual, FFN and CLIP output projection.
Its usual CLS forward is unchanged. Within-region learned attention uses exact
patch-area overlap: a 7x7 ViT-B/32 lattice can be pooled into 2x2 regions without
rounding a region to arbitrary patch indices. A residual adapter produces region
vectors r. A shared gate combines native z and r to produce region weights w.
Sketch weights include an ink-mass prior; photos use equal region visibility.
Blank sketch regions get zero prior, and completely blank sketches use uniform.
Ink mass is a fixed heuristic, not a learned measure of sketchability.

The deployed descriptor is:

```
d = normalize(z + beta * fusion(sum_r w_r * r_r))
```

The fusion output is initialized to zero, so the initial descriptor reproduces
native CLIP with the original seeded prompts. The residual enters the descriptor
used for retrieval; it is not an auxiliary projection discarded at inference.
Default beta=0.5 is a scale, not a hard bound on correction norm. Diagnostics
measure that norm and descriptor/native cosine to detect excessive drift.

Teacher gate targets are proportional to
`prior_r * exp(cos(teacher_crop_r, teacher_global) / temperature)`.
Region selection uses no category names or labels. This semantic agreement is
**not** claimed to be causal importance or teacher transformer attention.
The method learns student pooling attention using these crop semantics.

For each modality, losses are:

```
L_descriptor = mean(1 - cosine(d, normalize(teacher_global @ Q)))
L_region     = mean(sum_r prior_r * (1 - cosine(r_r, normalize(teacher_crop_r @ Q))))
L_gate       = mean(KL(teacher_region_weights || student_region_weights))
L_reference  = mean(1 - cosine(native_z, initial_native_reference))
L_spread     = 512 * mean(relu(0.5 * std(teacher_target) - std(d))**2)
L = L_descriptor + L_region + 0.1*L_gate + 0.1*L_reference + L_spread
```

Photo/sketch losses are averaged with equal weight. Default prompts and backbone
are frozen, so L_reference is a monitoring constant in head-only runs. The spread
term uses minibatches, and is not a guarantee against full-set collapse. Only
seen category labels are used for calibration sampling. No student pairwise
ranking, triplet or contrastive loss is added. Dataset photo selection remains
same-category sampling, and is never interpreted as instance correspondence.

The local crop supervision follows an established direction:
[CLIPSelf, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e7947b5e1d30864ebbe8714dbdd611d9-Abstract-Conference.html),
[official implementation](https://github.com/wusize/CLIPSelf).
The sketch ink prior, fixed cross-teacher alignment and residual retrieval fusion
are hypotheses tested here, not established contributions or performance claims.

## Kaggle setup and commands

1. Paste `test/kaggle_online.py` into a notebook with Internet enabled. Save the
   notebook output and attach its **new semantic_region_bundle** to the offline
   notebook, together with the Sketchy dataset.
2. Disable Internet, enable a GPU, and paste `test/kaggle_offline.py`. The pair
   pins the same training source commit and validates checkpoint/source hashes.
   These setup cells do not train. A differing existing working project is
   renamed to a timestamped backup so its checkpoints remain available there.
3. Paste `test/kaggle_region_prepare.ipy`. It reuses the compatible teacher cache,
   or tunes/materializes the teacher once if it does not exist. It then prepares
   crop targets, evaluates teacher targets and initial native student, and exits
   before any student optimizer steps. Preparation reloads DFN5B and its saved
   tuned prompts for crops; ordinary student training with a complete cache does
   not load DFN5B.
4. Start with the global, uniform and semantic commands below, each as a separate
   fresh run. Inspect the initial validation and teacher oracle results before
   spending time on longer runs. A weak aligned teacher target is a reason to
   revisit alignment, rather than assume attention can fix it.
5. Paste `test/kaggle_region_report.py` and download the diagnostics ZIP.

| Command file | What it tests |
| --- | --- |
| `kaggle_main_baseline_train.ipy` | Original main: domain=3, modality=1, SGD LR=0.01 |
| `kaggle_region_global_train.ipy` | Native-global residual head + same global KD; local/gate branches frozen |
| `kaggle_region_uniform_train.ipy` | Region attention pooling; gate supervised to content prior |
| `kaggle_region_semantic_train.ipy` | Gate supervised to teacher crop/global semantic agreement |
| `kaggle_region_random_train.ipy` | Gate supervised to fixed seeded random weights times content prior |
| `kaggle_region_prompts_train.ipy` | Semantic method with additional trainable prompts at LR=0.0001 |

All new commands use domain=modality=0, 5 epochs, AdamW head LR=0.001, seed42,
student n_ctx3/depth12, the same cached teacher global/crop targets and Q. Region
loss visibility weights are identical across uniform/random/semantic runs. Gate
target changes; gate architecture, loss coefficient, initialization and fusion
are shared. Global uses the same fusion network on native globals and freezes
unused local parameters. Report parameter counts alongside the comparison.
The optional prompt run can add domain=3/modality=1 for a main-loss combination
ablation, but that should be named separately. Prompts are initially the same
in all modes and alignment is fitted to those initial prompts, including the
optional trainable-prompt run.

The cache is a **directory**, approximately 0.6â€“0.7 GiB for grid2 on Sketchy1.
Source/teacher/image hashes, configuration and initial student state are checked.
Completed shards are verified and reused after interruption. Writes are atomic,
partial .tmp files are cleaned on serialization failure, and space is checked.
Use another `--region_cache_dir` for another grid, seed or source version.
Training checkpoints include the teacher alignment, effective args, source
hashes, target metadata, optimizer and scheduler. `final.ckpt`, `last.ckpt` and
best-P@100 checkpoints are saved. To resume the exact same experiment, add
`--ckpt_path /.../last.ckpt`, use the same args/cache/run name, and set epochs to
the desired **total**. Both model and optimizer/scheduler state are restored.
A full unchanged region cache is still required for resuming training.

## Measurements and decision criteria

`tb_logs/<run>/version_*/region_diagnostics` contains:

- `initial_validation.json` and `epochs.csv`: full validation mAP/P@100 for
  deployed and native descriptors, centered covariance effective rank/spread for
  each modality, and training epoch means. Initial values have no training loss.
- `steps.csv`: sampled component losses, gate entropy, correction norm,
  descriptor/native cosine and batch target/descriptor spread.
- `gradients.csv`: once per epoch, weighted component gradient norms and cosine
  with descriptor KD, separately for head and prompts if trainable. Zero norms
  produce undefined cosine. These are gradients before the optimizer and do not
  include its momentum/Adam preconditioning.
- `attention_epoch_*.png`: images, teacher semantic gates, student gates and
  spatial student pooling weights. In uniform/random controls the displayed
  teacher semantic gate is a reference, not that run's supervised gate target.
- `training_diagnostics.png`: full metrics, full-set effective rank and losses.

Teacher preparation emits `teacher_evaluation.json`: original teacher globals,
aligned globals, uniform/semantic teacher crop pooling and initial student native.
Unseen labels are used **only for metric computation** in that oracle. Teacher
prompt/checkpoint selection still follows the original main unseen-validation
protocol, described below; this branch does not repair that evaluation protocol.

Loss reduction does not mathematically imply unseen mAP improvement. The useful
result is semantic beating global and uniform controls consistently on a proper
held-out protocol and several seeds, with no geometry collapse. An isolated gain
over initial CLIP is insufficient to attribute value to teacher region guidance.
New head/oracle scoring uses FP32; original main keeps its native descriptor
dtype. Recompute both in FP32 when assessing very small differences.
There are no full Sketchy training results for this branch yet.

## Teacher-free checkpoint inference

```python
import torch
from src.semantic_region_inference import load_region_checkpoint
module = load_region_checkpoint('/.../final.ckpt', device='cuda')
# images must use src.dataset.normal_transform(224), with modality supplied
with torch.no_grad():
    descriptor = module.model.extract_feature(images.cuda(), 'sketch')
```

The checkpoint contains the frozen backbone and region head. Loading and feature
extraction require no teacher, crop cache, student weight download or test labels.

## Local validation

```
python -m unittest discover -s tests -p "test_semantic_region*.py" -v
```

Tests cover crop geometry, area overlap, ink prior, orthogonal alignment, hooks,
frozen-backbone gradients, control weighting, atomic failures, corruption checks,
cache resume and a real small CLIP/Lightning train-save-inference cycle. The
integration fixture substitutes a fake crop teacher and a known alignment, and
fits no target on validation data. It is not a DFN5B/Sketchy performance test.

## Original main pipeline and evaluation protocol

# KD-SBIR: Teacher Visual Prompts, Student Visual-Only Prompts

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
