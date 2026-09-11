# Semantic-conditioned structural distillation

Branch: experiment/semantic-structural-distillation.
Baseline: main, b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6.
This is an experimental implementation, not a benchmarked improvement or a claim
that attribute semantics survive visual prompt tuning.

## Objective and switches

The total objective is the sum of six independently weighted terms:

    lambda_domain   * original sketch-photo relational KL
  + lambda_modality * original photo-text and sketch-text symmetric KL
  + lambda_retrieval * bidirectional multi-positive supervised contrastive loss
  + lambda_semantic * centered-cosine global semantic distillation
  + lambda_sfgw     * same-image local structural/semantic transport
  + lambda_contract * class-prototype semantic gap contraction

All four new lambdas default to zero. The original two functions in src/losses.py
are unchanged. The default sampler is main's deterministic sampler. Teacher
pretraining, teacher cache format 6, teacher selection on unseen precision, student
backbone and inference remain as in main. This preserves the existing selection
protocol; it is not a claim of a new validation split.

Retrieval treats all same-class photos as positives and averages their negative
log probabilities. Reverse retrieval is symmetric. A batch with no negative class
contributes zero retrieval loss.

The anchors use only training class names. --semantic_anchors all concatenates
'a class', 'a photo of a class', and 'a sketch of a class'; class selects only
the first group. Each model uses its own frozen text encoder. Global semantic
loss centers each image's signature across anchors and computes one minus cosine,
averaged across both modalities. Teacher signatures with near-zero centered norm
are skipped. The 60 manually written audit concepts are not used in this method.

Contraction averages unit image embeddings within each observed class and
normalizes the resulting prototype. For each model it computes the difference of
sketch and photo prototype signatures. The loss is:

    mean_c relu(norm(delta_student_c) - rho * norm(delta_teacher_c)) ** 2

Teacher prototypes are detached. Student prototypes come from the current batch;
classes with fewer than --contract_min_count samples are skipped. There is no
persistent prototype memory. This can be noisy under main's sampler; log
contract_valid_classes and use the class sampler for the primary experiments.
No assertion is made that smaller prototype gaps guarantee better retrieval.

## Local representation and solver

Scoped hooks capture the output of the final transformer block. CLS and appended
prompt tokens are excluded. Student retains its native 7x7 grid for ViT-B/32.
Teacher raw final tokens are pooled by exact area overlap, default 16x16 to 8x8.
This is spatial pooling, not learned semantic pooling.

Structural matrices use 1 - cosine between raw pooled final-block tokens.
Patch semantic signatures use the model's final LN and visual projection applied
to pooled tokens, followed by cosine against its own text anchors. Applying a
global CLIP projection to patch tokens is an experimental hypothesis, not guaranteed
region-language alignment.

FGW combines the mean squared semantic-signature difference per anchor with the
squared GW structural distortion. a and b are uniform marginals of total mass 1.
The solver adds epsilon * sum(plan * log(plan)) to the objective. It uses a detached
entropic conditional-gradient iteration, log-domain Sinkhorn, and a finite
line-search including the unchanged plan. It is an approximate nonconvex solver,
not guaranteed to find a global optimum. Small plan change is not proof of optimality.

The outer gradient differentiates student costs with the detached approximate
plan; there is no unrolled solver or gradient into teacher targets. The entropy
term is retained in the reported scalar but is constant for this outer gradient.
Consequently sfgw and even total training loss can be negative; use retrieval
metrics and separate gw_structure / gw_semantic logs, not scalar sign, to assess
training.

GW uses batched matrix products, not a materialized Nt x Nt x Ns x Ns tensor.
The costs, solver and outer loss run in FP32 even under autocast. Sinkhorn fails
explicitly if marginal error exceeds tolerance; increasing sinkhorn_iterations
or transport_epsilon is necessary in that case. Relevant defaults:

| Option | Default | Meaning |
|---|---:|---|
| fgw_alpha | 0.5 | Structural weight; 1 = structural only, 0 = semantic OT |
| transport_epsilon | 0.05 | Entropic regularization |
| gw_iterations | 10 | Maximum outer iterations |
| sinkhorn_iterations | 300 | Maximum iterations per transport subproblem |
| transport_tolerance | 0.0001 | Absolute marginal/plan change tolerance |
| contract_rho | 0.5 | Teacher-gap multiplier |
| contract_min_count | 2 | Minimum examples per class and modality |
| retrieval_temperature | 0.07 | Supervised contrastive temperature |

--local_matching spatial replaces transport by MSE of within-image structural
matrices on a common --spatial_grid (default 7); no upsampling. It uses
lambda_sfgw as the local objective weight for convenience. For a controlled
spatial-vs-GW comparison use --teacher_patch_grid 7 for the GW arm too. Do not
compare 7x7 vs 8x8 targets and attribute the entire difference to transport.

## Sampling and targets

--student_sampler class --samples_per_class 4 with batch_size 64 draws 16 distinct
seen classes, four unique sketches and four unique photos per class. Images are
sampled independently within the class. Batches sample with replacement across
steps; each epoch has floor(number_of_sketches / batch_size) steps. It is not a
full without-replacement pass over every sketch. Every class must contain at least
K photos and sketches; otherwise it fails instead of silently duplicating samples.
Sampling is deterministic by seed and epoch, independent of DataLoader workers.
Teacher pretraining always uses main's original sampler.

Global teacher targets use main's existing cache/materialization. Structural setup
then reloads DFN once; for a tuned teacher it restores exactly the visual prompt
state saved in that cache. It never substitutes the raw DFN silently.
New global semantic targets require DFN to encode text once, even if the image
cache already exists. Student and teacher anchors are frozen and not checkpointed.

--local_target_mode online keeps the frozen teacher on GPU and computes local
targets in microbatches (--local_teacher_batch_size 8). It creates no local disk
cache, but global caching from main still occurs. Teacher computation and residency
cost more runtime and GPU memory than cache mode.

--local_target_mode cache writes FP16 mmap structure/signature files, then frees
the teacher. Lookup uses exact dataset indices, not paths inferred from labels.
Cache keys include main metadata/path fingerprints, full tuned global-cache hash,
library versions, anchor sentences, grid, pooling/projection definitions.
A completed manifest is written last. Partial files are not reused. Cache file
lengths are validated. No automatic deletion of old configurations is performed.
The original main cache identifies paths but does not hash all original image
bytes; replacing image contents under the same paths requires rebuilding caches.

Estimated local bytes = N * 2 * (Nt*Nt + Nt*K) with semantic transport, or
N * 2 * Nt*Nt for structural-only/spatial. With 104 classes and all templates,
K=312. N=130532 and Nt=64 need about 5.85 GiB for local targets, plus global
cache, checkpoints and other output. Two GiB of free disk is reserved before
building; choose online or a smaller grid if this check fails.

## Kaggle run: new method without either original loss

Use the local Kaggle online/offline scripts pinned to this branch to build and
restore the bundle. The source ZIP below is an optional alternative for an older
bundle; it is not needed with the new pinned bundle. This command reuses the
one-epoch teacher cache from the audit
when present, or builds it using the same teacher settings. Starting weights below
are an ablation starting point, not tuned optimal values.

    %cd /kaggle/working/KD-SBIR
    !python -m src.train \
      --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
      --dataset sketchy_2 \
      --epochs 5 --workers 8 --batch_size 64 --test_batch_size 32 \
      --n_ctx_visual 3 --prompt_depth 12 \
      --teacher_pretrain_epochs 1 --teacher_pretrain_batch_size 64 \
      --teacher_n_ctx_visual 10 --teacher_prompt_depth 12 \
      --teacher_prompt_std 0.02 --teacher_prompt_seed 42 \
      --teacher_prompt_lr 3e-2 --teacher_prompt_gradient_checkpointing \
      --teacher_momentum 0.9 --teacher_weight_decay 1e-3 \
      --lambda_teacher_retrieval 1.5 --teacher_triplet_margin 0.2 \
      --teacher_cache_path /kaggle/working/teacher_cache/sketchy2_teacher_1ep.pt \
      --lambda_domain 0 --lambda_modality 0 \
      --lambda_retrieval 1 --lambda_semantic 1 --lambda_sfgw 1 --lambda_contract 1 \
      --student_sampler class --samples_per_class 4 \
      --semantic_anchors all --retrieval_temperature 0.07 \
      --contract_rho 0.5 --contract_min_count 2 \
      --local_matching fgw --teacher_patch_grid 8 --fgw_alpha 0.5 \
      --transport_epsilon 0.05 --gw_iterations 10 --sinkhorn_iterations 300 \
      --local_target_mode online --local_teacher_batch_size 8 \
      --local_report_dir /kaggle/working/structural_report_new_only \
      --photo_text_kd_temperature 0.15 --sketch_text_kd_temperature 0.02 \
      --kd_temperature 0.07 \
      --lr 1e-2 --momentum 0.9 --weight_decay 1e-3 \
      --seed 42 --exp_name semantic_structural_new_only --progress

To use disk targets, replace online with cache and add:

    --local_cache_dir /kaggle/working/local_teacher_cache

Use the same teacher checkpoint, training steps and sampler for controlled loss
ablations. If changing teacher settings, choose a different global cache path.
Keep default main sampler in the exact-main reference; also run main with class
sampling to measure the sampler's effect independently.

## Ablation switches

| Arm | domain | modality | retrieval | semantic | sfgw | contract |
|---|---:|---:|---:|---:|---:|---:|
| main | 3 | 1 | 0 | 0 | 0 | 0 |
| retrieval only | 0 | 0 | 1 | 0 | 0 | 0 |
| new only | 0 | 0 | 1 | 1 | 1 | 1 |
| new minus semantic | 0 | 0 | 1 | 0 | 1 | 1 |
| new minus local | 0 | 0 | 1 | 1 | 0 | 1 |
| new minus contraction | 0 | 0 | 1 | 1 | 1 | 0 |
| new plus domain | 3 | 0 | 1 | 1 | 1 | 1 |
| new plus modality | 0 | 1 | 1 | 1 | 1 | 1 |
| all six | 3 | 1 | 1 | 1 | 1 | 1 |

Use separate exp_name/report paths. For a genuinely teacher-free retrieval-only
arm, also set teacher_pretrain_epochs=0 and omit teacher_cache_path.
At least one loss must be enabled. Class sampler is optional even when only old
losses are active. The teacher configuration remains independent of the sampler.

## Reports and validation

--local_report_dir optionally saves the first training image per modality with
teacher/student grids, transport heatmap and raw JSON matrices. These files are
not a semantic accuracy audit. Existing report files are not overwritten.

Logged new terms: retrieval, semantic, contract, contract_valid_classes, sfgw,
photo/sketch_gw_structure, photo/sketch_gw_semantic, transport entropy, marginal
error and plan change. Transport logs are prefixed by modality.

    python -m unittest discover -s tests -p test_structural.py -v
    python -m src.train --help

Tests check explicit GW cost and gradients, Sinkhorn marginals/failure, alpha
endpoints, heterogeneous dimensions, frozen teacher, all 12 student prompt layers,
all six objectives, original-loss value/gradient equivalence when disabled,
online/cache consistency and restored teacher prompts. They use small models and
synthetic data; full DFN/Sketchy retrieval has not been benchmarked locally.

## Install source ZIP after the existing Kaggle setup

Skip this section when using the semantic-structural online/offline bundle.

The delivery ZIP is a complete code snapshot of this branch, without .git,
weights or training data. This lets the current offline checkpoint bundle be
reused. Set the archive path to the attached ZIP in the notebook:

    from pathlib import Path
    from zipfile import ZipFile
    import shutil

    archive = Path('/kaggle/input/YOUR_ATTACHMENT/semantic_structural_source.zip')
    project = Path('/kaggle/working/KD-SBIR').resolve()
    assert (project / 'src/model.py').is_file(), 'Run offline setup first'
    with ZipFile(archive) as package:
        for name in package.namelist():
            relative = Path(name)
            target = (project / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(project):
                raise ValueError('Invalid source archive path')
            if '.git' in relative.parts:
                raise ValueError('Source archive must not contain .git')
        package.extractall(project)
    print('Source installed. The .git HEAD still identifies the original bundle, not this overlay.')

Run the tests above before training. The snapshot contains SOURCE_COMMIT.txt
with its actual source revision. Re-running offline setup replaces the working
project, so install the snapshot after it again if needed.

## Limits

Uniform marginals and full-background patches may transfer unhelpful structure.
Teacher attribute mistakes can enter semantic signatures. The global projection
is not trained for local attributes. Prototype contraction can remove useful
variation. None of these are solved merely by introducing transport. This branch
implements the report's experiment so that these hypotheses can be tested, not
assumed. Layer OT, learned foreground weighting and automatic concept selection
are not included.
