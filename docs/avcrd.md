# Attention-Verified Counterfactual Retrieval Distillation (AVCRD)

Experimental implementation from `experiment/patch-attention-output-kd`.
The main loss and native global retrieval descriptor are preserved at AVCRD
weight zero. No teacher/student cross-model feature projector is introduced.

## What is transferred

The teacher's last-block CLS-to-patch contributions are
`c_p = concat_h(A_h[CLS,p] V_h[p]) W_O`. The attention denominator includes
CLS, image and prompt keys; image patches alone form the proposal map.
`||c_p||_2 sqrt(ink_mass_p)` proposes `K` non-overlapping square ink windows.
These are raster **ink regions**, not recovered pen strokes or causal object
parts. We export raw CLS attention separately from the AVWO proposal norm.

Every proposal is whitened only at raster ink pixels, with a soft alpha derived
from image darkness. The original main resize/CLIP normalization is retained.
The teacher encodes each erased sketch and measures its clean-minus-erased
cosine similarity field against a fixed bank of seen training photos. The
proposal with the largest centered field RMS is selected. Labels are not used
in proposing, verification, or the AVCRD objective.

For independently normalized teacher/student descriptors:

```
G_M(i,j) = cos(z_M(sketch_i), z_M(photo_j))
D_M(i,j) = G_M(i,j) - cos(z_M(erased_sketch_i), z_M(photo_j))
center(F) = F - row_mean(F)
```

`align(F_S,F_T)` is a teacher-energy-weighted row cosine alignment of centered
fields plus a SmoothL1 penalty matching row RMS relative to teacher mean RMS.
Magnitude coefficient defaults to 0.25. Rows with negligible teacher energy
are skipped; all target fields are detached. Fields are computed in FP32.

```
L_AVCRD = w_clean align(G_S,G_T) + w_effect align(D_S,D_T)
L_total = lambda_domain L_domain + lambda_modality L_modality
          + lambda_avcrd L_AVCRD
```

The clean term is generic global-geometry KD, **not the proposed novelty**.
It anchors relative clean geometry for standalone training. The contribution is
attention-proposed, teacher-verified sketch intervention response transfer.
The recommended first comparison uses `main + effect` (`w_clean=0`) against
main alone, so any gain cannot be attributed to adding a generic clean term.
Standalone `clean + effect` must exceed the identical `clean only` control.
No pairwise ranking, top-k retrieval target, margin, class prototype, or label
loss is introduced. Training fields use the photos from the unchanged main
batch sampler. Verification uses a fixed separate photo bank.

## Controls and interpretation

A random shortlist has the same candidate count and window area. Each random
candidate approximately matches its corresponding attention candidate's ink
mass; candidates with mass error <=20% are preferred, otherwise the closest
ones are used. Actual mass errors are exported. Random candidates receive the
same teacher encoding/response-maximization budget. After independently picking
max response, the selected pair can differ in ink mass; report this difference
and inspect `candidate_effects.csv` before claiming an attention advantage.

`attention_first` and `random_first` select the first candidate without response
verification. All features/boxes are in the same cache, so those ablations add
no teacher encoding. `shuffled_effect` cyclically swaps effect field rows while
preserving the target distribution. It is a diagnostic association control.

The exploratory teacher gate compares **first candidates before verification**:
attention median response >=1.2 times random and attention wins on >=65% of
sketches. This threshold is a predeclared heuristic, not a significance test.
Response RMS measures influence; a larger response can also remove irrelevant
or harmful evidence. Passing this gate alone does not establish retrieval
benefit, causality, or novelty. A failed gate is a reason to examine proposals
before paying for full target preparation.

## Cache, cost and provenance

Full cache stores original and four selected/first teacher global sketch vectors,
pixel boxes, small attention maps and candidate scalar diagnostics. It does not
persist patch feature tensors or generated image views. Approximately 56,000
sketches with 1024-D teacher vectors need about 0.65 GiB, plus metadata; the
other model/global cache/checkpoint storage is additional. The builder checks
free disk with a 2 GiB reserve before loading the teacher. It needs one clean
and `2*K` masked teacher encodings per sketch, so compact storage is not cheap
preparation. The teacher is released before student fitting; only one erased
student sketch forward is added per train batch (none for geometry-only).

Cache keys include teacher checkpoint SHA256, teacher configuration, sketch
and photo content hashes, exact selected indices, photo bank indices, view
parameters, seed, package versions, view-source hashes and microbatch. Incompatible existing caches
fail without replacement. Atomic temporary files are removed on serialization
failure. A re-encoded clean teacher vector must agree with the original main
cache (cosine >=0.999). The inherited main teacher cache format itself records
path/config fingerprints, not historical image-content hashes.

Checkpoint metadata saves loss options, target metadata and hashes of training
source. Experiment names contain time and seed. Fresh experiments pass no
checkpoint path. The inherited main `--ckpt_path` behavior is a weight warm
start, not optimizer/scheduler resume; use fresh runs for these comparisons.

## Diagnostics

- Teacher: raw CLS attention/AVWO maps, original and erased images; first/selected
  response histograms and equal-budget comparison; ink-mass matching residuals;
  full candidate measurements and gate result.
- Training: full initial unseen retrieval, then full mAP/P@K each epoch; centered
  covariance effective rank, spread, and between/within-class spread per modality;
  per-query AP/P@K, per-class summaries, and training plus validation wall time.
- Fixed seen batch: clean/effect field alignment and RMS; native descriptor drift;
  main vs weighted AVCRD raw gradient cosine/norm for photo/sketch prompts and
  every prompt parameter; clean and counterfactual similarity field heatmaps
  with shared teacher/student color scales; raw actual/supervised/student effect CSV.

Diagnostics use `autograd.grad`, preserve RNG, do not step an optimizer and do
not overwrite `.grad` buffers. Tests verify identical SGD states with diagnostics
on/off. Covariance diagnostics use the full unseen set; response/gradient fields
use a fixed seen batch and a batch-sized gallery. These scopes are labeled.
`--no_avcrd_diagnostics` disables fixed probes and initial validation for runtime
comparison. Final and best checkpoints are both retained by the main pipeline.

Inference uses native independent global sketch/photo descriptors. No masks,
teacher, photo bank, cross-attention or reranking are needed at deployment.

## Evaluation and claim boundary

Use the same teacher, split, transforms and metrics for all runs. Compare multiple
seeds after the first matched seed experiment; report preparation time and train
cost. The inherited main policy selects teacher/student checkpoints by unseen
P@K and therefore has test-set model-selection leakage. For publication, use
held-out seen classes for selection for every method and rerun main consistently.

This code has numerical/integration tests, not a full DFN5B/Sketchy result. Claim
the specific intervention-response contribution only if main+effect beats main,
verified beats random/unverified/shuffled controls, and standalone beats clean-only
under the same protocol. Loss reduction does not mathematically guarantee mAP
improvement. Do not describe ordinary masks, attention matching, or geometry KD
as novel on their own. The literature review is in the separately delivered
`REPORT_VI.md` and `CONTRIBUTION_SPEC.md` research artifacts.

## Tests

`python -m unittest discover -s tests -v`

Includes explicit per-head AVWO equivalence, ink erasure replay, cache build/hit/
content invalidation, target detachment and gradients to both modalities, zero-
response numerical stability, CUDA FP16 prompt updates, baseline parity at zero
AVCRD weight, full initial/epoch validation and diagnostics-on/off SGD identity.
