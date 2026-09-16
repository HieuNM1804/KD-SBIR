# SGCD: direct supervision of native CLIP prompts

The September 16 run had nearly constant localization agreement: map cosine
0.587034 to 0.587593 in five epochs. The trainable key/query map, evidence
adapter and fusion head provided alternative places to reduce auxiliary loss.
Aggregate head gradient norms alone do not prove which component caused the
retrieval change. Both this routing ambiguity and the flat localization trace
need to be addressed before another retrieval experiment.

## Implementation

`--retrieval_head sgcd --sgcd_student_mode native_prompt` creates no SGCD head.
Only the original independently seeded visual prompts are trainable; CLIP
parameters remain frozen. The historical `legacy_head` mode remains available
for reproducing old implementations.

- **Where:** read final-block CLS-to-patch attention using CLIP's frozen Q/K
  weights; renormalize on sketch ink support and match the cached teacher map
  with the existing confidence-weighted Hellinger loss.
- **What:** pool projected CLIP patch tokens with this map. There is no learned
  evidence adapter. The existing field alignment therefore updates prompts.
- **Effect:** compare clean/erased native sketch descriptors against the same
  photo gallery. There is no fusion layer that can absorb this loss.
- **Rank, if enabled:** operates on the native descriptor and detached photos.
  It is optional; the previous no-rank ablation did not show a rank benefit.
- **Retrieval:** exactly the main descriptor and native-dtype normalization.
  No projection or correction is added. `sgcd_beta` and `lambda_sgcd_anchor`
  must be zero; the legacy head's temperature and learning rate are unused.

The attention readout is a localization signal, not a causal attribution of
retrieval. The native clean/erased effect loss provides a separate constraint
on retrieval behavior. Backbone weights are frozen, but the encoder's outputs
can change through the prompts. With final-block input tokens, most where-loss
localization learning comes through earlier prompts; per-layer gradients are
recorded rather than assuming all layers receive equal supervision.

## Verified locally before further experiments

Tests cover native/main descriptor equality in FP16 and FP32; attention readout
agreement with MultiheadAttention; nonzero, finite gradients from each active
auxiliary component to sketch prompts; absent auxiliary photo/head gradients;
and fixed-batch optimization while frozen weights remain unchanged.

The GPU capacity probe used the local pretrained CLIP ViT-B/32, six synthetic
sketches, seed 42 and 60 steps. The targets select alternating left/right stroke
regions; they are not DFN5B teacher targets and are not a retrieval benchmark.

| Diagnostic | Initial | SGD, lr 0.01 | Adam, lr 0.03 |
| --- | ---: | ---: | ---: |
| Where loss | 0.31430 | 0.13351 | 0.01858 |
| Map/target cosine | 0.64690 | 0.89768 | 0.99174 |
| Native/initial descriptor cosine | 1.00000 | 0.86548 | 0.42178 |

SGD uses momentum 0.9 and weight decay 0.0005, as in main. The measured loss
decrease demonstrates gradient routing and learnability, not generalization.
The descriptor drift is why full training must retain the main retrieval
objective. The legacy head's initial map/ink cosine was 0.999988 and its semantic
logit standard deviation was 0.004718 on the same synthetic batch: its initial
map was almost entirely determined by the ink prior.

## Next check: existing seen teacher targets, no full training

Run `test/kaggle_sgcd_prompt_preflight.py` after installing this branch's source.
It reads the existing teacher and target caches, validates tensor structure,
class/path fingerprints and teacher-cache SHA256, and retains historical target
source hashes in its report. This probe intentionally changes the student while
holding the historical targets fixed. It does not regenerate, edit or re-label
the cache, and does not relax full training's strict source validation.

The probe temporarily trains only sketch prompts on a fixed seen batch, then
restores prompt tensors, previous gradients, module training flags and Torch
RNG. Frozen CLIP and photo prompt hashes must remain unchanged. It writes
`summary.json`, `localization.csv`, `prompt_gradients.csv`, `learning.png` and
`evidence.png` as separate files. Check loss trajectory, individual examples,
per-layer gradients and descriptor drift before interpreting the result. A
small loss reduction on this batch alone is insufficient evidence for a claim.

Full training diagnostics additionally export `component_gradients.csv` and
`prompt_learning.png`: separate loss gradients per layer, map change from
initialization, and agreement above a fixed ink prior. Do not compare old head
runs and native-prompt runs as if they shared the same numeric retrieval path.
The earlier teacher and SGCD target caches need not be deleted for this
student-only change. Native mode accepts only the expected student-source drift
when reusing the historical target cache; dataset, teacher, loss, and stroke
geometry fingerprints remain strict.

## Matched component ablations

The first native run uses W + What + Effect. Isolate the three auxiliary paths
before tuning their magnitudes:

| Cell | Where | What | Effect |
| --- | ---: | ---: | ---: |
| `kaggle_sgcd_native_where_ablation.ipy` | 1.0 | 0 | 0 |
| `kaggle_sgcd_native_where_what_ablation.ipy` | 1.0 | 0.25 | 0 |
| `kaggle_sgcd_native_where_effect_ablation.ipy` | 1.0 | 0 | 0.25 |
| `kaggle_sgcd_native_prompt_train.ipy` | 1.0 | 0.25 | 0.25 |

All four use seed 42, five epochs, verified pairwise targets, the same cache and
main losses, native descriptors, and zero anchor/rank/beta. Run the report only
after the selected ablations finish. It records final metrics separately from
mAP and precision at the same precision-selected checkpoint.

## Target controls and seed replication

The component ablation identifies W + Effect as the current candidate. Test
whether its result depends on the verified teacher target before changing any
loss weight. `kaggle_sgcd_native_random_control.ipy` and
`kaggle_sgcd_native_shuffle_control.ipy` copy every W + Effect setting and alter
only `sgcd_target`. Compare them with the completed verified W + Effect run at
seed 42. The report writes `target_control_summary.csv` and
`target_controls.png`; incomplete runs and configurations with nonzero
anchor/rank/beta are excluded.

Only if verified targets outperform both controls should the result be repeated
with seeds 43 and 44. Run each method beside its matched main baseline using the
four `*_s43.ipy` and `*_s44.ipy` cells. The report pairs runs by seed and writes
`seed_replication_deltas.csv`, `seed_replication_aggregate.csv`, and
`seed_replication_deltas.png`. A small positive result from seed 42 alone is a
development signal, not evidence for a final claim.
