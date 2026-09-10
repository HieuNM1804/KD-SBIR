# Part-query semantic teacher for fine-grained SBIR

This experiment develops the teacher before distilling a student. It starts
from `experiment/fine-grained-teacher-semantic-refinement` and keeps the exact
Sketchy-FG sketch-photo pairing, each category's complete 100-photo gallery,
unseen-category Acc@1/Acc@5 selection, persistent teacher cache, and the
matched Phase-C control.

The preceding seed-42 experiment improved unseen teacher retrieval from
`0.3677/0.6459` to `0.3751/0.6581`, but its seen-train Acc@1 was only `0.4107`.
Its dynamic text branch also topped out near the visual branch (`0.3831`
image-to-text Acc@1), because fixed spatial average pooling mostly learned a
copy of the teacher's global visual representation. This branch targets both
the weak visual teacher and the missing instance-level text information.

## Phase A: stronger cross-modal visual teacher

The frozen DFN5B backbone now uses deep prompts parameterized as

```text
V_photo[l]  = V_shared[l] + alpha * Delta[l]
V_sketch[l] = V_shared[l] - alpha * Delta[l]
```

`alpha` is `--teacher_prompt_residual_scale`. The midpoint is shared across
photo and sketch, while one antisymmetric residual retains modality-specific
capacity. It has exactly the same parameter count as the previous two fully
independent prompt banks.

Phase A optimizes:

```text
L_A = lambda_retrieval * L_bidirectional_visual_InfoNCE
    + lambda_hard * L_hardest_same_category_negative
```

Sketch-to-photo uses all 100 photos as candidates. Photo-to-sketch is a
multi-positive reverse InfoNCE: when several sketches depict the same photo,
all of them are positives instead of false negatives. The margin term focuses
directly on the wrong photo currently outranking the exact pair.

## Phase B: learned semantic-part queries

Fixed adaptive spatial bins are replaced by `M` learned part queries, where
`M = --teacher_n_ctx_text`. For final-layer spatial patches `P(x)`:

```text
K = normalize(W_k * LN(P(x)))
V = W_v * LN(P(x))
A = softmax(scale * normalize(Q) * K^T)
C(x) = C_base + gate * LN(A * V)
```

Each query can follow a semantic part across sketch/photo spatial
misalignment. CLS and visual prompt tokens are excluded. The resulting
continuous contexts are inserted into the frozen CLIP text encoder before the
modality-specific class phrase:

- `[C(x)] a photo of a {class}.`
- `[C(x)] a sketch of a {class}.`

Only the text prompt learner is trainable in Phase B. Visual features and
patches are detached. Its objective is

```text
L_B = lambda_prompt * L_four_way_image_text_InfoNCE
    + lambda_text_pair * L_sketch_text_photo_text_InfoNCE
    + lambda_diversity * L_part_attention_diversity
    + lambda_anchor * L_fixed_class_semantic_anchor
```

Both reverse directions use multi-positive targets. Text-to-text InfoNCE makes
the learned parts retain exact-pair information common to sketch and photo;
the diversity loss prevents every query from selecting the same patch. Phase
B reports image-to-text, text-to-image, and text-to-text exact-instance
Acc@1/Acc@5 plus normalized attention entropy.

The metrics sidecar also stores `histories.visual_train`, so a run can be
audited for underfitting on seen pairs rather than judging only unseen Acc.

## Phase C: measured text-to-visual transfer

Two tracks restart from the exact same best Phase-A checkpoint and consume the
same deterministic batch sequence, optimizer, learning rate, scheduler, and
epoch count:

```text
control  = visual InfoNCE + hard-negative margin + visual preservation
semantic = control + warmed frozen-text semantic InfoNCE
```

Only current teacher visual prompts receive gradients. The text learner and a
copy of the Phase-A visual source are frozen. The decisive metric is

```text
[Teacher Text Added Value] = semantic best - matched-control best
```

The cache always stores the best unseen Acc@1/Acc@5 checkpoint among Phase A,
the visual-only continuation, and semantic refinement. Thus a failed text
experiment cannot weaken the teacher later used for distillation.

## Recommended Kaggle teacher-only run

Run `src.train_fg`, not category-level `src.train`:

```python
%cd /kaggle/working/KD-SBIR

!python -m src.train_fg \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg \
  --dataset sketchy_2 \
  --workers 8 \
  --batch_size 64 \
  --test_batch_size 1024 \
  --teacher_pretrain_epochs 8 \
  --teacher_text_pretrain_epochs 3 \
  --teacher_semantic_refine_epochs 3 \
  --teacher_pretrain_batch_size 64 \
  --teacher_n_ctx_visual 10 \
  --teacher_prompt_depth 12 \
  --teacher_prompt_std 0.02 \
  --teacher_prompt_lr 3e-2 \
  --teacher_visual_prompt_coupling shared_residual \
  --teacher_prompt_residual_scale 0.1 \
  --teacher_momentum 0.9 \
  --teacher_weight_decay 1e-3 \
  --lambda_teacher_retrieval 1.5 \
  --teacher_instance_temperature 0.07 \
  --teacher_reverse_infonce_weight 1.0 \
  --lambda_teacher_hard_negative 0.5 \
  --teacher_hard_negative_margin 0.1 \
  --teacher_n_ctx_text 8 \
  --teacher_text_context_generator part_query \
  --part_attention_temperature 0.07 \
  --text_prompt_gate_init 0.1 \
  --teacher_text_prompt_lr 1e-3 \
  --teacher_text_prompt_weight_decay 1e-4 \
  --teacher_prompt_infonce_temperature 0.07 \
  --lambda_teacher_prompt_infonce 1.0 \
  --lambda_teacher_text_pair_infonce 0.5 \
  --lambda_teacher_part_diversity 0.02 \
  --lambda_teacher_text_anchor 0.05 \
  --teacher_semantic_refine_lr 1e-2 \
  --teacher_semantic_warmup_epochs 2 \
  --lambda_teacher_semantic_refine 0.25 \
  --lambda_teacher_visual_keep 0.1 \
  --seed 42 \
  --exp_name fg_teacher_part_query_m8_seed42 \
  --teacher_only \
  --progress
```

First compare these lines with the preceding run:

```text
[Teacher Phase A Best]
[Teacher Phase B Best]
[Teacher Matched Control Best]
[Teacher Semantic Best]
[Teacher Text Added Value]
[Teacher Best Gain]
```

Only after seed 42 is promising should the same configuration be repeated for
seeds 43 and 44. Aggregate the new reports with:

```python
!python -m src.teacher_refinement_report \
  /kaggle/working/teacher_cache/*.pt.metrics.json \
  --minimum_runs 3 \
  --require_all_positive
```

`--require_all_positive` evaluates text-added Acc@1 against the matched
visual-only continuation, not merely against the earlier Phase-A checkpoint.

## Offline Kaggle bundle

- `test/kaggle_online.py` downloads the pinned source, offline wheels,
  ViT-B/32, and DFN5B into `/kaggle/working/offline_bundle`.
- `test/kaggle_offline.py` validates and restores that bundle in the
  Internet-disabled GPU notebook, then runs deterministic CUDA smoke tests.

The online builder pins source commit `SOURCE_COMMIT_TO_PIN`. The later bundle
commit intentionally differs so its manifest does not recursively pin itself.

## Design references

The image-conditional context principle follows CoCoOp; explicit coupling of
visual and language adaptation is motivated by MaPLe. For FG-SBIR, the design
retains exact-instance structural discrimination rather than replacing it with
category classification, consistent with the CLIP FG-ZS-SBIR formulation and
its emphasis on instance-level structure.
