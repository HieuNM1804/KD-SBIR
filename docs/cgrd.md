# Counterfactual Gap Ranking Distillation (CGRD)

## Purpose

Gap-CoRe compares the modality-specific teacher prompts with their average.
That one-sided contrast can identify a useful correction, but it cannot tell
whether the correction is tied to the correct sketch/photo assignment. CGRD
adds a negative prompt intervention:

```text
correct: photo -> P_photo, sketch -> P_sketch
common:  photo -> P_common, sketch -> P_common
swapped: photo -> P_sketch, sketch -> P_photo

P_common = (P_photo + P_sketch) / 2
```

The method uses the same learned prompts in all three states. It adds no model
parameters. The swapped state asks whether assigning each modality the wrong
prompt harms the same retrieval relation that the correct assignment helps.

## Ranking target

For each direction, CGRD chooses the best same-class gallery item and the
top-K different-class items under the teacher common state. Those identities
stay fixed for every teacher and student state. For each fixed pair it computes

```text
d_full = margin(T_correct) - margin(T_common)
d_swap = margin(T_common) - margin(T_swapped)
```

A pair is used only when both corrections exceed their configured thresholds.
The expected partial order is therefore

```text
margin(correct) > margin(common) > margin(swapped)
```

The student receives the same three prompt interventions. Its two corrections
are matched to the teacher with Smooth-L1 losses:

```text
L_cgrd = w * [SmoothL1(dS_full, d_full)
              + alpha_swap * SmoothL1(dS_swap, d_swap)]
```

`w` is the smaller of the two positive teacher corrections, clipped by
`cgrd_max_weight` and normalized over retained pairs. This bottleneck weight
prevents a strong one-sided effect from hiding a weak or absent swapped effect.

The full objective is

```text
L = L_main + lambda_gap_core * L_gap + lambda_cgrd * L_cgrd
```

Setting `lambda_cgrd=0` skips the swapped student forwards. Existing main and
Gap-CoRe experiments remain valid ablations.

## Controls and diagnostics

- `verified` uses each query's measured two-sided corrections.
- `shuffled` rolls both correction tensors across query identities while
  preserving their joint distribution.
- `reversed` negates both corrections.

The implementation logs pair coverage, query coverage, teacher monotonicity,
both teacher and student corrections, sign agreement, absolute error, all
three margins, and the CGRD/main gradient norm ratio and cosine.

The teacher audit is a required screening step. It reports the correct,
common, and swapped unseen retrieval scores and bootstrapped fixed-pair
corrections. A passing audit requires correct retrieval above common, common
above swapped, and positive lower 95% bootstrap bounds for both corrections.
This gate is evidence about the teacher intervention; it is not evidence that
the student loss improves retrieval. The matched student controls establish
that separately.

## Cache format

Teacher cache format v9 contains full, common, swapped, and base teacher
features plus base student features. `src.gap_core_cache` upgrades a v7 or v8
cache without retraining teacher prompts. A v8 source reuses its common
features and encodes only the two swapped feature sets.

The standard one-epoch cache names are:

```text
v8 source: sketchy2_gap_core_teacher1_v8.pt
v9 target: sketchy2_cgrd_teacher1_v9.pt
```

## Kaggle run order

1. Run `test/kaggle_gap_core_online.py` with Internet enabled and save output.
2. Attach that output and Sketchy to an offline GPU notebook.
3. Run `test/kaggle_gap_core_offline.py` once.
4. Run `test/kaggle_cgrd_teacher_audit.py`.
5. If the teacher gate passes, run `test/kaggle_cgrd_compare.py`.

The comparison trains each student for three epochs on seed 42 and evaluates
matched main, one-sided Gap-CoRe, verified CGRD, shuffled CGRD, and reversed
CGRD. Checkpoint creation is disabled. The result ZIP contains scalar curves,
compact metrics, necessary logs, and no model weights.

The starting CGRD configuration is deliberately conservative:

```text
lambda_cgrd=0.5
cgrd_hard_negative_topk=8
cgrd_huber_beta=0.02
cgrd_min_full_correction=0.0
cgrd_min_swap_correction=0.0
cgrd_max_weight=0.25
cgrd_swapped_loss_weight=1.0
cgrd_direction=bidirectional
```

Treat seed 42 as method development. Freeze the selected configuration before
running paired confirmation on new student seeds. A single tuned seed cannot
support a statistical significance claim.
