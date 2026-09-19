# Gap-CoRe distillation

## Motivation

The original CoRe target compared a prompt-tuned teacher with the generic
teacher. That difference mixed task adaptation and photo-sketch gap handling.
On the matched Sketchy2 experiment, the student learned the correction but
retrieval decreased.

The teacher-only common/gap audit isolates the modality-specific component.
With the same learned prompt parameters, it compares:

```
full photo prompt   = P_photo
full sketch prompt  = P_sketch
common prompt       = (P_photo + P_sketch) / 2
```

On seed 42, the full teacher improved unseen mAP@200 from 85.11% to 92.77%
over the common teacher. Its fixed-pair margin correction was +0.1213, versus
+0.0268 for shuffled residuals. Verified-minus-shuffled had a 95% bootstrap
interval of [+0.0887, +0.1002], and all 104 seen classes had a positive mean
verified-minus-shuffled correction.

Absolute positive cosine decreased because the shared prompt compressed the
cross-modal score range. Full prompts reduced hard-negative similarity more,
so retrieval margins and unseen retrieval improved. Gap-CoRe therefore models
relative margin change rather than absolute similarity promotion.

## Student objective

The student already has independent photo and sketch prompts. No parameter is
added. Each training batch receives two student views:

- `S_full`: the existing independent photo/sketch prompts.
- `S_common`: the layerwise average prompt for both modalities.

Teacher full and common features are cached. For each query, Gap-CoRe selects
the best same-class gallery item and hardest different-class item once under
the teacher common state. The same identities are evaluated under every other
state. The correction is

```
c_T = margin(T_full) - margin(T_common)
c_S = margin(S_full) - margin(S_common)
L_gap = weighted SmoothL1(c_S, c_T)
```

Only positive teacher corrections are retained. Weighting is clipped and
normalized. Smooth-L1 matches correction magnitude; it avoids the saturating
sign-only target used by the first CoRe experiment.

The full objective is

```
L = L_main + lambda_gap_core * L_gap
```

`lambda_gap_core=0` skips both common student forwards and preserves the main
baseline path.

## Controls and diagnostics

- `verified`: each query receives its own teacher correction.
- `shuffled`: the correction distribution is preserved but reassigned across
  query identities.
- `reversed`: correction signs are reversed.

The first comparison runs main, verified, and shuffled with the same seed,
teacher, optimizer, losses, and epochs. It logs correction coverage, teacher
and student magnitude, sign agreement, absolute error, common/full margins,
and the norm ratio and cosine between main and weighted Gap-CoRe gradients on
the first batch of every epoch.

The initial weight is `lambda_gap_core=0.25`. Interpret the gradient ratio
before tuning: an auxiliary/main ratio around 0.1--0.3 is the intended range.
If it is much larger, lower the weight before running additional seeds.

All follow-up runs pretrain the teacher prompt for one epoch. Every compared
condition reuses the same resulting cache, so teacher cost is paid once.

## Cache

Format v8 adds full-dataset common teacher features. If the format-v7 CoRe
cache is attached, `src.gap_core_cache` reuses its trained prompt state and
existing tensors, encodes only the common state, and writes the v8 cache. This
avoids repeating teacher prompt pretraining.

## Kaggle run order

1. `test/kaggle_gap_core_online.py` with Internet enabled; save notebook output.
2. `test/kaggle_gap_core_offline.py` in the offline GPU notebook.
3. `test/kaggle_gap_core_compare.py` in the same GPU notebook.
4. Send the emitted `gap_core_sketchy2_comparison_*.zip`.
