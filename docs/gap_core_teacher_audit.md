# Gap-CoRe teacher audit

The first CoRe-KD experiment used the change from the generic DFN5B teacher
to the prompt-tuned teacher as its target. On Sketchy2, the student reproduced
that target, but the verified condition underperformed the matched main model.
The diagnostic showed that the target mostly encoded stronger negative
suppression and did not promote the cross-modal positive. The difference mixed
ordinary task adaptation with modality-gap adaptation.

Gap-CoRe tests a narrower counterfactual before any new student training. For
each layer, let the learned teacher prompts be `P_photo` and `P_sketch`. The two
states are:

```
P_common = (P_photo + P_sketch) / 2
full photo = P_photo
full sketch = P_sketch
common photo = common sketch = P_common
```

This changes no parameters. The full state is the existing teacher. The common
state preserves shared prompt adaptation while removing the modality-specific
prompt difference.

For every sampled seen query, the audit chooses the best same-class item and
hardest different-class item under the common state. It freezes those two
identities, then measures how the full state changes the positive score,
negative score, and retrieval margin. A causal control shuffles the
`full - common` feature residuals across image identities. Bootstrap intervals
are computed over bidirectional query rows.

The gate passes only when all three checks hold:

1. Full prompts improve unseen mAP@200 over common prompts.
2. The 95% bootstrap interval for verified-minus-shuffled margin correction is
   above zero.
3. The 95% bootstrap interval for full-minus-common margin correction is above
   zero.

Absolute positive cosine is reported as a diagnostic, but it is not a gate.
Using a shared prompt can raise all cross-modal similarities by collapsing the
score range. Retrieval improves when the positive-negative margin grows, even
if both absolute scores decrease.

Only a passing gate justifies implementing and tuning student Gap-CoRe loss.
The audit creates a ZIP with JSON, per-query CSV, plot, manifest, and logs. It
does not include caches, checkpoints, or feature tensors.

Run in this order:

1. Run `test/kaggle_gap_core_online.py` in an Internet-enabled CPU notebook and
   save its output.
2. Attach that output and Sketchy to an offline GPU notebook, then run
   `test/kaggle_gap_core_offline.py` as one cell.
3. Run `test/kaggle_gap_core_teacher_audit.py` as one cell and send the emitted
   ZIP for analysis.

Future runs use one teacher-pretraining epoch and the cache names
`sketchy2_core_teacher1_v7.pt` or `sketchy2_gap_core_teacher1_v8.pt`.
If absent, the audit prepares it once with the exact matched teacher settings.
