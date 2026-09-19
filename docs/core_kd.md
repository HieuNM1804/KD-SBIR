# CoRe-KD: Cross-Modal Retrieval Correction Distillation

CoRe-KD starts from `main`. The baseline relational and modality losses and the
native student descriptor remain unchanged. The added target transfers how an
SBIR-adapted DFN5B teacher improves cross-modal positive-negative margins over
its own unprompted base state.

For a sketch query, same-class photo and different-class hard negative, define

```text
teacher correction =
    (adapted teacher positive score - adapted teacher negative score)
  - (base teacher positive score - base teacher negative score)
```

The student forms the same quantity from prompted and frozen-unprompted CLIP.
A soft Bernoulli ranking target transfers the magnitude and direction of the
teacher correction. The photo-to-sketch direction is trained symmetrically.

Within each batch CoRe-KD chooses:

- the same-class anchor most promoted by teacher adaptation;
- among the base teacher's top-k cross-class confusions, the anchor most
  suppressed by adaptation;
- only pairs whose teacher margin correction exceeds the configured minimum.

`verified` uses the real anchor corrections. `shuffled` preserves their values
but rolls anchor identities. `reversed` negates the teacher adaptation delta.
These controls are claim-critical: a useful correction target must outperform
both.

## Cache contract

Cache format 7 contains four feature states for every seen training image:

- adapted teacher `T1`;
- unprompted teacher `T0`;
- frozen unprompted student `S0`;
- prompted student `S_gamma` is evaluated online.

Old main caches are intentionally incompatible and must be rebuilt once. The
new cache is shared by the matched main and CoRe runs.

## Primary comparison

Keep all main hyperparameters identical and change only:

```text
main: lambda_core=0
CoRe: lambda_core=0.5, core_control=verified
```

Start with the native descriptor and bidirectional correction. Tune only after
the primary run establishes non-zero coverage and verified correction beats the
matched main and controls.

## First tuning parameters

1. `lambda_core`: 0.25, 0.5, 1.0.
2. `core_student_temperature`: 0.03, 0.05, 0.10.
3. `core_hard_negative_topk`: 4, 8, 16.
4. `core_min_teacher_correction`: 0.0, 0.005, 0.01.

Do not tune all four grids jointly. Select the weight first, then temperature,
and retain settings only when verified correction remains better than shuffled
and reversed targets.
