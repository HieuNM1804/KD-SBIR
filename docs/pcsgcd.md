# PC-SGCD: Pairwise Counterfactual Stroke-Graph Distillation

PC-SGCD extends SGCD by defining a useful stroke through retrieval ranking.
For each seen sketch, the teacher selects class-representative positive photos
and the most confusing negative classes. Erasing path `j` produces

```
effect_j = margin(clean, positive, hard negatives)
         - margin(erased_j, positive, hard negatives)
```

where `margin = positive_similarity - mean(hard_negative_similarity)`. A
positive effect means that removing the path damages the teacher's ability to
rank the correct photo above confusing photos.

Local sketch-path/photo-patch correspondence still limits verification to the
top proposals. The verified target is now a temperature-soft mixture of those
paths, which avoids unstable winner-takes-all labels. The cache also records
the hard-negative class, clean and erased margins, candidate path effects, and
soft path weights.

The student keeps SGCD's where/what/effect/anchor losses and adds a
class-aware in-batch hard-negative hinge loss on the deployed descriptor. The
photo descriptor is detached for this auxiliary loss. At inference, neither
the teacher nor the stroke extractor is required.

The main claims require all of the following:

1. Pairwise teacher audit passes before full cache generation.
2. Pairwise verified beats the matched main baseline.
3. Pairwise verified beats local, random, and shuffled controls.
4. Pairwise verified beats the positive-only SGCD ablation.
5. Removing the ranking term reduces the gain.
6. The result repeats across student seeds.

The diagnostic report contains target effects, path weights, target entropy,
clean/masked teacher margins, student rank margin, ranking violation rate,
retrieval metrics, evidence maps, descriptor intervention, and gradient
interaction.
