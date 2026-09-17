# Photo-Sketch Retrieval Vocabulary Distillation

This experiment starts from `main`. It is independent of SGCD and does not use
stroke graphs, erasure targets, evidence heads, or SGCD caches.

## Hypothesis

PromptKD makes distillation easier by expressing teacher and student outputs as
logits over a fixed semantic basis. Text class vectors are not a natural basis
for sketch-based image retrieval. This branch instead builds a vocabulary of
paired sketch-photo retrieval landmarks from the prompt-tuned teacher.

The resulting objective asks whether a stable, global retrieval coordinate
system transfers teacher geometry better than the changing in-batch photo set
used by the original relational KD loss.

## Vocabulary construction

Only seen training images and their cached teacher features are used.

1. Select a central sample followed by farthest-point samples within every
   class and modality.
2. Build the teacher sketch-photo similarity matrix over these candidates.
3. Keep label-consistent mutual top-k sketch-photo edges.
4. Score each edge by similarity and its margin over cross-class retrievals.
5. Select unique edges using quality-weighted diversity with a per-class cap.

Labels are used to balance coverage and reject known cross-class edges. They do
not define the vocabulary vectors or the student target distribution.

Each vocabulary item is therefore a two-sided identity containing one teacher
sketch landmark and one teacher photo landmark. It is not a class centroid.
The student encodes the same real images once using frozen unprompted CLIP,
creating fixed landmarks in its own embedding dimension. No trainable
projector or vocabulary parameter can absorb the loss.

## Training objective

For a sketch query, teacher and student form distributions over the photo side
of the vocabulary. For a photo query, they use the sketch side. One-way KL
transfers the two rankings, weighted per sample by teacher confidence derived
from normalized entropy.

An optional Jensen-Shannon term aligns the sketch and photo distributions of a
positive training pair over the common landmark identities.

```text
lambda_domain * in_batch_relational_kd
+ lambda_modality * image_text_kd
+ lambda_retrieval_vocab * bidirectional_vocabulary_kd
+ lambda_retrieval_vocab_pair * paired_coordinate_consistency
```

Validation can use one of three descriptors:

- `native`: original CLIP descriptor; exact main inference path.
- `vocabulary`: centered landmark log-probability coordinates.
- `hybrid`: normalized concatenation of native and vocabulary descriptors.

## First experiment

```text
lambda_domain=3.0
lambda_modality=0.0
lambda_retrieval_vocab=0.5
lambda_retrieval_vocab_pair=0.0
retrieval_vocab_size=128
retrieval_vocab_candidates_per_class=8
retrieval_vocab_mutual_topk=5
retrieval_vocab_min_class_coverage=0.5
retrieval_vocab_min_mean_margin=0.0
retrieval_vocab_teacher_temperature=0.07
retrieval_vocab_student_temperature=0.07
retrieval_vocab_descriptor=native
```

This isolates vocabulary distillation while retaining the original retrieval
descriptor. Coordinate consistency and vocabulary/hybrid inference should only
be enabled after this run establishes that the teacher vocabulary is valid.

## Required ablations

1. Main: domain 3, modality 1, vocabulary 0.
2. Main without text: domain 3, modality 0, vocabulary 0.
3. Random paired landmarks.
4. Class-centroid image prototypes.
5. Photo-only landmarks.
6. Paired mutual retrieval landmarks.
7. Paired landmarks plus coordinate consistency.
8. Native, vocabulary, and hybrid inference from the same trained checkpoint.

The method is supported only if paired mutual landmarks outperform random
landmarks and class centroids over matched seeds, not merely if one run exceeds
the main baseline.
