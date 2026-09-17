# Photo-Sketch PromptKD

This experiment starts from `main`. It is independent of SGCD and does not use
stroke graphs, erasure targets, evidence heads, or SGCD caches.

PromptKD uses teacher text features as shared class-vector identities. For
zero-shot sketch-based image retrieval, this branch replaces them with a bank
of cross-modal prototypes built from real training sketches and photos.

For each seen class, the teacher cache selects the configured number of central
sketch and photo anchors. Their normalized features form one cross-modal
teacher prototype. Before the first training epoch, the frozen unprompted CLIP
student encodes the same real anchors and forms a fixed prototype in the student
embedding space. The teacher and student dimensions may differ because only
prototype identities and ordering are shared.

For either an input sketch or photo, the method computes a distribution over
the prototype bank. One-way KL divergence transfers the teacher ranking to the
student. Both student modalities receive prompt gradients, while the frozen
backbone and detached prototype bank cannot absorb the objective. Validation
and inference continue to use the native student image descriptors.

The complete student objective is:

```text
lambda_domain * relational_kd
+ lambda_modality * image_text_kd
+ lambda_prototype * photo_sketch_prototype_kd
```

The first text-free screening configuration is:

```text
lambda_domain=3.0
lambda_modality=0.0
lambda_prototype=0.5
prototype_anchors_per_class=4
prototype_teacher_temperature=0.07
prototype_student_temperature=0.07
```

Recommended ablations keep all teacher, prompt, optimizer, data, and RNG
settings fixed:

1. Main: domain 3, modality 1, prototype 0.
2. Main without text: domain 3, modality 0, prototype 0.
3. Main plus photo-sketch prototypes: domain 3, modality 0, prototype 0.5.
4. Prototype only: domain 0, modality 0, prototype 0.5.
5. Main plus both targets: domain 3, modality 1, prototype 0.5.

Only seen training images may enter the prototype bank. Unseen validation images
are never used to select or refresh prototypes.
