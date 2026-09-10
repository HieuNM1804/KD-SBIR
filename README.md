# Staged teacher semantic refinement for fine-grained SBIR

This experiment is based directly on
`experiment/fine-grained-teacher-train-metrics`. It preserves exact-instance
fine-grained training, the 100-photo category gallery, teacher/student visual
prompts, Domain KD, Modality KD, and Acc@1/Acc@5 model selection.

This branch fixes the unstable joint teacher optimization observed when visual
and randomly initialized text prompts were updated together. Teacher training
is now staged so text semantics must first become a stable target and can then
refine the visual representation used for retrieval. Every contrastive target
is the exact paired photo among the category's 100-photo gallery.

## Image-to-text prompt path

For each photo or sketch, only the real final-layer spatial patch tokens are
used; CLS and visual prompt tokens are excluded. If the patch tensor is
`P(x) in R^(N x Dv)`, each model's requested number of text context tokens is

```text
C(x) = C_base + g * LN(Pool_M(W * LN(P(x))))
```

`W` is a trainable visual-to-text projection, `Pool_M` adaptively reduces all
patches to exactly M tokens, and `g` is a trainable gate. The adaptive average
is implemented as a fixed pooling matrix followed by GEMM, so its CUDA backward
remains compatible with deterministic training. `C_base`, `W`, and `g` are
shared between photo and sketch; the class suffix remains modality-specific:

- `[C(x)] a photo of a {class}.`
- `[C(x)] a sketch of a {class}.`

Thus every text prompt is conditioned on the current image while keeping one
common patch-to-text mapping across both domains.

## Objectives

For gallery photo `p_j`, the teacher or student generates
`t_photo_j = Text(Prompt(Patches(p_j)), class)`. Each sketch similarly produces
`t_sketch_i`. The prompt InfoNCE objective averages two exact-instance
directions over all 100 gallery photos:

```text
L_prompt = 0.5 * [
    CE(sim(sketch_image, photo_prompt_text) / tau_prompt, paired_photo_index)
  + CE(sim(sketch_prompt_text, photo_image) / tau_prompt, paired_photo_index)
]
```

The first direction makes each sketch select the text conditioned by its paired
photo. The second makes the text conditioned by each sketch select its paired
photo image. This retains both photo and sketch prompt learners without the
same-image/category-classification shortcut.

Teacher training uses three phases:

```text
Phase A: update visual prompts only with exact-instance visual InfoNCE.

Phase B: freeze visual prompts and train text prompts with
         L_text = lambda_prompt * L_prompt
                + lambda_anchor * L_class_semantic_anchor.

Phase C: freeze text prompts and the best Phase-A visual source; update only
         current visual prompts with
         L_refine = lambda_retrieval * L_visual
                  + warmup(lambda_semantic) * L_prompt_to_visual
                  + lambda_keep * L_visual_preservation.
```

All Phase-B image and patch tensors are detached. In Phase C the semantic text
features come from a frozen Phase-A visual source and are detached inside the
loss, so gradients flow in one direction: stable text target -> current teacher
visual prompts. The best Phase-A checkpoint remains a fallback candidate;
semantic refinement is accepted only if retrieval Acc@1 (then Acc@5) improves.

Student training uses

```text
L_student = lambda_domain * L_domain_KD
          + lambda_modality * L_modality_KD
          + lambda_student_retrieval * L_visual_exact_instance_InfoNCE
          + lambda_prompt_infonce * L_prompt
```

The CLIP/DFN backbones stay frozen. Retrieval inference uses visual features
only. `--teacher_only` runs the three teacher phases, writes the persistent
cache, reports the selected phase/checkpoint, and stops before student
training. This is the recommended first experiment; train the student only
after Phase C demonstrates a repeatable teacher gain.

## Kaggle command

Run `src.train_fg`, not the category-level `src.train` entry point:

```python
%cd /kaggle/working/KD-SBIR

!python -m src.train_fg \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg \
  --dataset sketchy_2 \
  --epochs 7 \
  --workers 8 \
  --batch_size 64 \
  --test_batch_size 1024 \
  --n_ctx_visual 3 \
  --prompt_depth 12 \
  --student_n_ctx_text 8 \
  --teacher_n_ctx_text 15 \
  --text_prompt_gate_init 0.1 \
  --text_prompt_lr 1e-3 \
  --text_prompt_weight_decay 1e-4 \
  --student_instance_temperature 0.07 \
  --lambda_student_retrieval 1.0 \
  --prompt_infonce_temperature 0.07 \
  --lambda_prompt_infonce 1.0 \
  --teacher_pretrain_epochs 8 \
  --teacher_text_pretrain_epochs 3 \
  --teacher_semantic_refine_epochs 3 \
  --teacher_pretrain_batch_size 64 \
  --teacher_n_ctx_visual 10 \
  --teacher_prompt_depth 12 \
  --teacher_prompt_std 0.02 \
  --teacher_prompt_lr 3e-2 \
  --teacher_text_prompt_lr 1e-3 \
  --teacher_text_prompt_weight_decay 1e-4 \
  --lambda_teacher_retrieval 1.5 \
  --teacher_prompt_infonce_temperature 0.07 \
  --lambda_teacher_prompt_infonce 1.0 \
  --lambda_teacher_text_anchor 0.05 \
  --teacher_semantic_refine_lr 1e-2 \
  --teacher_semantic_warmup_epochs 2 \
  --lambda_teacher_semantic_refine 0.25 \
  --lambda_teacher_visual_keep 0.1 \
  --teacher_momentum 0.9 \
  --teacher_weight_decay 1e-3 \
  --lambda_domain 3.0 \
  --kd_temperature 0.07 \
  --lambda_modality 1.0 \
  --image_text_kd_temperature 0.1 \
  --lr 1e-2 \
  --momentum 0.9 \
  --weight_decay 1e-3 \
  --seed 42 \
  --exp_name fg_teacher_staged_semantic_m15 \
  --teacher_only \
  --progress
```

After a teacher-only run improves over Phase A across the required seeds,
remove `--teacher_only` and reuse the same automatic cache path to train the
student without reloading DFN5B.

The first staged experiment deliberately keeps `teacher_n_ctx_text=15`, the
best token count in the preceding joint-training runs. This isolates the
training schedule as the changed variable. Phase B reports exact-instance
Acc@1/Acc@5 in both prompt directions, and the final
`[Teacher Semantic Gain]` line reports the Phase-C improvement over the best
Phase-A visual teacher.

Every teacher cache also writes a lightweight sibling report named
`<cache>.metrics.json`. After repeating the same command with seeds 42, 43,
and 44, aggregate compatible reports without loading the 140 MB feature
caches:

```python
!python -m src.teacher_refinement_report \
  /kaggle/working/teacher_cache/*.pt.metrics.json \
  --minimum_runs 3 \
  --require_all_positive
```

The command rejects reports whose non-seed teacher configuration differs. It
returns exit code 2 when any seed has non-positive Acc@1 gain, preventing a
single favorable run from being treated as evidence that text improves the
teacher.

Changing `--teacher_n_ctx_text` or another teacher text-prompt setting produces
a different automatic teacher-cache key. Changing only
`--student_n_ctx_text` reuses the same compatible teacher cache. Old caches
from the baseline are intentionally incompatible. Use
`--rebuild_teacher_cache` only when replacing an existing cache at the same
explicit path.

## Offline Kaggle bundle

Two notebook scripts are included:

- `test/kaggle_online.py`: run once with Internet enabled to download the
  pinned source, Python wheels, ViT-B/32 checkpoint, and DFN5B checkpoint into
  `/kaggle/working/offline_bundle`. Save that notebook version with output and
  expose its output as an input dataset.
- `test/kaggle_offline.py`: attach the saved bundle and the `sketchy-fg`
  dataset to an Internet-disabled GPU notebook, then run this script. It checks
  the manifest, commit, checkpoint sizes and SHA256 values, restores both model
  caches, copies the repository to `/kaggle/working/KD-SBIR`, and runs import,
  projection, loss, and CLI smoke tests.

The bundle intentionally pins source commit
`684a71d2f55f7b06504b2902448971b01a8e664d`. The later commit containing the
bundle scripts is not used as training source, preventing the bundle metadata
from changing itself.
