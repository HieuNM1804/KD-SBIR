# Fine-grained SBIR with image-conditioned text prompts

This experiment is based directly on
`experiment/fine-grained-teacher-train-metrics`. It preserves exact-instance
fine-grained training, the 100-photo category gallery, teacher/student visual
prompts, Domain KD, Modality KD, and Acc@1/Acc@5 model selection.

The new branch adds a text prompt learner to both the frozen CLIP ViT-B/32
student and the frozen DFN5B ViT-H/14 teacher. Their soft-token counts are
controlled independently by `--student_n_ctx_text` and
`--teacher_n_ctx_text`. The legacy `--n_ctx_text` and `--text_prompt_tokens`
aliases set only the student count.

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

The fine-grained sampler creates a category-pure batch, so its batch alone has
no category negatives. Classification therefore uses the fixed text prototype
bank of every seen class as negatives. For each image, the true-class column is
replaced by the score from its image-conditioned text prompt before applying
cross-entropy.

Teacher pretraining uses

```text
L_teacher = lambda_teacher_retrieval * L_exact_instance_InfoNCE
          + lambda_teacher_text_cls * (L_photo_text_CE + L_sketch_text_CE) / 2
```

Student training uses

```text
L_student = lambda_domain * L_domain_KD
          + lambda_modality * L_modality_KD
          + lambda_text_cls * (L_photo_text_CE + L_sketch_text_CE) / 2
```

The CLIP/DFN backbones stay frozen. Teacher visual prompts and teacher text
prompts are optimized jointly during teacher pretraining. Student visual
prompts and student text prompts are optimized jointly during student
training, with separate learning rates. Retrieval inference is unchanged and
uses the student visual features only.

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
  --teacher_n_ctx_text 12 \
  --text_prompt_gate_init 0.1 \
  --text_prompt_lr 1e-3 \
  --text_prompt_weight_decay 1e-4 \
  --text_cls_temperature 0.07 \
  --lambda_text_cls 1.0 \
  --teacher_pretrain_epochs 2 \
  --teacher_pretrain_batch_size 64 \
  --teacher_n_ctx_visual 10 \
  --teacher_prompt_depth 12 \
  --teacher_prompt_std 0.02 \
  --teacher_prompt_lr 3e-2 \
  --teacher_text_prompt_lr 1e-3 \
  --teacher_text_prompt_weight_decay 1e-4 \
  --teacher_text_cls_temperature 0.07 \
  --lambda_teacher_retrieval 1.5 \
  --lambda_teacher_text_cls 1.0 \
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
  --exp_name fg_image_conditioned_text_m8 \
  --progress
```

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
`8b4dddcaf8d0c854e43ffaded6149f19ab005f3c`. The later commit containing the
bundle scripts is not used as training source, preventing the bundle metadata
from changing itself.
