# Fine-grained SBIR with exact-instance text-prompt InfoNCE

This experiment is based directly on
`experiment/fine-grained-teacher-train-metrics`. It preserves exact-instance
fine-grained training, the 100-photo category gallery, teacher/student visual
prompts, Domain KD, Modality KD, and Acc@1/Acc@5 model selection.

The new branch adds a text prompt learner to both the frozen CLIP ViT-B/32
student and the frozen DFN5B ViT-H/14 teacher. Unlike the preceding category
classification experiment, every new prompt objective targets the exact paired
photo among the category's 100-photo gallery. Soft-token counts are
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

Teacher pretraining uses

```text
L_teacher = lambda_teacher_retrieval * L_visual_exact_instance_InfoNCE
          + lambda_teacher_prompt_infonce * L_prompt
```

Student training uses

```text
L_student = lambda_domain * L_domain_KD
          + lambda_modality * L_modality_KD
          + lambda_student_retrieval * L_visual_exact_instance_InfoNCE
          + lambda_prompt_infonce * L_prompt
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
  --student_instance_temperature 0.07 \
  --lambda_student_retrieval 1.0 \
  --prompt_infonce_temperature 0.07 \
  --lambda_prompt_infonce 1.0 \
  --teacher_pretrain_epochs 2 \
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
  --exp_name fg_exact_instance_text_infonce_m8_m12 \
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
`a244c06fd8d6fb79e34af35b29c8e529b9dfffa6`. The later commit containing the
bundle scripts is not used as training source, preventing the bundle metadata
from changing itself.
