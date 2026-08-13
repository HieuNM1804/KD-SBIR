```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 20 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_text 3 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --lambda_cls 0.0 \
    --lambda_kd 7.0 \
    --lambda_photo_text_kd 1.0 \
    --lambda_sketch_text_kd 1.0 \
    --photo_text_kd_temperature 0.07 \
    --sketch_text_kd_temperature 0.07 \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight_decay 1e-3 \
    --teacher_pretrain_epochs 3 \
    --teacher_pretrain_batch_size 64 \
    --teacher_adapter_lr 2e-5 \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --seed 42 \
    --exp_name sketchy2_independent_prompts
```

Photo and sketch have separate deep prompts in both the text and visual
transformers. Every prompted layer has its own independent parameters; there
is no text-to-visual projection or token sharing. Text prompts with one to
three tokens are initialized from CLIP's token embeddings for `a photo of`
and `a sketch of`. With four or more tokens, the complete text prompt is
initialized from `Normal(0, 0.02)`. Visual prompts are always initialized
independently from `Normal(0, 0.02)`.

`--prompt_depth` controls how many transformer layers receive prompts.
Both `--n_ctx_text` and `--n_ctx_visual` accept zero for branch-specific
ablations.

Training uses fixed resize/normalize transforms without augmentation. Student
classification, sketch-photo relational KD, photo-text KD, and sketch-text KD
have independent weights. Set `--lambda_cls 0` for distillation-only student
training.

Photo-text and sketch-text KD temperatures can be set independently. The
existing `--image_text_kd_temperature` remains available as the fallback for
either modality-specific temperature that is not supplied.

When `--teacher_pretrain_epochs` is positive, raw DFN5B image features are
encoded once and the modality adapters are pretrained using feature-only
batches. The trained adapters are then frozen and their outputs are
materialized for every seen sketch and photo. The adapted image features,
teacher text targets, adapter state, and configuration metadata are saved
automatically under `/kaggle/working/teacher_cache`. The filename is derived
from the dataset and complete teacher configuration. Reusing the same teacher
settings in a later cell loads the file and skips both DFN5B encoding and
teacher-adapter pretraining. Student losses, prompts, and optimizer settings
may change without invalidating the teacher cache. Pass
`--rebuild_teacher_cache` to overwrite the matching cache intentionally, or
`--teacher_cache_path` to use an explicit file.

The persistent cache is validated against the dataset paths and all teacher
pretraining settings. Use a different cache path when changing those settings.
Files under `/kaggle/working` persist across cells in the current notebook; to
reuse them in another Kaggle session, save the notebook output and attach it as
an input.

The student CLIP backbone, including every LayerNorm, is fully frozen. After
teacher pretraining, only active student prompt parameters are optimized.
