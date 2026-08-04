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
    --lambda_kd 25.0 \
    --lambda_photo_text_kd 1.0 \
    --lambda_sketch_text_kd 1.0 \
    --image_text_kd_temperature 0.07 \
    --lr 1e-3 \
    --teacher_adapter_lr 2e-5 \
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
classification is removed completely. The student learns only from DFN5B
through sketch-photo relational KD, photo-text KD, and sketch-text KD. Photo
and sketch image-text losses have independent weights. The baseline teacher
path is unchanged: the modality adapters are jointly trained by teacher
retrieval and semantic losses.

The student CLIP backbone, including every LayerNorm, is fully frozen. Only
active prompt parameters and the teacher adapters are trainable.
