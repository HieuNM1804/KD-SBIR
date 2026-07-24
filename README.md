```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 4 \
    --n_ctx_text 4 \
    --n_ctx_visual 8 \
    --lambda_kd 3.0 \
    --seed 42 \
    --exp_name sketchy2_independent_prompts
```

Text prompts are random learnable tokens appended after each class name and
before the period. Visual prompts are initialized independently and are not
projected from text. Both `--n_ctx_text` and `--n_ctx_visual` accept zero.

Training uses fixed resize/normalize transforms without augmentation. The only
losses are classification and DFN5B sketch-photo relational distillation.
Within the student CLIP backbone, only LayerNorm parameters are trainable;
prompt parameters are trained separately.
