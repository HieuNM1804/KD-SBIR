```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 4 \
    --n_ctx_text 4 \
    --n_ctx_visual 8 \
    --lambda_nt_xent 1.0 \
    --nt_xent_temperature 0.07 \
    --lambda_kd 3.0 \
    --teacher_adapter_lr 2e-5 \
    --seed 42 \
    --exp_name sketchy2_independent_prompts_nt_xent
```

Text prompts are random learnable tokens appended after each class name and
before the period. Visual prompts are initialized independently and are not
projected from text. Both `--n_ctx_text` and `--n_ctx_visual` accept zero.

Training uses fixed resize/normalize transforms without augmentation. The
student losses are classification, paired sketch-photo NT-Xent, and DFN5B
sketch-photo relational distillation. The baseline teacher path is unchanged:
the modality adapters are jointly trained by teacher retrieval and semantic
losses.

Within the student CLIP backbone, only LayerNorm parameters are trainable;
prompt parameters and the teacher adapters are trained separately.
