```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

AdamW experiment defaults:

```bash
!python -B -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --optimizer adamw \
    --lr 1e-5 \
    --teacher_adapter_lr 2e-5 \
    --weight_decay 1e-4 \
    --exp_name sketchy1_adamw
```
