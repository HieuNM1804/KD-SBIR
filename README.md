```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

This experimental branch replaces the teacher adapter's symmetric batch-hard
triplet with a one-way triplet using a random different-class photo from the batch.
