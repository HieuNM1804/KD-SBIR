```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

This experimental branch applies modality-aware augmentation to student inputs:
geometric/stroke dropout for sketches and crop/color augmentation for photos.
DFN5B receives resize-only views so relational-KD targets remain stable.
