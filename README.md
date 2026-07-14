```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

This experimental branch computes P@K with
`torchmetrics.functional.retrieval_precision` instead of the project-compatible
`P@min(K, relevant_count)` helper.
