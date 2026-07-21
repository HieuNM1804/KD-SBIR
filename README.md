```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 4 \
    --seed 42 \
    --exp_name sketchy2_no_student_triplet_worker_invariant
```

With the same `--seed`, `--workers 4`, `--workers 5`, and other worker counts
produce the same batch order, positive-photo selection, and augmentation. The
worker count only changes data-loading throughput.
