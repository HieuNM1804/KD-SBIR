```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 4 \
    --seed 42 \
    --exp_name sketchy2_no_student_triplet_worker_invariant
```

`--workers 4`, `--workers 5` và các worker count khác nhận cùng thứ tự batch,
positive photo và augmentation khi giữ nguyên `--seed`. Số worker chỉ thay đổi
tốc độ nạp dữ liệu.
