```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

### CLIP ViT-B/32 zero-shot inference baseline

Chạy một dataset:

```bash
python -B -m src.eval_clip32 \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --batch_size 512 \
    --output clip32_sketchy1.json
```

Chạy đủ 4 split/dataset:

```bash
python -B -m src.eval_clip32 \
    --dataset all \
    --sketchy_root /content/sketchy/Sketchy \
    --tuberlin_root /content/tuberlin/TUBerlin \
    --quickdraw_root /content/quickdraw/QuickDraw \
    --batch_size 512 \
    --output clip32_all.json
```

### Foundation CLIP teacher inference candidates

Liệt kê các teacher có sẵn trong script:

```bash
python -B -m src.eval_foundation_teachers --list_teachers
```

Chạy nhóm teacher khuyến nghị, tự skip model có tổng params >= 5B:

```bash
python -B -m src.eval_foundation_teachers \
    --dataset all \
    --sketchy_root /content/sketchy/Sketchy \
    --tuberlin_root /content/tuberlin/TUBerlin \
    --quickdraw_root /content/quickdraw/QuickDraw \
    --teacher recommended \
    --max_params_b 5 \
    --batch_size 128 \
    --output teacher_candidates.json
```

Chạy một teacher cụ thể:

```bash
python -B -m src.eval_foundation_teachers \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --teacher siglip2_so400m_384 \
    --batch_size 128
```
