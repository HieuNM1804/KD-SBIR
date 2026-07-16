# EVA01-g-14 distill-only ablation

Nhánh này giữ student distill-only và chỉ dùng một loss:

```text
total_loss = lambda_kd * relational_kd_loss(student sketch-photo, teacher sketch-photo)
```

Đã loại bỏ:

- student classification loss
- student triplet loss
- teacher adapter
- teacher triplet loss
- teacher semantic/classification loss

Teacher frozen:

```text
model: EVA01-g-14
pretrained: laion400m_s11b_b41k
input: 224
output: 1024
```

## Chạy 1 dataset

```bash
python -B -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --exp_name sketchy1_eva01_g14_distill_only
```

## Chạy đủ 4 bộ

```bash
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_1 --epochs 5 --exp_name sketchy1_eva01_g14_distill_only
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_2 --epochs 5 --exp_name sketchy2_eva01_g14_distill_only
python -B -m src.train --root /content/tuberlin/TUBerlin --dataset tuberlin --epochs 5 --exp_name tuberlin_eva01_g14_distill_only
python -B -m src.train --root /content/quickdraw/QuickDraw --dataset quickdraw --epochs 5 --exp_name quickdraw_eva01_g14_distill_only
```

Tham số chính còn lại:

```bash
--lambda_kd 3.0
--kd_temperature 0.07
--lr 4e-5
--batch_size 64
--epochs 5
```
