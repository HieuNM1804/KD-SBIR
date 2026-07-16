# EVA01-g-14 + student cls + NT-Xent + distill

Nhánh này dùng teacher frozen:

```text
model: EVA01-g-14
pretrained: laion400m_s11b_b41k
input: 224
output: 1024
```

Loss tổng:

```text
total_loss =
    lambda_cls * student_cls_loss
  + lambda_nt_xent * NT-Xent(student sketch, student photo)
  + lambda_kd * relational_kd_loss
```

Đã loại bỏ:

- student triplet loss
- teacher adapter
- teacher triplet loss
- teacher semantic/classification loss

## Chạy 1 dataset

```bash
python -B -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --exp_name sketchy1_eva01_g14_cls_ntxent_kd
```

## Chạy đủ 4 bộ

```bash
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_1 --epochs 5 --exp_name sketchy1_eva01_g14_cls_ntxent_kd
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_2 --epochs 5 --exp_name sketchy2_eva01_g14_cls_ntxent_kd
python -B -m src.train --root /content/tuberlin/TUBerlin --dataset tuberlin --epochs 5 --exp_name tuberlin_eva01_g14_cls_ntxent_kd
python -B -m src.train --root /content/quickdraw/QuickDraw --dataset quickdraw --epochs 5 --exp_name quickdraw_eva01_g14_cls_ntxent_kd
```

Tham số chính:

```bash
--lambda_cls 1.0
--lambda_nt_xent 1.0
--nt_xent_temperature 0.07
--lambda_kd 3.0
--kd_temperature 0.07
--lr 4e-5
--batch_size 64
```
