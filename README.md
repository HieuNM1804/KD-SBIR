# EVA01-g-14 teacher adapter + NT-Xent

Nhánh này lấy từ `experiment/teacher-eva01-g14`, giữ:

- EVA01-g-14 teacher frozen
- teacher residual adapter
- teacher semantic loss
- student classification loss
- relational KD loss

và thay toàn bộ triplet loss bằng NT-Xent:

- student sketch-photo NT-Xent
- teacher-adapter sketch-photo NT-Xent

Loss tổng:

```text
total_loss =
    lambda_cls * student_cls
  + lambda_nt_xent * student_nt_xent
  + lambda_kd * relational_kd
  + lambda_teacher_nt_xent * teacher_adapter_nt_xent
  + lambda_teacher_semantic * teacher_semantic
```

## Chạy 1 dataset

```bash
python -B -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --exp_name sketchy1_eva01_adapter_ntxent
```

## Chạy đủ 4 bộ

```bash
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_1 --epochs 5 --exp_name sketchy1_eva01_adapter_ntxent
python -B -m src.train --root /content/sketchy/Sketchy --dataset sketchy_2 --epochs 5 --exp_name sketchy2_eva01_adapter_ntxent
python -B -m src.train --root /content/tuberlin/TUBerlin --dataset tuberlin --epochs 5 --exp_name tuberlin_eva01_adapter_ntxent
python -B -m src.train --root /content/quickdraw/QuickDraw --dataset quickdraw --epochs 5 --exp_name quickdraw_eva01_adapter_ntxent
```

Tham số chính:

```bash
--lambda_cls 1.0
--lambda_nt_xent 1.0
--nt_xent_temperature 0.07
--lambda_kd 3.0
--kd_temperature 0.07
--lambda_teacher_nt_xent 1.0
--teacher_nt_xent_temperature 0.07
--lambda_teacher_semantic 1.0
--teacher_adapter_bottleneck 64
--teacher_adapter_lr 2e-5
```
