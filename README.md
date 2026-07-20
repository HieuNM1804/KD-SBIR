```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

## CLIP-KD text feature distillation with teacher text adapter

Nhánh `experiment/clipkd-teacher-text-adapter` thêm một linear projector dùng chung
cho student sketch-text và photo-text, cùng một residual adapter dùng chung ở phía
teacher text. Mỗi đầu ra được chuẩn hoá rồi khớp bằng MSE với DFN5B teacher text
feature cùng modality. Hai trọng số độc lập và mặc định bằng 0.

```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --workers 8 \
    --use_teacher_text_adapter \
    --lambda_sketch_text_feature_kd 1.0 \
    --lambda_photo_text_feature_kd 1.0 \
    --exp_name sketchy1_clipkd_teacher_text_adapter
```
