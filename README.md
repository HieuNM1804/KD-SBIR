```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

## CLIP-KD text feature distillation

Nhánh `experiment/clipkd-text-feature-mse` thêm một linear projector dùng chung cho
student sketch-text và photo-text. Mỗi đầu ra được chuẩn hoá rồi khớp bằng MSE với
DFN5B teacher text feature cùng modality. Hai trọng số độc lập và mặc định bằng 0,
vì vậy cấu hình cũ không đổi nếu không truyền hai cờ mới.

```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_1 \
    --epochs 5 \
    --workers 8 \
    --batch_size 32 \
    --lr 2e-5 \
    --lambda_sketch_text_feature_kd 1.0 \
    --lambda_photo_text_feature_kd 1.0 \
    --exp_name sketchy1_clipkd_text_feature
```
