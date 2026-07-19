```bash
!python -m src.main_train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --lambda_sketch_text_kd 1.0 \
    --lambda_photo_text_kd 1.0 \
    --text_kd_temperature 0.07 \
    --exp_name sketchy2_text_relational_kd
```

`--lambda_sketch_text_kd` weights class-relation KD computed only from student
and teacher sketch-text features. `--lambda_photo_text_kd` independently
weights the same objective for photo-text features. The legacy
`--lambda_text_kd` option remains available and sets both weights to the same
value.
