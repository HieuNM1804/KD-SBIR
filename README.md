# CLIP-KD Masked Feature Distillation for SBIR

This branch implements the MFD method from CLIP-KD as a controlled experiment
for the photo and sketch visual branches.

- Teacher: frozen DFN5B ViT-H/14, full unmasked input, 1024D image feature.
- Student: frozen CLIP ViT-B/32 with independent photo/sketch visual prompts.
- Student input: MAE-style random patch masking, independently sampled for each
  image on every training forward.
- Alignment: separate trainable `Linear(512, 1024)` projectors for photo and
  sketch.
- Objective: selectable cosine or MSE between L2-normalized projected
  masked-student features and detached full-image teacher features. Cosine is
  the default on this experimental branch.
- Evaluation: unmasked images and raw 512D student embeddings. The MFD
  projectors are train-time only.

The implementation follows the CLIP-KD/MAE encoder masking order:

1. Convert the image into patch tokens.
2. Add each patch's positional embedding.
3. Generate uniform random noise separately for every sample and sort it.
4. Keep the first `floor(L * (1 - mask_ratio))` patch tokens.
5. Prepend the unmasked CLS token and continue through the visual transformer.

Masked patches are removed from the encoder sequence. No mask token or MAE
decoder is needed because MFD distills CLIP's global CLS-derived feature.

```text
s_photo = photo_projector(student(mask(photo)))
s_sketch = sketch_projector(student(mask(sketch)))

L_MFD = lambda_mfd * 0.5 * (
    feature_loss(normalize(s_photo), normalize(stopgrad(teacher(photo))))
  + feature_loss(normalize(s_sketch), normalize(stopgrad(teacher(sketch))))
)
```

Set `--mfd_loss cosine` for `mean(1 - cosine_similarity)` or
`--mfd_loss mse` to reproduce the original normalized-MSE implementation.
For 1024D unit features, normalized MSE equals `2 / 1024` times cosine loss,
so the two weights must not be compared at the same numerical scale.

The teacher cache contains full-image teacher targets and remains reusable; the
mask ratios do not belong in its cache key.

## Example run

The CLIP-KD reference MFD experiment uses a mask ratio of `0.75`. This branch
exposes separate ratios so photo and sketch can also be ablated independently.

```bash
python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 20 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --teacher_pretrain_epochs 1 \
    --teacher_pretrain_batch_size 64 \
    --teacher_n_ctx_visual 10 \
    --teacher_prompt_depth 12 \
    --teacher_prompt_std 0.02 \
    --teacher_prompt_lr 3e-2 \
    --teacher_prompt_seed 42 \
    --teacher_prompt_gradient_checkpointing \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --lambda_teacher_retrieval 1.5 \
    --teacher_triplet_margin 0.2 \
    --photo_mask_ratio 0.75 \
    --sketch_mask_ratio 0.75 \
    --mfd_loss cosine \
    --lambda_mfd 1.0 \
    --lr 1e-2 \
    --momentum 0.95 \
    --weight_decay 5e-4 \
    --seed 42 \
    --exp_name clip_kd_mfd_cosine_r075_sketchy2 \
    --progress
```

For the mask-ratio ablation, hold all other parameters fixed and test `0.0`,
`0.25`, `0.5`, and `0.75`. The `0.0` run is the no-mask control through the
same code path. Training logs are `MFD_PHOTO`, `MFD_SKETCH`, `MFD`, and
`train_loss`; retrieval metrics remain `mAP@200` and `P@200` for `sketchy_2`.

References: [CLIP-KD](https://arxiv.org/abs/2307.12732),
[official CLIP-KD code](https://github.com/winycg/CLIP-KD), and
[official MAE code](https://github.com/facebookresearch/mae).
