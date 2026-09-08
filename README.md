# CLIP-KD Cross-Domain Visual ICL

This branch adapts Interactive Contrastive Learning (ICL) to zero-shot
sketch-based image retrieval. The first implementation covers the two visual
modalities only:

- student sketch anchors contrast against teacher photo candidates;
- student photo anchors contrast against teacher sketch candidates.

Teacher DFN5B features are detached 1024-dimensional targets. The frozen
OpenAI CLIP ViT-B/32 student produces 512-dimensional features using independent
trainable photo and sketch deep visual prompts. Two train-time-only
`Linear(512, 1024)` projectors map the modalities into the teacher dimension.

## Objective

For an anchor `i`, every candidate with the same category is a positive:

```text
positive(i, j) = label[i] == label[j]

L_sketch_to_photo = -mean_i log(
    sum_{j: positive(i,j)} exp(sim(P_sketch(s_i), teacher_photo_j) / tau)
    -----------------------------------------------------------------------
    sum_j                  exp(sim(P_sketch(s_i), teacher_photo_j) / tau)
)

L_photo_to_sketch is defined in the reverse cross-domain direction.

L_ICL = lambda_icl * 0.5 * (L_sketch_to_photo + L_photo_to_sketch)
```

Using all same-class positives is important for Sketchy: diagonal-only CLIP
cross-entropy would incorrectly treat repeated examples of a category as
negatives. Negatives in this implementation come from the current batch. The
cross-model logit scale is trainable and is initialized as
`log(1 / icl_temperature)`.

This is a cross-domain visual adaptation of CLIP-KD ICL, not its original
image-to-text/text-to-image formulation. It intentionally excludes feature
MSE/cosine loss and text ICL so that the visual ICL contribution can be measured
in isolation.

## Inference

Validation and inference use the raw L2-normalized 512-dimensional student
features. Neither ICL projector nor the teacher is used at inference.

## Kaggle run

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
    --icl_temperature 0.07 \
    --lambda_icl 1.0 \
    --lr 1e-2 \
    --momentum 0.95 \
    --weight_decay 5e-4 \
    --seed 42 \
    --exp_name clip_kd_visual_icl_sketchy2 \
    --progress
```

Training logs are `ICL_SK2PH`, `ICL_PH2SK`, `ICL`, and `train_loss`.
Retrieval evaluation remains `mAP@200` and `P@200` for `sketchy_2`.
