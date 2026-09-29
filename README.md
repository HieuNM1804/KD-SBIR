# CLIP-KD Visual-and-Text ICL for SBIR

This branch adapts Interactive Contrastive Learning (ICL) to zero-shot
sketch-based image retrieval. It replaces the main branch's domain and
image-text distribution matching with four direct student-to-teacher
contrastive directions:

- student sketch anchors contrast against teacher photo candidates;
- student photo anchors contrast against teacher sketch candidates.
- student sketch anchors contrast against teacher sketch-text prototypes;
- student photo anchors contrast against teacher photo-text prototypes.

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

L_visual = 0.5 * (L_sketch_to_photo + L_photo_to_sketch)
L_text   = 0.5 * (L_sketch_to_sketch_text + L_photo_to_photo_text)

L_ICL = lambda_icl * (
    lambda_icl_visual * L_visual + lambda_icl_text * L_text
) / (lambda_icl_visual + lambda_icl_text)
```

Using all same-class positives is important for Sketchy: diagonal-only CLIP
cross-entropy would incorrectly treat repeated examples of a category as
negatives. Visual negatives come from the current batch. Text candidates are
all seen-class teacher prototypes, so each image has its class prototype as the
positive and every other seen class as a negative. The cross-model logit scale
is shared by all four directions, is trainable, and is initialized as
`log(1 / icl_temperature)`.

The text templates are modality specific: `a sketch of a <class>.` and
`a photo of a <class>.`. Teacher image and text targets are detached. The same
separate sketch/photo projectors are used for both the visual and text axes,
which places student images in the teacher's 1024-dimensional space without
adding another trainable head. Set `--lambda_icl_text 0` for the former
visual-only experiment or `--lambda_icl_visual 0` for a text-only ablation.

The branch intentionally excludes feature MSE/cosine loss and the main branch's
KL distribution losses, so the ICL contribution is measured in isolation.

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
    --lambda_icl_visual 1.0 \
    --lambda_icl_text 1.0 \
    --lr 1e-2 \
    --momentum 0.95 \
    --weight_decay 5e-4 \
    --seed 42 \
    --exp_name clip_kd_visual_text_icl_sketchy2 \
    --progress
```

Training logs are `ICL_SK2PH`, `ICL_PH2SK`, `ICL_SK2TX`, `ICL_PH2TX`,
`ICL_VIS`, `ICL_TXT`, `ICL`, and `train_loss`. Retrieval evaluation remains
`mAP@200` and `P@200` for `sketchy_2`.
