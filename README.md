```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

## Pretrain the adapter, then distill

The two-stage Sketchy-1 experiment first trains the modality adapters with a
frozen DFN5B teacher and fresh sketch/photo augmentations on every batch. It
stops when held-out seen-class retrieval mAP stops improving, then loads the
best adapter checkpoint, freezes it, and starts student distillation.

```bash
bash scripts/run_sketchy1_adapter_then_distill.sh /content/sketchy/Sketchy
```

The adapter stage defaults to at most 50 epochs with a five-epoch minimum and
five-epoch early-stopping patience. The student stage uses eight workers and
does not update the converged teacher adapter.
