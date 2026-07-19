```bash
!python -m src.train \
    --root /content/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --exp_name sketchy2_teacher_adapter_triplet_baseline
```

## Pretrain the adapter, then distill

The two-stage Sketchy-1 experiment first trains the modality adapters with a
frozen DFN5B teacher, all seen-class images, and fresh sketch/photo
augmentations on every batch. It runs for the requested number of epochs, then
loads the final adapter checkpoint, freezes it, and starts student distillation.

```bash
bash scripts/run_sketchy1_adapter_then_distill.sh /content/sketchy/Sketchy 20
```

The final argument (`20` above) is the exact number of adapter-training epochs.
There is no validation split or early stopping. The student stage uses eight
workers and does not update the pretrained teacher adapter.
