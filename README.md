# TinyCLIP raw-inference suite for SBIR

This branch evaluates five official TinyCLIP checkpoints on the same
sketch-to-photo retrieval split:

- TinyCLIP ViT-8M/16 Text-3M
- TinyCLIP ViT-22M/32 Text-10M
- TinyCLIP ViT-40M/32 Text-19M
- TinyCLIP ViT-45M/32 Text-18M
- TinyCLIP ViT-61M/32 Text-29M

The benchmark performs **no training or prompt learning**. Every checkpoint is
frozen, `torch.inference_mode()` is active, and raw normalized image embeddings
are compared by cosine similarity. The default protocol uses the unseen classes
of `sketchy_1`, reports mAP@all and P@100, and processes the entire gallery.

The three standard checkpoints (8M, 40M, and 61M) use pinned Hugging Face
snapshots. The official auto-pruned `.pt` checkpoints are used for 22M and 45M.
All inputs use 224x224 resolution and the OpenAI CLIP mean/std.

```bash
python -m src.infer_tinyclip_suite \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_1 \
    --scope unseen \
    --models-root /kaggle/working/tinyclip_models \
    --batch-size 256 \
    --workers 4 \
    --output-dir /kaggle/working/tinyclip_inference_results
```

The command writes `tinyclip_inference_results.json` and
`tinyclip_inference_results.csv`. Forward latency excludes image loading and
retrieval ranking; it measures only batched image-encoder execution after warmup.

Use `--scope all` to evaluate every category shared by `sketch/` and `photo/`.
Use `--models 8m,40m,61m` for a subset. `--max-per-class` exists only for quick
smoke tests and defaults to zero, which means every image is evaluated.

For fully offline Kaggle execution, run
`test/kaggle_tinyclip_suite_online.py` once in an Internet-enabled notebook and
save its output. Attach that output and the Sketchy dataset to the GPU notebook,
then run `test/kaggle_tinyclip_suite_offline.py`. The offline script validates
every wheel and checkpoint before launching this inference entrypoint.
