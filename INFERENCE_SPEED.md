# Prompted student inference speed: CLIP ViT-B/32 and TinyCLIP

Branch: `experiment/inference-speed-clip-tinyclip`, based on main `741ea27`.

Five students: CLIP ViT-B/32, TinyCLIP ViT-22M/32, 40M/32, 45M/32 and 61M/32.
8M/16 is excluded. Every measured forward includes independent sketch/photo visual
prompts, appending 3 tokens and replacing them through depth 12, followed by
feature normalization, as in main. Prompt values are initialized with main's
seed convention; the benchmark does not train or claim trained retrieval quality.
Teacher, text encoder and loss functions are absent from the timed execution.
Only the vision tower and prompts remain on the GPU.

Three datasets: Sketchy-1, TU-Berlin and QuickDraw, using each source unseen split.
Adjust the three mounted roots in `test/speed_datasets.json` if Kaggle uses different paths.

## Kaggle

1. Internet notebook: paste **all** of `test/kaggle_speed_online.py` in one cell.
   It downloads the five official pretrained backbones and non-CUDA wheels, pins
   Hugging Face revisions and validates release checksums. No dataset download,
   teacher or training. Save the notebook version with its output.
2. GPU notebook: attach that output and the three datasets. **Cell 1:** paste
   `test/kaggle_speed_offline.py`. **Cell 2:** paste `test/kaggle_speed_run.ipy`
   (`%cd` plus `!python`). Source is isolated at `/kaggle/working/KD-SBIR-speed`.
   Online bundles include Linux wheels for Python 3.12 and 3.13. Offline setup installs directly with pip against the actual interpreter and installed Kaggle packages;
   a builder/runtime Python difference alone is not an error.

Send back `/kaggle/working/inference_speed/<timestamp>/summary.json` and
`measurements.csv`. JSON includes every raw latency sample, sampled image paths,
full gallery counts/path fingerprints, GPU/driver/library versions, model parameter
counts, seed, prompt shape and any failures. Partial results are saved after every
model/dataset. A failed model exits nonzero and `complete` is false.

## Interpretation

- `encoder_only`, batch 1: latency of the prompted image encoder with a preprocessed
  image tensor already on GPU. Both sketch and photo are measured independently.
- `encoder_only`, batch 64: synchronized batch latency plus a separate sustained
  throughput block without synchronization after each call. Do not call batch-64
  amortized milliseconds/image single-query latency.
- `search_only`: normalized query feature against the entire resident unseen photo
  gallery, with cosine matrix multiplication and top-200 (or K if smaller).
- `encoder_plus_search`: prompted sketch encoder + normalization + that search.
- `gallery_build_seconds`: with `--gallery real`, the one-time gallery build includes
  CPU image loading, preprocessing, transfers, feature extraction and returning CPU
  features. It is not per-query inference latency.

The main runner uses `--gallery real`: each model encodes the actual full gallery
once per dataset. For a faster encoder-focused run use `--gallery synthetic` (same
actual gallery count, seeded random normalized features, explicitly marked in every
row), or `--gallery none`. Synthetic search does not establish real retrieval speed
for a particular feature distribution. No accuracy metrics are computed in either mode.

Timing uses inference_mode, eval, 30 warmup calls and 100 measured repetitions,
`perf_counter` with CUDA synchronization before/after each latency sample. Disk
I/O, model loading and CPU-to-GPU transfer are excluded from encoder/search timings.
All inputs are 224x224 using the project's RGB resize and CLIP normalization. All
backbones use the selected fp16/fp32 computation precision; prompts remain FP32 as in main.
OpenAI retains FP32 LayerNorm parameters through selective conversion, exactly as main;
HF and auto-pruned loaders retain their branch-specific normalization behavior.
TF32 and cuDNN autotuning are disabled, torch.compile is not used, HF attention is
explicitly eager. Implementations still differ (OpenAI, HF and physically pruned
TinyCLIP); these are implementation comparisons, not FLOP-only architecture rankings.

The same seeded real sample batch is reused during timed repetitions to isolate
compute. Identical 224x224 shapes can give very similar encoder speed on different
datasets; gallery size mainly changes the search stage. This is expected.

`peak_allocated_mib` is PyTorch live allocated GPU memory during the stage, including
resident tensors; it is not an isolated activation footprint or nvidia-smi usage.
Do not compare it across stages as if they used identical resident tensors.

No trained SBIR checkpoint is required for architecture timing. For publication,
identify this as a prompted-architecture benchmark with initialized prompts, or rerun
with the corresponding trained implementations/checkpoints when claiming deployed
method latency. This script does not benchmark SECO-SBIR, SpLIP or their adapters.

Validation: `python -m unittest discover -s tests -p test_inference_speed.py -v`.
Tests cover main CLIP weight restoration, HF/pruned per-layer prompt insertion,
raw-forward equivalence without prompts, deterministic dataset sampling, warmup
exclusion, real/synthetic gallery execution and preservation of failed-run reports.
No local smoke-test timing should be used as the actual Kaggle benchmark result.

References: official TinyCLIP models https://github.com/microsoft/Cream/tree/main/TinyCLIP
and PyTorch asynchronous CUDA timing https://docs.pytorch.org/docs/stable/notes/cuda.html

## Existing bundle reports a Python version mismatch

Copy the updated offline cell from this branch and rerun it first. It accepts the
old manifest and installs packages directly. There is no Python-version gate
or preliminary dry-run. Pip reports any actual missing dependency itself.
If pip reports missing compatible wheels, run the updated online cell with the old
bundle attached as input: it reuses its verified model files and builds fresh wheels
for Python 3.12/3.13. No model redownload is necessary. Save the new output and attach
it instead of the old output; offline remains two cells. If multiple old bundles are
attached to the online notebook, set REUSE_BUNDLE to the exact offline_bundle folder.

Setup regression tests: `python -m unittest discover -s tests -p test_speed_kaggle_setup.py -v`.

Gallery construction defaults to batch 1024 and 8 loader workers (prefetch 1, pinned CUDA input). These affect gallery build time, not the isolated batch-1/batch-64 encoder timings. Lower --workers to 0 if the notebook multiprocessing runtime fails.
