# Prompted encoder cost: DFN5B, CLIP ViT-B/32, TinyCLIP-61M

Branch `experiment/prompted-encoder-cost`, based on main `741ea27`.
This experiment measures the three image encoders in the paper's cost table.
This inference mode performs no training and reports no accuracy. Main's training
files are unchanged. The separate [training-cost mode](TRAINING_COST.md) uses the
same bundle and has its own run cell; it never starts from the inference command.

Every forward includes modality-specific deep visual prompts, projection and
feature normalization. Teacher forward reuses `src/teacher_prompts.py` directly
from main. CLIP-B/32 reuses the project's prompted `clip/model.py`; TinyCLIP-61M
reuses the Hugging Face prompted forward from the TinyCLIP/speed branches.
Only the visual tower and both prompt banks remain in the benchmark model; the
text tower is discarded before timing. No teacher-student losses are constructed.

## Kaggle setup

1. Online notebook, Internet enabled: paste **all** of
   `test/kaggle_encoder_cost_online.py` into one cell. Save Version → Save & Run All
   → Always save output. It bundles pinned source and the three original backbones.
   `REUSE_WEIGHTS` can point to existing attached DFN5B/CLIP checkpoint files and
   the TinyCLIP snapshot directory to avoid downloading those weights again.
   The cell also automatically reuses checksum-verified weights from previous
   `/kaggle/working/encoder_cost_*/offline_bundle` folders, including an interrupted
   bundle. CLIP-B/32 downloads prefer OpenAI's direct Azure Blob origin, with the
   original CDN as fallback, retries, then curl if urllib has TLS failures. HTTPS
   verification remains enabled, and every candidate must match the original
   OpenAI SHA256. Partial downloads are never accepted as completed checkpoints.
2. Offline GPU notebook: attach that output. No datasets are needed for inference. Paste **all**
   of `test/kaggle_encoder_cost_offline.py` into the first cell.
3. Paste `test/kaggle_encoder_cost_run.ipy` into the next cell. This runs all three
   models sequentially on GPU FP16 then CPU FP16. CLIP/OpenCLIP use their original
   mixed precision (FP32 norms/prompts); TinyCLIP uses an FP16 vision tower and
   FP32 prompts, as in its training branch. The same precision policy applies
   on CPU and GPU. There is no automatic fallback to FP32; the requested precision
   is recorded in each measurement.
   DFN5B is constructed with OpenCLIP's FP16 architecture policy, selecting
   `LayerNormFp32` just as main does. Checkpoint tensors are loaded into FP32
   storage before per-device conversion so CPU FP32 preserves original values.
   OpenCLIP's initial random-initialization warning occurs before the explicit
   strict checkpoint load; the benchmark prints confirmation after loading.

The wheel resolver is reused from classification/GZS: Python 3.12 and 3.13 are
resolved independently, including compatible NumPy wheels. PyTorch/CUDA are not
downloaded. Offline setup lets pip select compatible wheels against its interpreter;
an online/offline Python version difference alone does not block execution.
Source and dependency hashes are checked offline; large weight hashes were checked
online and are rechecked offline only if `VERIFY_WEIGHT_HASHES=True`. Weights stay
on attached input through a symlink, avoiding a second multi-GB working copy.

## Run configuration

The run cell is shared by Kaggle and Colab. On Colab, first run
`test/colab_cost_setup.py` (clone/update branch + install runtime), then paste
`test/kaggle_encoder_cost_run.ipy`. The current project directory is detected;
there is no `%cd /kaggle/...` on Colab. Missing original weights are downloaded
and checksum-verified before timing into `/content/encoder_cost_models`.
Subsequent runs reuse verified files. Results default to `/content/encoder_cost`.
No offline Kaggle bundle is needed for Colab.

On Kaggle the same command defaults to offline, using the project's bundled
`models` symlink and `/kaggle/working/encoder_cost` for reports. Missing offline
files cause a local-file error without a network attempt. Mode can be overridden
explicitly using `--download-mode online` or `--download-mode offline` on either
platform. `--models-root` and `--output-dir` are optional explicit overrides.

Default: one sketch query, batch 1, 224×224, teacher 9 tokens/depth 12,
students 3 tokens/depth 12. These are explicit, adjustable arguments; match them
to the configuration used for your reported accuracy before filling the paper.
Change `--teacher-n-ctx 3` if that is the intended experiment.

Input is a fixed seeded synthetic tensor already on the device. No image decoding,
disk reads, host-to-device transfer, gallery encoding, or nearest-neighbor search
is timed. These are **encoder latency** measurements, not end-to-end retrieval.
There is no dataloader and no dataset download or scan. Shape is identical across
datasets, so this table does not require three full dataset runs.

Each device/model has 30 warm-up and 100 measured calls. GPU calls are synchronized
before and after each timed forward. Reported latency is wall-clock milliseconds
per query, including Python launch overhead. Median, mean, p95, standard deviation
and every raw sample are saved. There is no `torch.compile`, CUDA graph, TF32,
or cuDNN autotuning. CLIP/OpenCLIP use PyTorch MultiheadAttention's default
dispatch. TinyCLIP uses Transformers SDPA, matching the supported default in its
training branch with the pinned Transformers/PyTorch stack (not the older speed
branch's forced eager attention). The implementation is recorded per model.
CPU threads default to 8; change `--cpu-threads` explicitly to compare thread counts.
GPU and CPU speedups are computed within each device/dtype, not across devices.

Other runs:

```bash
# Photo path too (not required for a sketch-query cost table).
python -m src.benchmark_encoder_cost --models-root models --modalities sketch photo
# CPU only, e.g. a CPU notebook.
python -m src.benchmark_encoder_cost --models-root models --devices cpu --cpu-threads 8
# FP32 GPU comparison, separate from the default FP16 result directory.
python -m src.benchmark_encoder_cost --models-root models --devices cuda --gpu-precision fp32
# Explicit FP32 on both devices instead of the default FP16.
python -m src.benchmark_encoder_cost --models-root models --gpu-precision fp32 --cpu-precision fp32
```

Prompt values default to seeded initialization, with the same insertion/replacement
mechanism and tensor dimensions as training. Training is unnecessary to measure
this fixed computation graph. These runs must not be described as trained
checkpoint accuracy evaluations. To time your trained prompt values, additionally
pass any of:

```text
--teacher-prompts /kaggle/input/.../teacher_cache.pt
--clip-prompts /kaggle/input/.../student_b32.ckpt
--tiny-prompts /kaggle/input/.../student_tiny61.ckpt
```

Teacher caches must contain `teacher_prompt_state_dict`. Student checkpoints must
contain both `sketch_visual_prompt.*` and `photo_visual_prompt.*` (Lightning
`model.`/`module.` prefixes accepted). Use matching token counts/depth; mismatched
shapes cause an error. Backbone weights remain the pinned frozen originals.

## Parameters and GFLOPs

Two parameter totals are provided explicitly:

- `query_parameters`: one visual tower + the sketch prompt bank used for a query.
- `deployed_parameters`: one visual tower + sketch and photo banks stored for
  retrieval. The shared tower is counted once. No text encoder is included.

Pick one definition consistently in the paper. Individual tower/bank counts are
also exported, so the choice does not require rerunning inference.

GFLOPs count patch convolution, Q/K/V and output projections, both MLP linear
layers, attention QKᵀ and AV products, and final embedding projection. One MAC is
two FLOPs. Bias, normalization, activation, softmax, prompt copies and feature
normalization operations are excluded from this standard dominant-operation count
(but they are included in measured latency). Prompt tokens persist in all blocks,
including blocks after the last prompt replacement. Values are computed from
validated model dimensions and checked against an actual untimed forward using
PyTorch's `FlopCounterMode`, with native/fused attention disabled **only for that
validation pass**. A discrepancy greater than 0.1% stops the run and records the
operation counts for inspection. Do not use a partial run with such an error.

## Results to send back

Under `/kaggle/working/encoder_cost/<UTC_timestamp>/` on Kaggle, or
`/content/encoder_cost/<UTC_timestamp>/` on Colab:

- `table4.csv`: one row per model, params/GFLOPs/dimension, GPU/CPU median latency
  and speedups relative to the prompted teacher.
- `measurements.csv`: per-model/device/modality results, p95 and memory statistics.
- `summary.json`: raw samples, hardware, CPU threads, software versions, complete
  arguments, prompt sources, FLOP breakdown and forward-profile coverage check.

Reports are saved after every measurement, preserving completed rows if interrupted.
All artifacts must come from the same hardware session for direct comparisons.
CPU timing can take longer than GPU timing for DFN5B; progress is printed before
each model/device measurement. The inference model is released between backbones.

Local validation uses tiny randomly initialized versions of the real three
forward implementations and fixture checkpoint files. With CUDA available it
also executes all three FP16 forwards and FLOP checks on GPU, including the
teacher loader's FP32-normalization policy. It does not download or load the full
pretrained models. Full-scale timing is performed on the notebook hardware.
