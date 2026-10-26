# Main training cost: teacher adaptation, cache, student distillation

This is the second, independent mode of `experiment/prompted-encoder-cost`.
It runs the existing DFN5B teacher + CLIP ViT-B/32 student pipeline from main
`741ea27`, then reports synchronized single-GPU wall times. It does not train
TinyCLIP or change any loss, optimizer, sampler, scheduler or checkpoint selection.

## Kaggle

Use the **updated** `test/kaggle_encoder_cost_online.py` and
`test/kaggle_encoder_cost_offline.py`, shared with inference. The bundle now also
includes Lightning, torchmetrics and TensorBoard runtime wheels. Rebuild an older
bundle with the new online cell; `REUSE_WEIGHTS` can reuse the same three original
backbones, avoiding weight downloads. No dataset is downloaded into the bundle.

After offline setup, attach the Sketchy dataset and paste the complete
`test/kaggle_training_cost_run.ipy` into a separate cell. The supplied run is
Sketchy-Ext-2, teacher 1 epoch, student 5 epochs, teacher 9 tokens, student 3 tokens,
depth 12. It explicitly uses the parameters previously requested for Sketchy-2:
teacher LR 0.03, student LR 0.11, SGD momentum 0.9/0.95, teacher triplet margin
0.3, lambda teacher retrieval 1.5, domain/modal weights 2/2, temperatures
0.07/0.15/0.02, seed 42, batch 64 and validation batch 1024. Change the cell to
the configuration whose cost you intend to report. The benchmark forwards all
normal training arguments unchanged to `src.train`.

Inference remains a separate command/cell, `test/kaggle_encoder_cost_run.ipy`;
it never starts training. Training retains main's precision policy rather than
applying the inference benchmark's CPU/GPU precision options.

## Stage boundaries

`src.train`, `src.model`, `src.losses`, `src.teacher_prompts`, `src.dataset`, and
`src.data_config` remain byte-for-byte the main implementations. The benchmark
entrypoint invokes `src.train` through `runpy` and temporarily wraps its original
methods, restoring them when execution ends. The original model loaders are
redirected to the pinned DFN5B and OpenAI files already in the offline bundle.
There is no network fallback during training.

Reports distinguish:

| Stage | Included work |
| --- | --- |
| `pipeline` | Entire invoked main pipeline, including imports, dataset setup and all stages below |
| `model_initialization` | Original teacher/student loading and prompt/text-bank setup in the Lightning model constructor |
| `teacher_adaptation` | Original prompt pretraining, data loading, all teacher validations and best-prompt restoration |
| `teacher_validation` | Nested subset: validation encoding and retrieval metrics per teacher epoch |
| `teacher_feature_cache_build` | Teacher preparation minus adaptation: text targets, seen sketch/photo feature encoding, transfers, serialization and teacher release |
| `teacher_cache_load` | Loading an existing compatible cache, if explicitly requested |
| `student_distillation` | Original `Trainer.fit`, including startup, sanity validation, train/validation epochs and checkpoint writes |
| `student_validation` | Nested subset: normal validation loops within `Trainer.fit` |
| `student_sanity_validation` | Nested subset: Lightning's initial sanity check |

Do **not** sum the nested validation rows with their parent stages. The three
non-overlapping primary costs are teacher adaptation, cache build and student fit.
Their sum excludes initialization and dataset/setup overhead, which are already
included in the separately measured total `pipeline`.

Timing uses a monotonic clock and CUDA synchronization at stage boundaries. A
single GPU is used, so GPU-hours = elapsed wall seconds / 3600. This is allocated
single-GPU pipeline time, including CPU work and I/O inside each stage; it is not
GPU kernel busy time or an energy measurement. Boundary synchronization and report
writes introduce small measurement overhead, included in enclosing stage totals.

## Cache and checkpoint handling

By default each run gets a fresh path:

```text
/kaggle/working/teacher_cache_training_cost/teacher_<run_id>.pt
```

Thus the default run really fine-tunes the teacher, even if a cache from a previous
experiment is present elsewhere. Existing cache files are not overwritten by
default. To intentionally rebuild a specified file, use main's
`--teacher_cache_path ... --rebuild_teacher_cache`.

To time student training with a compatible existing cache, explicitly pass:

```text
--teacher_cache_path /kaggle/input/.../teacher_cache.pt --allow-cached-teacher
```

Use the teacher parameters that match that cache's metadata. Such a run records
`cache_hit=true`, leaves teacher adaptation/cache-build timing as `not_run`, and
sets `full_fresh_training_measurement=false`; it cannot supply the full teacher
training cost. Resumed student runs are likewise marked as not full fresh runs.

Experiment names get a unique run-id suffix so logs and checkpoints from previous
measurements are preserved. The effective experiment name and actual checkpoint
paths are recorded. The feature cache and checkpoints are outside the report
directory, so sharing the JSON/CSV does not require copying large weights.

## Output

Under `/kaggle/working/training_cost/<UTC_run_id>/`:

- `training_cost.csv`: stage times, GPU-hours, status and nested-validation markers.
- `training_cost.json`: the CSV summary plus individual timing events, all effective
  training arguments, command, original source hashes, hardware/software, dataset
  image/class counts, teacher best epoch/precision and student checkpoint/metrics.

The JSON also contains `training_gpu_hours_including_validation_excluding_cache`
and `training_and_cache_gpu_hours`, each computed without double counting.
Reports are written after each stage. Python exceptions and interruptions preserve
partial times and a failed/interrupted status. A graceful Lightning interrupt is
also marked interrupted. Force-killing the process can leave a stage as `running`;
use only completed runs for final cost claims. Cached and resumed runs are clearly
marked. Send the two files back for analysis, along with inference reports if both
modes were run.

Local tests exercise the full original teacher pretrain -> cache -> student flow
on tiny randomly initialized CPU models with fixture images. They also verify
nested timing arithmetic, interruption reports, cache-hit labeling and offline
weight redirection. They do not download pretrained models or replace the required
full-size single-GPU Kaggle measurement.
