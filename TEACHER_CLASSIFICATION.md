# Frozen teacher classification audit

This branch evaluates the original five teacher checkpoints before pseudo-label
prompt training. It never trains a model, adds teacher prompts, constructs a
student, or retrieves photos. Training code inherited from main is unchanged.

| Key | OpenCLIP architecture | Original pretrained tag |
|---|---|---|
| `dfn5b_h14` | ViT-H-14-quickgelu | dfn5b |
| `dfn2b_l14` | ViT-L-14 | dfn2b_s39b |
| `laion_h14` | ViT-H-14 | laion2b_s32b_b79k |
| `metaclip_l14` | ViT-L-14 | metaclip_fullcc |
| `openai_l14` | ViT-L-14 | openai |

## Kaggle

1. With Internet enabled, paste the whole
   `test/kaggle_teacher_classification_online.py` into one cell. Save its output
   as a notebook version. It bundles source, original weights, and non-CUDA
   wheels for Python 3.12 and 3.13, resolving each interpreter separately.
   It downloads no datasets or student checkpoints. Optional `REUSE_WEIGHTS`
   entries reuse exact checkpoint files from attached older bundles.
2. In the offline GPU notebook, attach that output and the three image datasets.
   Paste `test/kaggle_teacher_classification_offline.py` into the first cell.
   This installs offline wheels and links weights from input without duplicating
   them. Different online/offline Python versions are not blocked in advance;
   pip selects compatible wheels. Installed Kaggle torch/torchvision are retained.
3. Paste `test/kaggle_teacher_classification_run.ipy` as the second cell
   (`%cd` plus `!python`). It evaluates all five teachers on Sketchy-1, Sketchy-2,
   TU-Berlin, and QuickDraw (four protocols over three physical datasets). All images are evaluated by default. Omit unneeded datasets
   from `--datasets` and attach only those you request.

Default batch size is 256, workers 8, prefetch factor 2. Reduce batch size on a
smaller GPU. For quick screening, add `--max-per-class 100`; this uses a fixed
subset for every teacher and marks the output as screening, not full evaluation.
For Sketchy-2 use `--datasets sketchy_2`; it shares Sketchy's root but uses its
own seen/unseen class split. For an alternate root, use exactly one dataset and
`--root /your/path`.

## Protocol

The default is **seen images, classified against every seen class name**. This
matches the pool from which pseudo-label training examples will be obtained.
Ground truth from folder names is used only for split membership and evaluation;
it is not passed to the teacher, used to restrict the class candidates for an
individual image, or used to construct sketch/photo pairs. Photo and sketch are
evaluated independently. Class names and image lists are identical for all five
teachers and are saved in dataset manifests with a reproducible fingerprint.

`--split unseen` runs a separate unseen assessment, using unseen class names by
default; do not use unseen test scores for selecting teachers or hyperparameters.
`--candidate-scope all` explicitly changes the candidate vocabulary to all dataset
classes. Reports record both options. Do not compare accuracies across different
candidate vocabularies as though the task were the same.

Default preprocessing matches main: RGB, resize directly to 224x224, CLIP
normalization. `--preprocess official` uses OpenCLIP's checkpoint preprocessing
instead; report it as a separate experiment. Default text is `a photo of a
{class}.` for photos and `a sketch of a {class}.` for sketches, with underscores
changed to spaces as in main. `--template-mode photo` uses the photo template
for both domains. `--template-mode ensemble` averages normalized text embeddings
from fixed photo/sketch/drawing templates, then normalizes again. These are fixed
templates, never learned prompts. Avoid choosing templates using unseen test data.

All checkpoints are original, frozen weights loaded from local files; inference
does not access model download APIs. The program creates one teacher at a time,
releases it before loading the next, and uses no-grad inference throughout.
Pretrained-tag activation metadata is preserved when loading local files,
including QuickGELU for the OpenAI and MetaCLIP checkpoints.

## Metrics and artifacts

The output directory is `/kaggle/working/teacher_classification/<UTC timestamp>`.
The root contains separate `metrics_photo.csv/json` and `metrics_sketch.csv/json`
for comparing teachers, plus the exact run configuration and dataset manifests.
Each `<teacher>/<dataset>/` directory contains:

- `metrics_photo.json`, `metrics_sketch.json`: overall and pseudo-label metrics.
- `per_class_photo.csv`, `per_class_sketch.csv`: support, predictions, precision,
  recall and F1 per class.
- `confusion_photo.json`, `confusion_sketch.json`: matrix rows are true classes;
  columns are predicted classes.
- `predictions_photo.csv`, `predictions_sketch.csv`: each relative image path,
  true/predicted class, confidence, correctness and top-5 class names.

Metrics are fractions in [0,1], except NLL:

- **Accuracy (top-1):** fraction of all images correctly classified.
- **Macro accuracy / balanced accuracy:** mean recall over evaluated classes with
  images, preventing large classes from dominating the score.
- **Macro F1:** mean per-class F1 over those evaluated classes. An unpredicted
  class receives zero F1. For the default seen protocol all candidate classes
  have support.
- **Top-5 accuracy:** fraction with truth among the five highest scores (or fewer
  if the candidate vocabulary has fewer than five classes).
- **NLL and ECE:** probability quality; ECE uses 15 equal-width confidence bins.
  The native learned logit scale is used unless `--temperature` is specified.
  Confidence and calibration depend on scale/templates/candidate vocabulary.
  Never rank teachers by confidence alone.
- **Pseudo-label quality:** accuracy and retained fraction at confidence >=
  0.5/0.7/0.8/0.9/0.95, plus keeping the 16/64/256 highest-confidence images
  **per predicted class**. Selection does not inspect ground-truth correctness.
  True labels are used only to audit the selected subset's accuracy. Empty
  selections have null accuracy. Reports also show how many classes are predicted.
  `coverage_targets` additionally compares accuracy while retaining the most
  confident 10/25/50/75/100 percent of images, irrespective of each model's scale.

Prioritize macro accuracy and macro F1 in both domains, then accuracy/coverage of
the retained pseudo-labels, especially sketches. Top-5 is a secondary diagnostic;
hard pseudo-labels depend on top-1. Classification quality does not by itself
establish downstream SBIR retrieval quality.

Share `analysis_report.zip` for analysis. It contains the run configuration,
summary metrics, per-teacher metrics, per-class tables and confusion matrices;
large per-image predictions and manifests remain outside the archive.
Completed modality reports are written immediately, so they survive interruption
while a later teacher is running. No training is performed during this audit.

## Local verification (no weights or downloads)

```bash
python -B -m unittest discover -s tests -p 'test_teacher_classification.py' -v
python -m src.evaluate_teacher_classification --help
```

The tests independently check metric values, pseudo-label retention, split
selection and an end-to-end CLI run with tiny generated images and fake models.
