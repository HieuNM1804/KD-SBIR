# DFN5B-224 pseudo-label retrieval and distillation

Branch: `experiment/dfn5b-pseudo-label-retrieval`, based on main
`741ea272b258c9e7b92a46173618e2fe1d401eaa`.

This experiment replaces training image annotations with static predictions from
original frozen DFN5B ViT-H/14 at 224px. Teacher architecture, visual prompts,
retrieval loss, distillation losses, optimizers, schedulers and evaluation match main.
All training hyperparameters are explicit in [the Kaggle run cell](test/kaggle_train.ipy).
The existing argparse training defaults are inherited from committed main; use the
run cell (in particular `--teacher_pretrain_epochs 1`) for the full experiment.

## Three stages

1. **Frozen classification before creating prompts.** Classify every seen sketch
   and photo independently against the ENTIRE seen class-name vocabulary. Templates
   are `a sketch of a {class}.` and `a photo of a {class}.`, matching main. Use
   normalized image/text features and original teacher logit scale. Take top-1;
   no confidence threshold, filtering by ground truth, or label correction.
2. **Teacher prompt retrieval training.** Keep main's independent deep visual
   prompts and frozen backbone. Sample a photo predicted to have the sketch's
   predicted class; batch-hard positives/negatives use those pseudo-labels.
   Teacher objective stays sketch-to-photo triplet only, with no half factor.
3. **Student distillation.** Cache the tuned teacher features, release DFN5B, and
   train student visual prompts. Continue sampling photo pairs by pseudo-label.
   Student losses remain EXACTLY main's
   `0.5 * (KL(T || S) + KL(S || T))` for domain, photo-text and sketch-text.
   The reverse KL updates the student only and does not transpose the retrieval
   candidate matrix. There is no added classification or pseudo-label CE loss.

**Labels are static:** prompts never regenerate them. Known seen class names and
existing seen/unseen split membership are assumed available. True image-folder
categories are used for listing that split and evaluation only, not to select
training positives or construct triplet masks. Both photo and sketch labels are
predicted; photos do not keep their original annotations.

A sketch whose predicted class has no predicted photo positive cannot form a
pair. It is excluded from pair sampling and its count is printed/reported.
Original full feature-bank path ordering is preserved, so cache lookup stays
correct. If fewer than two shared predicted classes remain, training fails with
an explicit message instead of falling back to true labels. Single-class teacher
batches have no negative and are skipped/reported. No class-balanced resampling
or new hard-negative algorithm is introduced.

## Kaggle setup

1. Online notebook, Internet ON: paste [test/kaggle_online.py](test/kaggle_online.py)
   as one cell. It bundles pinned source, original DFN5B-224, original student
   ViT-B/32 and dependency wheels for Python 3.12 and 3.13. It does not train or
   download the image datasets. Optional `REUSE_TEACHER` / `REUSE_STUDENT` paths
   avoid downloading weights already attached to the online notebook.
2. Save Version -> Save & Run All -> Always save output.
3. Offline GPU notebook: attach that output and the desired image dataset.
   Paste [test/kaggle_offline.py](test/kaggle_offline.py) as the first setup cell.
   Small source is copied to `/kaggle/working/KD-SBIR`; checkpoints are linked to
   attached input rather than copied. Kaggle's torch/CUDA stack is retained.
4. Paste [test/kaggle_train.ipy](test/kaggle_train.ipy): `%cd` + `!python -m src.train`.
   The single command runs all three stages. Stage-1 batch size is 256, workers 8;
   teacher training batch size stays 64. `--pseudo_batch_size` changes inference
   batching only, not teacher optimization.

Source pins in setup refer to the runtime commit; the later setup-script commit
can intentionally differ. Python version differences are not hard-blocked: pip
selects compatible wheels from the per-ABI bundle.

### Dataset roots

| `--dataset` | `--root` |
|---|---|
| `sketchy_1` | `/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy` |
| `sketchy_2` | `/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy` |
| `tuberlin` | `/kaggle/input/datasets/b20dccn616nguynhutun/tuberlin` |
| `quickdraw` | `/kaggle/input/datasets/b20dccn616nguynhutun/quickdraw` |

Sketchy-1 and Sketchy-2 share image roots but use different splits. Change root,
dataset and experiment name in the run cell to run another dataset. Keep the
same hyperparameters as the supervised main comparison you want to reproduce.

## Outputs and reuse

- `/kaggle/working/pseudo_labels_dfn5b224/<dataset>_<hash>.pt`: pseudo-labels,
  confidence vectors and original teacher/source/split metadata.
- Same stem `_sketch.csv`, `_photo.csv`: relative path, predicted label, predicted
  class, confidence. No ground-truth labels are used for these assignments.
- Same stem `.json`: predicted class balance, mean confidence, pairable/skipped
  sketch counts and prediction fingerprint.
- `/kaggle/working/teacher_cache_pseudo_dfn5b224/<dataset>_<hash>.pt`: tuned teacher
  prompt state, text targets and full seen feature banks, as in main. The key
  includes exact pseudo-label fingerprint, so supervised or differently labeled
  teacher caches cannot be silently reused.
- `saved_models/<exp_name>/`: student best/last checkpoints; `tb_logs/<exp_name>/`.

Repeat runs reuse static labels; pass `--rebuild_pseudo_labels` to classify again
with the ORIGINAL frozen teacher. `--pseudo_label_path /exact/file.pt` can point
at an attached previous output; CSV/JSON reports are written to the current
`--pseudo_label_dir`. `--rebuild_teacher_cache` retrains teacher prompts.
Changing only student hyperparameters reuses a compatible tuned teacher cache.
Do not point this experiment at a supervised main teacher cache.

## Evaluation protocol retained from main

Teacher prompts are evaluated after each epoch on unseen retrieval and the best
unseen P@K epoch is restored. Student best checkpoint is also selected by unseen
P@K, exactly as main. **These selections use ground-truth unseen evaluation
labels.** The training objectives and positive/negative selection do not use
image annotations, but this inherited evaluation/selection protocol is NOT a
fully label-free model-selection or strict inductive ZS protocol. No claim of
that kind should be made from this branch. It intentionally isolates the effect
of replacing training annotations with frozen-teacher pseudo-labels.

## Local validation

`python -m unittest discover -s tests -p 'test_*.py' -v`

Tests use generated small images, fake frozen classification outputs and small
random networks, not downloaded checkpoints or datasets. They check leakage,
independent modality predictions, pseudo pair/mask selection, feature-bank
indices, incompatible caches, forward teacher triplet and symmetric student KD.
Full DFN5B training and GPU accuracy are to be tested on Kaggle.
