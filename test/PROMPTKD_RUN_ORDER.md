# Kaggle run order: photo-sketch retrieval vocabulary

This experiment uses branch `experiment/promptkd-photo-sketch-prototypes` and
the pinned training commit `cb328cf69815eaaa83b76d539fef0ecaf9178a75`.
Do not use the SGCD online/offline setup scripts for this experiment.

## A. One-time online bundle build

1. Create a Kaggle notebook with Internet enabled. A GPU is not required.
2. Run `test/kaggle_promptkd_online.py` as one cell.
3. Wait for `ONLINE PROMPTKD BUNDLE COMPLETE`.
4. Use **Save Version -> Save & Run All -> Always save output**.
5. Keep that notebook output. It contains source, wheels, ViT-B/32, and DFN5B.

Repeat this section only when the pinned training commit changes.

## B. Offline GPU training

1. Create a new Kaggle notebook with Internet disabled and a GPU enabled.
2. Attach these inputs:
   - `b20dccn616nguynhutun/sketchy`;
   - the saved output from section A;
   - optionally, one prior output containing
     `teacher_cache/sketchy1_teacher_1ep.pt`.
3. Run `test/kaggle_promptkd_offline.py` as one cell.
4. Wait for `OFFLINE PROMPTKD SETUP COMPLETE — READY TO TRAIN`.
5. Run `test/kaggle_promptkd_photo_sketch_train.ipy` as one cell.

On the first run, absence of the optional teacher cache is expected. The train
cell pretrains the teacher and writes the cache under `/kaggle/working`.

## C. Save outputs

After training finishes, save a Kaggle notebook version with outputs. Preserve:

- `/kaggle/working/teacher_cache/sketchy1_teacher_1ep.pt`;
- `/kaggle/working/KD-SBIR-AVKD/tb_logs/promptkd_retrieval_vocab_*`;
- `/kaggle/working/KD-SBIR-AVKD/saved_models/promptkd_retrieval_vocab_*`.

In later sessions, attach this saved output in section B. The offline setup
restores the teacher cache automatically when exactly one matching cache exists.

## D. Required checks before accepting a run

Read the following files under the selected TensorBoard version:

- `retrieval_vocabulary/summary.json`;
- `retrieval_vocabulary/landmarks.csv`;
- TensorBoard metrics `VOCAB`, teacher entropy, mAP, and precision.

The first experiment keeps `retrieval_vocab_descriptor=native`. Therefore its
retrieval metrics remain directly comparable with main while only the training
objective changes.
