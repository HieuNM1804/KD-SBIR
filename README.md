# CLIP-KD: four-teacher ensemble distillation

Branch: `experiment/clip-kd-teacher-ensemble4`, based on main commit
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.

The student is the same frozen OpenAI CLIP ViT-B/32 with independent photo and
sketch deep visual prompts. The four teachers are:

| Key | Model / pretrained tag | Feature dimension |
|---|---|---|
| `dfn5b` | ViT-H-14-quickgelu / dfn5b (main's teacher) | 1024 |
| `openai_l14` | ViT-L-14 / openai | 768 |
| `dfn2b_l14` | ViT-L-14 / dfn2b_s39b | 768 |
| `laion_h14` | ViT-H-14 / laion2b_s32b_b79k | 1024 |

Loader definitions and pinned pretrained checkpoint revisions come from the
existing teacher branches. In particular OpenAI means ViT-L/14 at 224 resolution.
There is no augmentation, no projector, and no ensemble at student inference.

## How targets are ensembled

For each of main's three relations, each teacher normalizes its own image/text
features and produces its own cosine matrix. The same relation temperature is
used by each teacher and the student, as on main. Ensemble probabilities are:

```text
P_teacher = sum_m w_m * softmax(R_teacher_m / temperature)
sum_m w_m = 1
```

The four models have equal default weights (0.25 each). `--teacher_ensemble_weights`
accepts non-negative coefficients in `--teacher_models` order and normalizes them.
A coefficient of 0 excludes that teacher from cache creation/loading.

Student loss remains:

```text
L = lambda_domain * KL(P_teacher_sketch_photo || P_student_sketch_photo)
  + lambda_modality * (SymKL(P_teacher_photo_text, P_student_photo_text)
                      + SymKL(P_teacher_sketch_text, P_student_sketch_text))
```

We average probabilities after each teacher's softmax, not raw logits, raw
features, or four independent KL losses. Sketch-photo is row KL (sketch -> photo),
as on main. SymKL is the mean of both KL directions. All four text towers use
`a photo of a {class}.` and `a sketch of a {class}.` with the same ordered seen
class list. Student temperatures and lambda values retain their main meanings.

Feature tensors are concatenated ONLY for compact DataLoader storage (3584 columns
for all four). Loss code splits them into 1024/768/768/1024 blocks before cosine
and softmax; there is no shared teacher feature coordinate system or averaged
embedding. The student retrieval embedding remains 512-dimensional.

## Teacher preparation and caching

Teachers are loaded one at a time. Each teacher independently trains its own
photo/sketch prompts with main's symmetric batch-hard retrieval triplet loss,
using the shared `teacher_*` CLI configuration. For example pretrain_epochs=2
means two epochs PER teacher (four training runs). Then it restores its best
prompt state, caches image/text features, and releases its backbone before the
next teacher. Set `--teacher_pretrain_epochs 0` to ensemble original frozen
pretrained teachers instead; those targets are also cached independently.

All teachers/student use the original main preprocessing. Photo-sketch sampling
remains category-level, with the same image paths/order represented in every
teacher cache. Teacher sampling/RNG is reset independently for each teacher and
restored before student training; changing teacher order does not shift another
teacher's RNG stream. Only student visual prompts enter its optimizer/checkpoint.

Use `--teacher_cache_dir` for multi-teacher runs. An explicit `--teacher_cache_path`
is rejected when several teachers are active. Automatic paths include each
model/pretrained tag and the original full teacher training config. Existing
compatible caches from main and the three teacher branches can be reused by
placing them in this cache directory. Original image/text tensor shapes and
metadata are checked. Changing ensemble weights alone reuses these caches.
Use `--rebuild_teacher_cache` to recreate each active teacher cache intentionally.

Compute/storage: one teacher backbone at a time on GPU, but four teacher training
and feature-extraction passes. The packed CPU image targets use 3584 dimensions
instead of main's 1024 (3.5x feature storage). Packing temporarily needs additional
CPU RAM. The online bundle contains roughly 11 GiB of pretrained weights plus
wheels/source; allow working/output disk space for both the bundle and restored
weights. Existing feature caches avoid repeated teacher training.

## Kaggle setup

1. Internet-enabled notebook: copy and run all of `test/kaggle_online.py` in one
   cell. It downloads all four teachers at fixed HF revisions with SHA256 checks,
   and the original ViT-B/32 student. Save Version -> Save & Run All -> Always save output.
2. GPU notebook: attach Sketchy and that saved notebook output, then copy/run
   `test/kaggle_offline.py`. Internet can be OFF. It restores four HF snapshot
   caches and verifies that OpenCLIP's tags resolve to the intended repositories.
   Use the two scripts from THIS branch; older one-teacher bundles are insufficient.
   Keep builder/restorer Python environments compatible for binary wheels.
   If Git reports `detected dubious ownership` for the mounted input source,
   rerun the updated offline cell: it supplies exact `safe.directory` paths to
   its Git checks and test subprocesses. No global Git configuration is changed,
   and the existing source bundle remains compatible.
3. Run `%cd /kaggle/working/KD-SBIR`, then the command below. It retains your old
   main settings and defaults to an equal-weight four-teacher ensemble.

```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --teacher_models dfn5b openai_l14 dfn2b_l14 laion_h14 \
    --teacher_ensemble_weights 1 1 1 1 \
    --lambda_domain 3.0 \
    --lambda_modality 1.0 \
    --kd_temperature 0.07 \
    --photo_text_kd_temperature 0.15 \
    --sketch_text_kd_temperature 0.02 \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --teacher_pretrain_epochs 2 \
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
    --seed 42 \
    --exp_name ensemble4_sketchy2 \
    --progress
```

Single-teacher ablation: replace the two teacher lines with
`--teacher_models dfn5b --teacher_ensemble_weights 1`. Its loss and gradients match
main exactly. Use one of the other keys to run that teacher by itself. You can
also specify any nonempty subset for pair/triple ensembles. Start with equal
weights; the code does not claim these are the best weights for retrieval.

## Visualize teacher and student distillation matrices in one Kaggle cell

After training, paste the entire contents of
[`test/kaggle_distillation_matrices.py`](test/kaggle_distillation_matrices.py)
into one notebook cell. It works with the existing source bundle pinned to
`d218b11`; rebuilding the online bundle is unnecessary.

Set `CHECKPOINT_PATH` to your trained student checkpoint. The default uses
`saved_models/ensemble4_sketchy2/last.ckpt`; select an `epoch=...ckpt` file to
inspect a different checkpoint. Match `ROOT`, `DATASET`, `TEACHER_MODELS`,
`TEACHER_WEIGHTS`, `SEED`, temperatures and `TEACHER_TRAINING` to your run.
The defaults match the command above. Prompt count and depth are inferred from
the checkpoint. `CACHE_DIR` defaults to `/kaggle/working/teacher_cache`;
`CACHE_PATHS` optionally supplies an exact file path for each teacher. Cache
metadata, feature dimensions and ordered dataset path fingerprints are checked.
Student checkpoints do not record the teacher cache configuration or ensemble
weights, so these settings must be supplied accurately.

`RELATION="sketch_photo"` displays two B-by-B matrices: the four-teacher
probability mixture and the student's row-softmax cosine similarities.
`sketch_text` and `photo_text` instead show B-by-C matrices against **all seen
class text anchors**, using their respective temperatures. Both panels use the
same color scale. The printed KD value matches the corresponding training loss
before its lambda multiplier (one-way KL for sketch-photo; symmetric KL for
image-text). It is a batch diagnostic, not full-gallery retrieval performance.

`BATCH_SIZE`, `EPOCH` and `BATCH_INDEX` reproduce the deterministic training
sampler and same-class photo selection, with identical row/column ordering for
teacher and student. Epoch and batch indices start at zero; sampler epoch is
independent of the checkpoint's epoch. Multiple photos of the same class are
valid positives; the diagonal is not the only correct match. These caches cover
the **seen training split**, not unseen evaluation images.

The cell reconstructs the frozen student backbone and prompts from its saved
weights, reads existing teacher caches, and performs no teacher training or
downloads. It saves a PNG, the two raw probability matrices in NPZ, and JSON
metadata including paths, labels, temperatures, weights and cache selection
under `/kaggle/working/distillation_visualization`.

## Validation

```bash
python -m unittest discover -s tests -p test_teacher_ensemble.py -v
python -m unittest discover -s tests -p test_distillation_visualization.py -v
python -m src.train --help
```

Tests verify normalized teacher weights, independent reference probability mixing
and gradients for all three losses, exact single-DFN5B equivalence to main,
loader/cache settings for the four pretrained models, sequential prompt training
on real tiny OpenCLIP towers, complete cache reuse without downloads/retraining,
cache-model rejection and exact dataset alignment, and actual Lightning steps
where only student prompts update. Full pretrained 4-model Kaggle training and
retrieval accuracy have not been run locally.

Teacher/student checkpoint selection is inherited from main: highest unseen P@K.
Each teacher selects independently; this is not validation of the ensemble's own
retrieval ranking. Because unseen labels participate in model selection, this
baseline protocol is not strict inductive ZS-SBIR. Test-time student retrieval
uses its own cosine similarity and a single 512-dimensional embedding.
