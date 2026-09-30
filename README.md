# Five Text-Prompt Pairs on the Main Baseline

Branch: `experiment/clip-kd-text-prompt-sweep`, based on main commit
`b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.

This experiment varies fixed class-text wording in both the DFN5B teacher and
CLIP ViT-B/32 student. Both models use the same pair of templates, encoded by
their own frozen text encoders. The teacher and student each retain main's
independent photo/sketch deep visual prompts. All CLIP backbone weights,
including LayerNorm, remain frozen. Main's domain and modality KD losses are
unchanged.

| Pair | Photo template | Sketch template |
| --- | --- | --- |
| `baseline` | `a photo of a {class}.` | `a sketch of a {class}.` |
| `depicting` | `a photograph depicting a {class}.` | `a drawing depicting a {class}.` |
| `shows` | `this photograph shows a {class}.` | `this drawing shows a {class}.` |
| `line_drawing` | `a photograph of a {class}.` | `a line drawing of a {class}.` |
| `hand_drawn` | `an image showing a {class}.` | `a hand-drawn sketch of a {class}.` |

## Kaggle setup

Run [test/kaggle_online.py](test/kaggle_online.py) in an Internet-enabled Kaggle
notebook. Save the notebook output with Save Version -> Save & Run All ->
Always save output. Attach that output and the Sketchy dataset to a GPU notebook
with Internet disabled, then run [test/kaggle_offline.py](test/kaggle_offline.py).
An existing gradient/AFD/shared-prompt bundle does not contain this branch.

## Run all five pairs

After the offline setup, run this notebook cell:

```python
%cd /kaggle/working/KD-SBIR
!python -u test/kaggle_text_prompt_sweep.py \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 8 \
    --output_dir /kaggle/working/text_prompt_results
```

The runner launches five fresh training processes. Every pair gets the exact
settings below, with seed 42, one teacher-pretraining epoch and five student
epochs. Teacher pretraining uses retrieval triplet loss, which is independent
of the text wording. The runner repeats that stage with the same seeds for each
pair and overwrites one cache file to limit disk use. The cache identity and
metadata include both templates; old caches cannot silently supply wrong text
targets.

The original example said `--dataset sketchy_2` but used an experiment name
ending in `sketchy1`. Run names now include `sketchy_2` and the template-pair ID.

## Training settings / run one pair

To train just one pair, change `--text_prompt_pair` below:

```bash
!python -u -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 5 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --teacher_pretrain_epochs 1 \
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
    --lambda_domain 3.0 \
    --lambda_modality 1.0 \
    --photo_text_kd_temperature 0.15 \
    --sketch_text_kd_temperature 0.02 \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --seed 42 \
    --text_prompt_pair line_drawing \
    --results_path /kaggle/working/line_drawing_result.json \
    --exp_name text_prompt_line_drawing_sketchy2 \
    --progress
```

## Results and selection

The sweep writes to `/kaggle/working/text_prompt_results`:

- `summary.csv`: one row per completed pair, with best epoch, P@200, mAP@200
  from that same epoch, templates, and checkpoint path. Scores are fractions
  from 0 to 1.
- `best_run.json`: winning pair, complete configuration, trained epoch history,
  checkpoint path, and whether all five pairs have completed.
- `best.ckpt`: copy of the winning student checkpoint after all pairs finish.
- `runs/<pair>.json`: all trained epoch metrics and best/last checkpoint paths.
- `logs/<pair>.log`: full stdout/stderr from each training process.
- `sweep_manifest.json`: source fingerprint, common settings, and template grid.

Selection follows main: highest unseen P@200 for `sketchy_2`. mAP@200 is
reported at the selected epoch, rather than maximized independently. Exact
precision ties keep the first epoch and then the first pair. Lightning sanity
validation is excluded. Teacher epoch selection and student epoch selection
also follow main's unseen precision protocol. Consequently, the final result
is selected on the unseen retrieval split, not an independent held-out final
test estimate.

Results are saved after each completed pair, so an interrupted sweep retains
its partial summary. Continue with:

```python
!python -u test/kaggle_text_prompt_sweep.py \
    --dataset sketchy_2 --epochs 5 --workers 8 \
    --output_dir /kaggle/working/text_prompt_results --resume
```

Resume requires identical source, configuration and templates, and checks that
completed runs still have their best checkpoints. Completed pairs are skipped;
an unfinished pair restarts from scratch. Keep the same project checkout and
saved_models directory. Use a new output directory for a different study.

This repository does not contain measured sweep scores. Run the experiment on
Kaggle to obtain `summary.csv` and `best_run.json`.

## Local checks

```bash
python -m unittest discover -s tests -v
python -m src.train --help
python test/kaggle_text_prompt_sweep.py --dry_run
```

Tests use small fake encoders and simulated training output to verify template
routing, cache separation, trained-epoch selection, checkpoint export, and
resume behavior. Those simulated scores are test fixtures, not experiment
results.
