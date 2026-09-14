# Sketch-region correspondence KD

Branch: `experiment/sketch-region-correspondence-kd`.
Pinned Kaggle training source: `065b8d26e86cd81d2fc61d93cd76892654ec160d`.
Parent: `experiment/semantic-region-attention-kd` at
`c03b39e507074fc20907ed80799e295d91fd46c1`.
This implements direction 1 only. Attribute prompts and counterfactual
interventions are future experiments, not implemented contributions.

## What is implemented

- Original DFN5B ViT-H/14 teacher: 32 visual blocks, hidden width 1280,
  output 1024. Student CLIP ViT-B/32: 12 blocks, width 768, output 512.
- Teacher grid crops are resized/encoded as images, not raw teacher attention.
  Student dense local final-block readout/area pooling comes from the parent.
- Raw global/crop teacher targets remain 1024-D. There is no Procrustes Q and
  no coordinate-wise MSE between incompatible features.
- Intramodel cosine relations give a shared query/candidate/region decision
  space. Region matching is row-conditional softmax with a no-match bin;
  it is NOT one-to-one OT or semantic part segmentation.
- Ground-truth symmetric multi-positive retrieval trains the descriptor
  actually used at inference: `normalize(native + beta*fusion(region_pool))`.
  The zero-initialized fusion initially reproduces the seeded native student.
- Teacher global/local scores supervise selected candidate ranking only when
  the teacher positive beats the mined negatives. FG mining prefers negatives
  of the same category. Rejected teacher queries still receive GT supervision.
- Matching KD is weighted by ink visibility and positive-minus-negative local
  evidence. Ink mass/no-match threshold are heuristics, not learned visibility
  or causal confidence. The crop granularity and teacher quality may limit gains.
- All loss similarities/KL computations are FP32 under outer AMP. No adaptive
  average pooling is used. Optional teacher tuning checkpoints each visual
  block, keeps prompt parameters FP32, and selects on held-out seen data only.

Loss: `L_GT + lambda_rank*L_rank + lambda_correspondence*L_match`.
Defaults: 1, 0.5, 0.25. These are pilot choices, not tuned best settings.
Track `train_teacher_acceptance`, `train_evidence_fraction`, each component,
descriptor drift and correction norm. Low KD acceptance/evidence is a reason
to inspect the teacher, not blindly increase lambda.

## Positives, negatives and evaluation

`category`: same-category positives; same-category random photo selection is
never called an exact pairing. Seen validation independently holds out both
sketch queries and photo gallery items per category. Full mAP and P@100 match
the metric form of the original Sketchy1 pipeline, but the training/selection
split, optimizer, sampler and head differ. Re-run matched controls; do not
attribute an unmatched main-vs-new gain to correspondence alone.

`fg`: each `<photo-id>-<sketch-id>` filename must resolve to a unique basic
photo. Extended photos are rejected; 100 photos/category are required unless
`--fg_photos_per_category 0` explicitly requests a custom dataset. Hold-out
photo instances and ALL their sketches are excluded from fitting. Training
uses paired class-balanced batches, NOT the previous 100-photo full-gallery
training loop. Negatives/steps must be identical across new-method controls.

FG default `--fg_gallery category` evaluates exact instance Acc@1/Acc@5 with
known-category galleries, matching the previous FG evaluation restriction.
Seen selection galleries contain only held-out photos (roughly 10/category),
so their accuracy is not comparable to full 100-photo train accuracy. Unseen
basic evaluation uses all 100 photos/category. `--fg_gallery all` explicitly
evaluates against the entire gallery instead; do not mix these results.

Neither teacher nor student checkpoint selection uses unseen metrics. Unseen
evaluation runs after selecting the student on seen validation. Teacher unseen
audit is diagnostic only and occurs after teacher selection. VLM pretraining
data overlap has not been audited; no strict pretraining-unseen claim is made.

Balanced sampling is seeded with the restored epoch and samples instances
without replacement within a class when enough are available. It does not
visit every sketch exactly once. Report seeds, sample budget and active params.

## Kaggle: local source, no GitHub push required

1. Upload `sketch-region-correspondence-source.bundle` from the supplied ZIP
   as a Kaggle dataset/file and attach it to an Internet-enabled notebook.
2. Paste `test/kaggle_online.py`. It clones the uploaded branch, checks the
   source pin, downloads verified CLIP/DFN weights and pinned dependency wheels.
   Save & Run All with output retained. The bundle is `correspondence_bundle`.
   If no source.bundle is attached, the builder falls back to GitHub and needs
   the new branch already pushed. The local branch has NOT been pushed by this task.
3. Attach the online notebook output and Sketchy/Sketchy-FG dataset to an
   offline notebook. Enable a GPU and paste `test/kaggle_offline.py`.
   It verifies hashes, restores weights and runs no-download CUDA/CPU tests.
   Existing `/kaggle/working/KD-SBIR` is renamed to a timestamped backup,
   never recursively deleted. Runs/cache live outside that project by default
   in the commands below.
4. Run the preparation command, then fresh matched controls. Run
   `test/kaggle_correspondence_report.py` to package JSON/CSV/TensorBoard
   diagnostics; download checkpoint files separately.

### Preparation: category Sketchy1

```python
%cd /kaggle/working/KD-SBIR
!python -m src.train_correspondence \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
  --dataset sketchy_1 --retrieval_protocol category \
  --output_dir /kaggle/working/correspondence_runs \
  --teacher_pretrain_epochs 0 --prepare_only --audit_teacher
```

Default 0 uses original frozen DFN5B. To tune teacher prompts, set
`--teacher_pretrain_epochs 2 --teacher_n_ctx_visual 10
--teacher_prompt_depth 12 --teacher_prompt_lr 2e-5` during BOTH preparation
and subsequent runs, with a different `--correspondence_cache_dir`.
Teacher SGD choices need experiments; n_ctx/depth are not teacher block count.
If tuning OOMs, reduce `--teacher_pretrain_batch_size` to 16 (divisible by 4).

### Student: category

```python
!python -m src.train_correspondence \
  --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
  --dataset sketchy_1 --retrieval_protocol category \
  --output_dir /kaggle/working/correspondence_runs \
  --correspondence_mode teacher --exp_name category_correspondence_s42 \
  --batch_size 64 --classes_per_batch 4 --epochs 5 --seed 42 \
  --n_ctx_visual 3 --prompt_depth 12 --region_grid 2 \
  --region_head_lr 1e-3 --lambda_retrieval 1 \
  --lambda_rank 0.5 --lambda_correspondence 0.25
```

### FG exact-instance variant

Use these replacements in BOTH commands above:

```text
--root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg
--dataset sketchy_2 --retrieval_protocol fg --fg_gallery category
--exp_name fg_correspondence_s42
```

The root must directly contain sketch/ and photo/. Basic pairing/count
validation fails early if an extended/unpaired dataset is attached.

### Matched controls

Use separate fresh `exp_name` values and change ONLY `correspondence_mode`:

| Mode | Target |
| --- | --- |
| gt | GT loss only; no teacher required during this run |
| global | GT + teacher global candidate ranking, matching KD off |
| uniform | GT + combined teacher ranking + uniform visible-region matching, same dustbin mass |
| teacher | GT + combined teacher ranking + evidence-weighted teacher matching |
| shuffled | Same ranking as teacher, but matching targets from another sketch |

All keep the same region-head architecture, native prompts, optimizer,
initialization, sampler and budgets. The meaningful matching comparison is
teacher vs uniform/shuffled; global checks extra local teacher information.
Optional `--region_train_prompts --lr 1e-4` must be applied to every control
in its own comparison series. Head-only is the initial series.

Resume with `--ckpt_path /.../last.ckpt`, identical method/cache/source config,
and `--epochs` set to desired total. Model, Adam moments, scheduler and sampler
epoch are restored. Existing checkpoints are not overwritten by fresh runs.

## Inference and tests

```python
from src.correspondence_model import load_correspondence_checkpoint
module = load_correspondence_checkpoint('/.../best-....ckpt', device='cuda')
with torch.inference_mode():
    descriptor = module.model.extract_feature(images.cuda(), 'sketch')
```

Use ordinary `normal_transform(224)`. Loading/inference need no teacher, crop
cache, external weight download, category names or test labels. Known-category
FG gallery restriction is an evaluation choice, not an input to the encoder.

```text
python -m unittest discover -s tests -p test_correspondence*.py -v
python -m unittest discover -s tests -p test_semantic_region*.py -v
python -m src.train_correspondence --help
```

Fixtures use small real CLIP/OpenCLIP backbones, not downloaded DFN5B or the
real dataset. Tests passing do not establish novelty, significant improvement
or Kaggle performance. Run at least 3 seeds and proper matched controls.
