# Cross-modal evidence perturbation distillation

Experimental branch from main `b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6`.
The two original main losses and sampler are preserved. This version implements
photo-only auxiliary training, compact disk targets, and a teacher audit.
It does not implement online targets, dynamic student-aware region mining or
sketch perturbation; these are follow-up experiments, not claims of this release.

## Method and limits

The nearest prior is Xiao et al., *Masked Images Are Counterfactual Samples for
Robust Fine-Tuning*, CVPR 2023:
https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_Masked_Images_Are_Counterfactual_Samples_for_Robust_Fine-Tuning_CVPR_2023_paper.html
That method uses CAM masking/refilling and pretrained feature distillation.
Our static masked-score control is an adaptation, not an exact reproduction.
No claim of novelty, causal identification or better Sketchy retrieval is made.

For M in {teacher, student}, normalize each reference image embedding and average
within each train class, then normalize again to obtain mu_M[c]. Use exactly the
same fixed reference images for both models (default: 8 sketches per class).
Teacher and student use their own embedding spaces. Student prototypes are
recomputed at each epoch start, detached and frozen for that epoch. They are not
saved into the student checkpoint and are reconstructed after loading weights.

q_M(x)[c] = cosine(z_M(x), mu_M[c])
m_T(x,y) = q_T(x)[y] - max_{c != y} q_T(x)[c]
r_M(x,r) = q_M(x) - q_M(mask(x,r))
L_evidence = mean_{selected samples, classes} Huber(r_S - stopgrad(r_T), delta=1)
L_total = lambda_domain L_domain + lambda_modality L_modality + lambda_evidence L_evidence

There is no hidden temperature, projector or response rescaling. Huber uses raw
cosines and can be much smaller than the main KL terms. Log teacher/student
response RMS and validate the auxiliary weight; 1.0 is only a starting value.
Prototype scores remain a similarity-based interface. The hypothesis concerns
the effect of removing evidence on the same image, not static image-pair logits.
Training-only class prototypes do not add an inference head.

Each photo has five equal-area square candidates: four corners and center.
Default area is 25%; coordinates do not depend on either ViT patch grid.
All views are made deterministically on CPU after the unchanged main transform.
Mean fill uses the region's channel means (equivalent to doing it before the
affine CLIP normalization). Donor fill uses the same-coordinate region of one
recorded train photo from another class. Crop uses bilinear resize. This crop
transform differs from the main image resize and is explicitly recorded.

Only photos with positive clean teacher margin are eligible. Important regions
must reduce correct-class margin by at least 0.01 and have positive crop margin.
Stable regions have clean-vs-masked score RMS <= 0.01. Important chooses maximal
drop; stable chooses minimal RMS. With `both`, each attempted sample chooses one
kind uniformly; there is no fallback if that kind is ineligible.
The random control shares teacher eligibility but picks any of the five boxes,
including potentially different evidence strengths. Thus it matches eligible
images/kinds and mask area without inheriting teacher region selection.

Teacher correctness/confidence is not shape ground truth. Background, crop resize,
fill artifacts, reference composition and prototype compression can bias targets.
Use the report and mean-vs-donor/reference ablations before interpreting regions
as useful object parts. No exact sketch-photo instance pairing is assumed.

## Preparation, storage and reproducibility

Run the main teacher pretraining/global cache first (automatically done by
src.train). On an evidence-cache miss, reload DFN and restore the exact saved
teacher prompt state. Encode balanced references and 11 views per selected photo
(clean + five masked + five crops). Save only float32 class scores and metadata,
then release DFN before student training. A cache hit skips this reload.

Disk payload estimate is N_photos * 11 * C_train * 4 bytes; at 72,945 photos and
104 classes about 318 MiB, plus metadata. Main global cache, model weights, wheels,
student checkpoints and logs are additional. Preparation needs 11 teacher views
per photo, so small disk size does NOT mean fast preparation. Teacher microbatch
defaults to 8. Student auxiliary attempts 25% of batch positions; ineligible or
uncached photos skip it and their coverage is logged. Full targets (0 photos per
class) are recommended for training; 10 per class is a small audit only.

Cache keys include SHA256 of the main teacher file, main metadata, content hashes
of selected images/references/donors, exact indices, view definition, packages
and microbatch. Existing incompatible caches fail and are never overwritten.
Writes are atomic; a .tmp is not reused. Two GiB free disk is reserved. Targets
are the same for response/static and teacher/random ablations, allowing reuse.
No generated views or large local-feature arrays are persisted.

Main currently evaluates on the designated unseen split and selects checkpoints
by its precision, including teacher pretraining. This behavior is preserved for
baseline parity; it is NOT an independent model-selection validation split.
Do not tune evidence thresholds/weights repeatedly on those final test scores.
For publication-quality selection, use held-out seen classes or a separate
validation protocol for ALL methods and rerun the corresponding baseline.

## Kaggle workflow

The two setup scripts are maintained locally outside this branch, as requested.
Use the NEW evidence online builder with Internet enabled, save its output, then
attach that bundle and Sketchy to the offline GPU notebook and run the NEW offline
restore script. Bundles from semantic-structural are pinned to a different source
commit and cannot be substituted. No audit data is needed for initial setup.

Set a reusable command in a Python cell (run after offline setup):

```python
import shlex, subprocess, sys
BASE = shlex.split('''
--root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy
--dataset sketchy_2 --epochs 5 --workers 8 --batch_size 64 --test_batch_size 32
--n_ctx_visual 3 --prompt_depth 12
--teacher_pretrain_epochs 1 --teacher_pretrain_batch_size 64
--teacher_n_ctx_visual 10 --teacher_prompt_depth 12 --teacher_prompt_std 0.02
--teacher_prompt_seed 42 --teacher_prompt_lr 3e-2
--teacher_prompt_gradient_checkpointing --teacher_momentum 0.9 --teacher_weight_decay 1e-3
--lambda_teacher_retrieval 1.5 --teacher_triplet_margin 0.2
--teacher_cache_path /kaggle/working/teacher_cache/sketchy2_teacher_1ep.pt
--lambda_domain 3 --lambda_modality 1 --kd_temperature 0.07
--photo_text_kd_temperature 0.15 --sketch_text_kd_temperature 0.02
--lr 1e-2 --momentum 0.9 --weight_decay 1e-3 --seed 42 --progress
''')
def run_evidence(extra):
    subprocess.run([sys.executable, '-m', 'src.train', *BASE, *shlex.split(extra)],
                   cwd='/kaggle/working/KD-SBIR', check=True)
```

First, pretrain teacher one epoch and audit 10 photos/class (no student training):

```python
run_evidence('''--evidence_prepare_only --evidence_photos_per_class 10
--evidence_cache_path /kaggle/working/evidence/audit_mean.pt
--evidence_report_dir /kaggle/working/evidence/audit_mean_report
--exp_name evidence_audit''')
```

Download report.html and inspect it. The report includes all eligibility flags
and raw drops/RMS in report.json. The gallery shows two images/class, including
failed candidates; strongest-drop illustrations are not automatically eligible.

Main B0 (reuses the same global teacher cache):

```python
run_evidence('--lambda_evidence 0 --exp_name evidence_B0_main')
```

B3 proposed method (first run builds full evidence targets, then trains student):

```python
run_evidence('''--lambda_evidence 1 --evidence_objective response
--evidence_selection teacher --evidence_reference sketch
--evidence_region_kind both --evidence_fill mean --evidence_area 0.25
--evidence_refs_per_class 8 --evidence_photos_per_class 0
--evidence_min_drop 0.01 --evidence_stable_rms 0.01
--evidence_batch_fraction 0.25 --evidence_teacher_batch_size 8
--evidence_cache_path /kaggle/working/evidence/full_mean_sketch.pt
--evidence_report_dir /kaggle/working/evidence/full_mean_report
--exp_name evidence_B3_response''')
```

For subsequent runs, omit evidence_report_dir or give a new directory.
Use the same B3 options and alter only the following:

| Run | objective | selection | reference | Cache |
|---|---|---|---|---|
| B1 | masked | random | sketch | reuse B3 |
| B2 | masked | teacher | sketch | reuse B3 |
| B3 | response | teacher | sketch | full_mean_sketch.pt |
| B4 | response | teacher | photo | NEW full_mean_photo.pt |
| Fill control | response | teacher | sketch, fill=donor | NEW full_donor_sketch.pt |

Always change exp_name. Important-only/stable-only reuse targets and use
`--evidence_region_kind important` or `stable`. Changing seed/reference/area/fill/
photos-per-class/teacher microbatch requires a new cache path. Test the selected
configuration across several seeds with corresponding matched B0 runs.

Audit-only success is not a training benchmark. Accept the method only if it
improves clean retrieval and exceeds the masked-score and random controls, not
just if its auxiliary loss decreases. Compare wall time including target prep.

## Tests

`python -m unittest discover -s tests -p test_evidence.py -v`

Includes CPU/CUDA deterministic AMP 12-layer student training, all prompt gradients,
unchanged baseline loss/gradients at weight zero, inference identity, equal-area
views, balanced references, teacher prompt restoration with different hidden/output
dimensions, target replay/cache reuse/content invalidation and report generation.
Tiny networks are used; these do not establish full DFN/Sketchy retrieval quality.
