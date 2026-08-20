```bash
!python -m src.train \
    --root /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy \
    --dataset sketchy_2 \
    --epochs 20 \
    --workers 8 \
    --batch_size 64 \
    --test_batch_size 1024 \
    --n_ctx_text 3 \
    --n_ctx_visual 3 \
    --prompt_depth 12 \
    --lambda_kd 7.0 \
    --lambda_photo_text_kd 1.0 \
    --lambda_sketch_text_kd 1.0 \
    --photo_text_kd_temperature 0.07 \
    --sketch_text_kd_temperature 0.07 \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight_decay 1e-3 \
    --teacher_pretrain_epochs 3 \
    --teacher_pretrain_batch_size 16 \
    --teacher_n_ctx_visual 3 \
    --teacher_prompt_depth 12 \
    --teacher_prompt_std 0.02 \
    --teacher_prompt_lr 2e-5 \
    --teacher_prompt_seed 42 \
    --teacher_prompt_gradient_checkpointing \
    --teacher_momentum 0.9 \
    --teacher_weight_decay 1e-3 \
    --teacher_scheduler_step_size 5 \
    --teacher_scheduler_gamma 0.1 \
    --seed 42 \
    --exp_name sketchy2_independent_prompts
```

Photo and sketch have separate deep prompts in both the text and visual
transformers. Every prompted layer has its own independent parameters; there
is no text-to-visual projection or token sharing. Text prompts with one to
three tokens are initialized from CLIP's token embeddings for `a photo of`
and `a sketch of`. With four or more tokens, the complete text prompt is
initialized from `Normal(0, 0.02)`. Visual prompts are always initialized
independently from `Normal(0, 0.02)`.

`--prompt_depth` controls how many transformer layers receive prompts.
Both `--n_ctx_text` and `--n_ctx_visual` accept zero for branch-specific
ablations.

Training uses fixed resize/normalize transforms without augmentation. Student
classification and teacher semantic classification are removed. The student
learns only from sketch-photo relational KD, photo-text KD, and sketch-text KD.
The teacher uses independent photo/sketch deep visual prompts and learns only
from retrieval triplet loss. Its pretrained visual and text weights remain
fully frozen; this branch contains neither teacher adapters nor teacher LoRA.

Photo-text and sketch-text KD temperatures can be set independently. The
existing `--image_text_kd_temperature` remains available as the fallback for
either modality-specific temperature that is not supplied.

When `--teacher_pretrain_epochs` is positive, modality-specific prompts are
inserted into the frozen DFN5B visual transformer. The prompt at layer zero is
appended after positional embeddings and before `ln_pre`. At every subsequent
prompted layer, the previous prompt tokens are replaced by that layer's own
parameters. Prompts are trained by forwarding seen photo/sketch images in every
teacher epoch. They are then frozen, and tuned teacher features are materialized
once for student distillation. The image features, original teacher text
targets, prompt state, and configuration metadata are saved automatically under
`/kaggle/working/teacher_cache`. The filename is derived from the dataset and
complete teacher configuration. Reusing the same teacher settings in a later
cell loads the file and skips both DFN5B loading and prompt pretraining. Student
losses, prompts, and optimizer settings may change without invalidating the
teacher cache. Pass
`--rebuild_teacher_cache` to overwrite the matching cache intentionally, or
`--teacher_cache_path` to use an explicit file.

The persistent cache is validated against the dataset paths and all teacher
pretraining settings. Use a different cache path when changing those settings.
Files under `/kaggle/working` persist across cells in the current notebook; to
reuse them in another Kaggle session, save the notebook output and attach it as
an input.

After every teacher prompt-pretraining epoch, the current teacher is evaluated
on the unseen sketch queries and photo gallery with the same retrieval metrics
used for the student. These metrics are printed for monitoring only and do not
select or restore a teacher checkpoint. Selecting teacher hyperparameters or an
epoch from these unseen metrics would make the experiment transductive rather
than strictly inductive.

The student CLIP backbone, including every LayerNorm, is fully frozen. After
teacher pretraining, only active student prompt parameters are optimized.

Teacher prompts use `Normal(0, teacher_prompt_std)` initialization with a local
seed that does not consume the global student RNG. `--teacher_prompt_depth -1`
prompts every teacher visual Transformer block; a positive value prompts the
first requested number of blocks. Photo and sketch never share prompt vectors.
