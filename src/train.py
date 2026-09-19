import os

# Required by deterministic CUDA matrix multiplication. It must be set before
# importing torch and before CUDA is initialized.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.data_config import UNSEEN_CLASSES
from src.dataset import TrainDataset, ValidDataset, WorkerInvariantSampler
from src.model import ZS_SBIR, default_teacher_cache_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(args):
    seed_everything(args.seed)

    train_dataset = TrainDataset(args)
    val_sketch = ValidDataset(args, mode="sketch")
    val_photo = ValidDataset(args)

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 4 if args.workers > 0 else None,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=WorkerInvariantSampler(train_dataset, args.seed),
        drop_last=True,  # RKD requires complete batches with at least two samples.
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_sketch_loader = DataLoader(
        dataset=val_sketch,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_kwargs,
    )
    val_photo_loader = DataLoader(
        dataset=val_photo,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2),
        **loader_kwargs,
    )

    return train_loader, val_sketch_loader, val_photo_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root containing sketch/ and photo/.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Student checkpoint to resume.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sketchy_1",
        choices=sorted(UNSEEN_CLASSES),
        help="Zero-shot split.",
    )
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument(
        "--n_ctx_visual",
        type=int,
        default=4,
        help=(
            "Number of independent random deep visual-prompt tokens per "
            "modality; 0 disables visual prompts."
        ),
    )
    parser.add_argument(
        "--prompt_depth",
        type=int,
        default=12,
        help=("Number of visual transformer layers receiving independent prompts."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, PyTorch, and DataLoader workers.",
    )
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum for student optimization.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="SGD weight decay for student optimization.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader workers; sample RNG is worker-count invariant.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show the tqdm training progress bar.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_false",
        dest="progress",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable model checkpoint creation for sweeps and diagnostics.",
    )
    parser.add_argument(
        "--teacher_n_ctx_visual",
        type=int,
        default=3,
        help="Number of independent visual prompt tokens per teacher modality.",
    )
    parser.add_argument(
        "--teacher_prompt_depth",
        type=int,
        default=12,
        help="Number of initial teacher visual blocks receiving prompts; -1 uses all.",
    )
    parser.add_argument(
        "--teacher_prompt_std",
        type=float,
        default=0.02,
        help="Standard deviation for teacher visual prompt initialization.",
    )
    parser.add_argument(
        "--teacher_prompt_lr",
        type=float,
        default=2e-5,
        help="SGD learning rate for teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_prompt_seed",
        type=int,
        default=None,
        help="Teacher-prompt initialization seed; defaults to --seed.",
    )
    parser.add_argument(
        "--teacher_prompt_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Recompute teacher forwards during backward to reduce memory.",
    )
    parser.add_argument(
        "--no_teacher_prompt_gradient_checkpointing",
        action="store_false",
        dest="teacher_prompt_gradient_checkpointing",
        help="Disable teacher prompt gradient checkpointing.",
    )
    parser.add_argument(
        "--teacher_momentum",
        type=float,
        default=0.9,
        help="SGD momentum for teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_weight_decay",
        "--weight_decay_teacher",
        dest="teacher_weight_decay",
        type=float,
        default=1e-3,
        help="SGD weight decay for teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_pretrain_epochs",
        type=int,
        default=0,
        help=(
            "Pretrain teacher visual prompts for this many epochs, freeze them, "
            "materialize tuned features before student training. Set 0 to "
            "use the original frozen DFN5B teacher."
        ),
    )
    parser.add_argument(
        "--teacher_pretrain_batch_size",
        type=int,
        default=64,
        help="Image batch size used during teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_scheduler_step_size",
        type=int,
        default=5,
        help="StepLR step size for teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_scheduler_gamma",
        type=float,
        default=0.1,
        help="StepLR decay factor for teacher prompt pretraining.",
    )
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="",
        help=(
            "Optional .pt file for persistent prompt-tuned teacher features and "
            "text targets. Existing compatible files skip DFN5B entirely."
        ),
    )
    parser.add_argument(
        "--teacher_cache_dir",
        type=str,
        default="",
        help=(
            "Directory for automatically named teacher caches. Defaults to "
            "/kaggle/working/teacher_cache on Kaggle and teacher_cache elsewhere."
        ),
    )
    parser.add_argument(
        "--rebuild_teacher_cache",
        action="store_true",
        help="Ignore and overwrite an existing persistent teacher cache.",
    )
    parser.add_argument(
        "--teacher_cache_only",
        action="store_true",
        help=(
            "Prepare or validate the persistent teacher cache, then exit before "
            "student training. Intended for teacher-only diagnostics."
        ),
    )
    parser.add_argument(
        "--lambda_teacher_retrieval",
        type=float,
        default=1.5,
        help="Weight for the teacher prompt retrieval loss.",
    )
    parser.add_argument("--teacher_triplet_margin", type=float, default=0.2)
    parser.add_argument(
        "--lambda_domain",
        type=float,
        default=3.0,
        help="Weight for sketch-photo domain distillation.",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=0.07,
        help="Temperature for the sketch-photo similarity distribution.",
    )
    parser.add_argument(
        "--lambda_modality",
        type=float,
        default=0.0,
        help=(
            "Shared weight for the sum of photo-text and sketch-text "
            "modality distillation losses."
        ),
    )
    parser.add_argument(
        "--image_text_kd_temperature",
        type=float,
        default=0.1,
        help="Shared fallback temperature for photo-text and sketch-text KD.",
    )
    parser.add_argument(
        "--photo_text_kd_temperature",
        type=float,
        default=None,
        help="Photo-text KD temperature; defaults to the shared temperature.",
    )
    parser.add_argument(
        "--sketch_text_kd_temperature",
        type=float,
        default=None,
        help="Sketch-text KD temperature; defaults to the shared temperature.",
    )
    parser.add_argument(
        "--lambda_core",
        type=float,
        default=0.0,
        help=(
            "Weight for adaptation-induced cross-modal margin correction "
            "distillation. Zero preserves the main baseline exactly."
        ),
    )
    parser.add_argument(
        "--core_teacher_temperature",
        type=float,
        default=0.05,
        help="Temperature for teacher margin-correction probabilities.",
    )
    parser.add_argument(
        "--core_student_temperature",
        type=float,
        default=0.05,
        help="Temperature for student margin-correction probabilities.",
    )
    parser.add_argument(
        "--core_hard_negative_topk",
        type=int,
        default=8,
        help=(
            "Search this many base-teacher hard negatives per query before "
            "choosing the negative most suppressed by adaptation."
        ),
    )
    parser.add_argument(
        "--core_min_teacher_correction",
        type=float,
        default=0.0,
        help="Minimum positive teacher margin correction retained for CoRe-KD.",
    )
    parser.add_argument(
        "--core_max_weight",
        type=float,
        default=0.25,
        help="Clip teacher correction weights before batch normalization.",
    )
    parser.add_argument(
        "--core_control",
        choices=("verified", "shuffled", "reversed"),
        default="verified",
        help=(
            "Correction target: verified method, anchor-identity shuffle, or "
            "sign reversal control."
        ),
    )
    parser.add_argument(
        "--core_direction",
        choices=("bidirectional", "sketch_to_photo", "photo_to_sketch"),
        default="bidirectional",
        help="Cross-modal correction direction used for training.",
    )
    parser.add_argument(
        "--lambda_gap_core",
        type=float,
        default=0.0,
        help=(
            "Weight for common-to-full modality-gap margin correction. "
            "Zero preserves the main baseline forward path."
        ),
    )
    parser.add_argument(
        "--gap_core_huber_beta",
        type=float,
        default=0.05,
        help="Smooth-L1 transition used to match correction magnitudes.",
    )
    parser.add_argument(
        "--gap_core_min_correction",
        type=float,
        default=0.0,
        help="Minimum verified teacher margin correction retained.",
    )
    parser.add_argument(
        "--gap_core_max_weight",
        type=float,
        default=0.25,
        help="Maximum teacher correction used for query weighting.",
    )
    parser.add_argument(
        "--gap_core_control",
        choices=("verified", "shuffled", "reversed"),
        default="verified",
        help="Verified, identity-shuffled, or sign-reversed gap target.",
    )
    parser.add_argument(
        "--gap_core_direction",
        choices=("bidirectional", "sketch_to_photo", "photo_to_sketch"),
        default="bidirectional",
        help="Cross-modal direction used by Gap-CoRe.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="teacher_visual_student_visual_only",
    )

    args = parser.parse_args()
    if args.teacher_prompt_seed is None:
        args.teacher_prompt_seed = args.seed
    if args.photo_text_kd_temperature is None:
        args.photo_text_kd_temperature = args.image_text_kd_temperature
    if args.sketch_text_kd_temperature is None:
        args.sketch_text_kd_temperature = args.image_text_kd_temperature
    if args.n_ctx_visual < 0:
        parser.error("--n_ctx_visual must be greater than or equal to 0.")
    if args.prompt_depth < 1:
        parser.error("--prompt_depth must be greater than or equal to 1.")
    if args.momentum < 0:
        parser.error("--momentum must be non-negative.")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be non-negative.")
    if args.teacher_n_ctx_visual < 1 and args.teacher_pretrain_epochs > 0:
        parser.error(
            "--teacher_n_ctx_visual must be at least 1 when teacher prompts train."
        )
    if args.teacher_prompt_depth == 0 or args.teacher_prompt_depth < -1:
        parser.error("--teacher_prompt_depth must be -1 or greater than 0.")
    if args.teacher_prompt_std <= 0:
        parser.error("--teacher_prompt_std must be greater than 0.")
    if args.teacher_prompt_lr <= 0:
        parser.error("--teacher_prompt_lr must be greater than 0.")
    if args.teacher_momentum < 0:
        parser.error("--teacher_momentum must be non-negative.")
    if args.teacher_weight_decay < 0:
        parser.error("--teacher_weight_decay must be non-negative.")
    if args.teacher_pretrain_epochs < 0:
        parser.error("--teacher_pretrain_epochs must be non-negative.")
    if args.teacher_pretrain_batch_size < 2:
        parser.error("--teacher_pretrain_batch_size must be at least 2.")
    if args.teacher_scheduler_step_size < 1:
        parser.error("--teacher_scheduler_step_size must be at least 1.")
    if args.teacher_scheduler_gamma <= 0:
        parser.error("--teacher_scheduler_gamma must be greater than 0.")
    if args.lambda_domain < 0:
        parser.error("--lambda_domain must be non-negative.")
    if args.lambda_modality < 0:
        parser.error("--lambda_modality must be non-negative.")
    if args.lambda_core < 0:
        parser.error("--lambda_core must be non-negative.")
    if args.lambda_core > 0 and args.teacher_pretrain_epochs <= 0:
        parser.error(
            "--lambda_core requires --teacher_pretrain_epochs greater than 0 "
            "so T0 and adapted T1 are distinct."
        )
    if args.core_teacher_temperature <= 0:
        parser.error("--core_teacher_temperature must be greater than 0.")
    if args.core_student_temperature <= 0:
        parser.error("--core_student_temperature must be greater than 0.")
    if args.core_hard_negative_topk < 1:
        parser.error("--core_hard_negative_topk must be at least 1.")
    if args.core_min_teacher_correction < 0:
        parser.error("--core_min_teacher_correction must be non-negative.")
    if args.core_max_weight <= 0:
        parser.error("--core_max_weight must be greater than 0.")
    if args.lambda_gap_core < 0:
        parser.error("--lambda_gap_core must be non-negative.")
    if args.lambda_gap_core > 0 and args.teacher_pretrain_epochs <= 0:
        parser.error(
            "--lambda_gap_core requires --teacher_pretrain_epochs greater "
            "than 0 so full and common teacher states are distinct."
        )
    if args.gap_core_huber_beta <= 0:
        parser.error("--gap_core_huber_beta must be greater than 0.")
    if args.gap_core_min_correction < 0:
        parser.error("--gap_core_min_correction must be non-negative.")
    if args.gap_core_max_weight <= 0:
        parser.error("--gap_core_max_weight must be greater than 0.")
    if args.image_text_kd_temperature <= 0:
        parser.error("--image_text_kd_temperature must be greater than 0.")
    if args.photo_text_kd_temperature <= 0:
        parser.error("--photo_text_kd_temperature must be greater than 0.")
    if args.sketch_text_kd_temperature <= 0:
        parser.error("--sketch_text_kd_temperature must be greater than 0.")
    logger = TensorBoardLogger("tb_logs", name=args.exp_name)

    train_loader, val_sketch_loader, val_photo_loader = get_loaders(args)
    if not args.teacher_cache_path and args.teacher_pretrain_epochs > 0:
        args.teacher_cache_path = default_teacher_cache_path(
            args,
            train_loader.dataset,
        )
        print(f"[Teacher Cache] automatic path: {args.teacher_cache_path}")

    cache_exists = bool(args.teacher_cache_path) and os.path.isfile(
        args.teacher_cache_path
    )
    if (
        args.teacher_cache_path
        and (args.rebuild_teacher_cache or not cache_exists)
        and args.teacher_pretrain_epochs == 0
    ):
        parser.error(
            "Creating a persistent teacher cache requires "
            "--teacher_pretrain_epochs greater than 0."
        )
    if args.rebuild_teacher_cache and not args.teacher_cache_path:
        parser.error(
            "--rebuild_teacher_cache requires --teacher_cache_path or "
            "--teacher_pretrain_epochs greater than 0."
        )
    callbacks = []
    if not args.no_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                monitor="precision",
                dirpath=f"saved_models/{args.exp_name}",
                filename="{epoch:02d}-{precision:.4f}",
                save_top_k=1,
                mode="max",
                save_last=True,
            )
        )
    if args.progress:
        callbacks.append(TQDMProgressBar(refresh_rate=20))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=args.epochs,
        benchmark=False,
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        enable_checkpointing=not args.no_checkpoints,
        callbacks=callbacks,
    )

    model = ZS_SBIR(args=args, classnames=train_loader.dataset.all_categories)
    if os.path.isfile(args.ckpt_path):
        print(f"Resuming training from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    model.cache_teacher_features(
        train_loader.dataset,
        val_sketch_loader,
        val_photo_loader,
        batch_size=args.teacher_pretrain_batch_size,
        workers=args.workers,
        show_progress=args.progress,
    )

    if args.teacher_cache_only:
        print("[Teacher Cache] cache-only stage complete; student training skipped.")
        raise SystemExit(0)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
