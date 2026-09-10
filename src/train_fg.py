import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.dataset_fg import (
    FineGrainedIndex,
    FineGrainedFullGalleryBatchSampler,
    FineGrainedTrainDataset,
    FineGrainedValidDataset,
)
from src.model_fg import FineGrainedZS_SBIR, default_teacher_cache_path


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
    index = FineGrainedIndex(args.root, args.dataset, validate_counts=False)
    train_dataset = FineGrainedTrainDataset(args, index=index)
    val_sketch = FineGrainedValidDataset(args, index, modality="sketch")
    val_photo = FineGrainedValidDataset(args, index, modality="photo")

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 4 if args.workers > 0 else None,
        "worker_init_fn": seed_worker,
    }
    train_sampler = FineGrainedFullGalleryBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_dataset.collate_full_gallery,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_sketch_loader = DataLoader(
        val_sketch,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_kwargs,
    )
    val_photo_loader = DataLoader(
        val_photo,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2),
        **loader_kwargs,
    )
    print(
        "[FG Dataset] "
        f"seen_sketches={len(train_dataset):,}, "
        f"seen_photos={len(train_dataset.all_photo_paths):,}, "
        f"unseen_sketches={len(val_sketch):,}, "
        f"unseen_photos={len(val_photo):,}"
    )
    print(
        "[FG Training] each batch uses up to "
        f"{args.batch_size} sketches from one category + its 100-photo gallery"
    )
    return train_loader, val_sketch_loader, val_photo_loader


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Staged teacher semantic refinement for exact-instance "
            "fine-grained ZS-SBIR distillation."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Sketchy Basic root directly containing sketch/ and photo/.",
    )
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--dataset", choices=["sketchy_2"], default="sketchy_2")
    parser.add_argument("--backbone", default="ViT-B/32")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--n_ctx_visual", type=int, default=3)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument(
        "--student_n_ctx_text",
        "--n_ctx_text",
        "--text_prompt_tokens",
        dest="student_n_ctx_text",
        type=int,
        default=8,
        help=(
            "Number of image-conditioned soft text tokens used by the "
            "student. --n_ctx_text is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--teacher_n_ctx_text",
        "--teacher_text_prompt_tokens",
        dest="teacher_n_ctx_text",
        type=int,
        default=8,
        help="Number of image-conditioned soft text tokens used by the teacher.",
    )
    parser.add_argument("--text_prompt_gate_init", type=float, default=0.1)
    parser.add_argument("--text_prompt_seed", type=int, default=None)
    parser.add_argument("--teacher_text_prompt_seed", type=int, default=None)
    parser.add_argument("--text_prompt_encode_chunk_size", type=int, default=64)
    parser.add_argument(
        "--teacher_text_prompt_encode_chunk_size",
        type=int,
        default=32,
    )
    parser.add_argument("--text_prompt_lr", type=float, default=1e-3)
    parser.add_argument("--text_prompt_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--teacher_text_prompt_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--teacher_text_prompt_weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--teacher_text_pretrain_epochs",
        type=int,
        default=3,
        help=(
            "Frozen-visual epochs used to bootstrap stable teacher text "
            "semantic targets after visual-only pretraining."
        ),
    )
    parser.add_argument(
        "--lambda_teacher_text_anchor",
        type=float,
        default=0.05,
        help="Keep dynamic teacher text prompts near frozen class semantics.",
    )
    parser.add_argument(
        "--text_prompt_gradient_checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_text_prompt_gradient_checkpointing",
        action="store_false",
        dest="text_prompt_gradient_checkpointing",
    )
    parser.add_argument("--lambda_student_retrieval", type=float, default=1.0)
    parser.add_argument("--student_instance_temperature", type=float, default=0.07)
    parser.add_argument("--lambda_prompt_infonce", type=float, default=1.0)
    parser.add_argument("--prompt_infonce_temperature", type=float, default=0.07)
    parser.add_argument(
        "--lambda_teacher_prompt_infonce",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--teacher_prompt_infonce_temperature",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--teacher_semantic_refine_epochs",
        type=int,
        default=3,
        help=(
            "Visual-only refinement epochs against frozen teacher text "
            "semantic targets."
        ),
    )
    parser.add_argument(
        "--teacher_semantic_refine_lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--teacher_semantic_warmup_epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lambda_teacher_semantic_refine",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--lambda_teacher_visual_keep",
        type=float,
        default=0.1,
        help="Preserve the best visual-bootstrap teacher during refinement.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument(
        "--no_progress", action="store_false", dest="progress"
    )

    parser.add_argument("--teacher_n_ctx_visual", type=int, default=10)
    parser.add_argument("--teacher_prompt_depth", type=int, default=12)
    parser.add_argument("--teacher_prompt_std", type=float, default=0.02)
    parser.add_argument("--teacher_prompt_lr", type=float, default=3e-2)
    parser.add_argument("--teacher_prompt_seed", type=int, default=None)
    parser.add_argument(
        "--teacher_prompt_gradient_checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_teacher_prompt_gradient_checkpointing",
        action="store_false",
        dest="teacher_prompt_gradient_checkpointing",
    )
    parser.add_argument("--teacher_momentum", type=float, default=0.9)
    parser.add_argument("--teacher_weight_decay", type=float, default=1e-3)
    parser.add_argument("--teacher_pretrain_epochs", type=int, default=2)
    parser.add_argument("--teacher_pretrain_batch_size", type=int, default=64)
    parser.add_argument("--teacher_scheduler_patience", type=int, default=3)
    parser.add_argument("--teacher_scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--teacher_cache_path", default="")
    parser.add_argument("--teacher_cache_dir", default="")
    parser.add_argument("--rebuild_teacher_cache", action="store_true")
    parser.add_argument(
        "--teacher_only",
        action="store_true",
        help=(
            "Build/evaluate the staged teacher cache and stop before student "
            "training."
        ),
    )
    parser.add_argument("--lambda_teacher_retrieval", type=float, default=1.5)
    parser.add_argument(
        "--teacher_instance_temperature",
        type=float,
        default=0.07,
        help="Temperature for teacher exact-instance InfoNCE over 100 photos.",
    )
    parser.add_argument(
        "--teacher_triplet_margin",
        type=float,
        default=0.2,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--lambda_domain", type=float, default=3.0)
    parser.add_argument("--kd_temperature", type=float, default=0.07)
    parser.add_argument("--lambda_modality", type=float, default=1.0)
    parser.add_argument("--image_text_kd_temperature", type=float, default=0.1)
    parser.add_argument("--photo_text_kd_temperature", type=float, default=None)
    parser.add_argument("--sketch_text_kd_temperature", type=float, default=None)
    parser.add_argument(
        "--exp_name",
        default="fine_grained_teacher_semantic_refinement",
    )
    return parser


def validate_args(parser, args):
    if args.teacher_prompt_seed is None:
        args.teacher_prompt_seed = args.seed
    if args.text_prompt_seed is None:
        args.text_prompt_seed = args.seed + 40_000
    if args.teacher_text_prompt_seed is None:
        args.teacher_text_prompt_seed = args.seed + 50_000
    if args.photo_text_kd_temperature is None:
        args.photo_text_kd_temperature = args.image_text_kd_temperature
    if args.sketch_text_kd_temperature is None:
        args.sketch_text_kd_temperature = args.image_text_kd_temperature

    if args.n_ctx_visual < 1:
        parser.error("--n_ctx_visual must be at least 1.")
    if args.prompt_depth < 1:
        parser.error("--prompt_depth must be at least 1.")
    if not 1 <= args.student_n_ctx_text <= 32:
        parser.error("--student_n_ctx_text must be in [1, 32].")
    if not 1 <= args.teacher_n_ctx_text <= 32:
        parser.error("--teacher_n_ctx_text must be in [1, 32].")
    if args.text_prompt_encode_chunk_size < 1:
        parser.error("--text_prompt_encode_chunk_size must be positive.")
    if args.teacher_text_prompt_encode_chunk_size < 1:
        parser.error(
            "--teacher_text_prompt_encode_chunk_size must be positive."
        )
    if args.batch_size < 1 or args.test_batch_size < 1:
        parser.error("Batch sizes must be positive.")
    if args.teacher_pretrain_batch_size < 1:
        parser.error("--teacher_pretrain_batch_size must be positive.")
    if args.teacher_pretrain_epochs < 0:
        parser.error("--teacher_pretrain_epochs must be non-negative.")
    if args.teacher_text_pretrain_epochs < 0:
        parser.error("--teacher_text_pretrain_epochs must be non-negative.")
    if args.teacher_semantic_refine_epochs < 0:
        parser.error("--teacher_semantic_refine_epochs must be non-negative.")
    if args.teacher_semantic_warmup_epochs < 1:
        parser.error("--teacher_semantic_warmup_epochs must be at least 1.")
    if args.teacher_pretrain_epochs > 0 and args.teacher_n_ctx_visual < 1:
        parser.error("Teacher prompt pretraining requires visual prompts.")
    if args.teacher_prompt_depth == 0 or args.teacher_prompt_depth < -1:
        parser.error("--teacher_prompt_depth must be -1 or greater than 0.")
    positive_values = {
        "--lr": args.lr,
        "--teacher_prompt_lr": args.teacher_prompt_lr,
        "--teacher_prompt_std": args.teacher_prompt_std,
        "--teacher_scheduler_gamma": args.teacher_scheduler_gamma,
        "--scheduler_gamma": args.scheduler_gamma,
        "--teacher_instance_temperature": args.teacher_instance_temperature,
        "--text_prompt_gate_init": args.text_prompt_gate_init,
        "--text_prompt_lr": args.text_prompt_lr,
        "--teacher_text_prompt_lr": args.teacher_text_prompt_lr,
        "--teacher_semantic_refine_lr": args.teacher_semantic_refine_lr,
        "--student_instance_temperature": args.student_instance_temperature,
        "--prompt_infonce_temperature": args.prompt_infonce_temperature,
        "--teacher_prompt_infonce_temperature": (
            args.teacher_prompt_infonce_temperature
        ),
        "--kd_temperature": args.kd_temperature,
        "--image_text_kd_temperature": args.image_text_kd_temperature,
        "--photo_text_kd_temperature": args.photo_text_kd_temperature,
        "--sketch_text_kd_temperature": args.sketch_text_kd_temperature,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{name} must be greater than 0.")
    nonnegative_values = {
        "--momentum": args.momentum,
        "--weight_decay": args.weight_decay,
        "--teacher_momentum": args.teacher_momentum,
        "--teacher_weight_decay": args.teacher_weight_decay,
        "--text_prompt_weight_decay": args.text_prompt_weight_decay,
        "--teacher_text_prompt_weight_decay": (
            args.teacher_text_prompt_weight_decay
        ),
        "--lambda_teacher_retrieval": args.lambda_teacher_retrieval,
        "--lambda_student_retrieval": args.lambda_student_retrieval,
        "--lambda_prompt_infonce": args.lambda_prompt_infonce,
        "--lambda_teacher_prompt_infonce": (
            args.lambda_teacher_prompt_infonce
        ),
        "--lambda_teacher_text_anchor": args.lambda_teacher_text_anchor,
        "--lambda_teacher_semantic_refine": (
            args.lambda_teacher_semantic_refine
        ),
        "--lambda_teacher_visual_keep": args.lambda_teacher_visual_keep,
        "--lambda_domain": args.lambda_domain,
        "--lambda_modality": args.lambda_modality,
    }
    for name, value in nonnegative_values.items():
        if value < 0:
            parser.error(f"{name} must be non-negative.")
    if args.teacher_scheduler_patience < 1:
        parser.error("--teacher_scheduler_patience must be at least 1.")
    if args.scheduler_patience < 1:
        parser.error("--scheduler_patience must be at least 1.")
    if args.teacher_scheduler_gamma >= 1:
        parser.error("--teacher_scheduler_gamma must be less than 1.")
    if args.scheduler_gamma >= 1:
        parser.error("--scheduler_gamma must be less than 1.")
    if (
        args.lambda_domain == 0
        and args.lambda_modality == 0
        and args.lambda_student_retrieval == 0
        and args.lambda_prompt_infonce == 0
    ):
        parser.error("At least one student loss must be active.")
    if (
        args.teacher_pretrain_epochs > 0
        and args.lambda_teacher_retrieval == 0
    ):
        parser.error("Teacher visual bootstrap requires retrieval loss.")
    if (
        args.teacher_pretrain_epochs > 0
        and args.teacher_semantic_refine_epochs > 0
        and args.teacher_text_pretrain_epochs == 0
    ):
        parser.error(
            "Teacher semantic refinement requires text bootstrap epochs."
        )
    if (
        args.teacher_text_pretrain_epochs > 0
        and args.lambda_teacher_prompt_infonce == 0
        and args.lambda_teacher_text_anchor == 0
    ):
        parser.error("At least one teacher text-bootstrap loss must be active.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    train_loader, val_sketch_loader, val_photo_loader = get_loaders(args)

    if not args.teacher_cache_path and args.teacher_pretrain_epochs > 0:
        args.teacher_cache_path = default_teacher_cache_path(
            args, train_loader.dataset
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
        parser.error("--rebuild_teacher_cache requires a cache path.")

    logger = TensorBoardLogger("tb_logs", name=args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="fg_selection",
        dirpath=f"saved_models/{args.exp_name}",
        filename="{epoch:02d}-{acc1:.4f}-{acc5:.4f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )
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
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=20)],
    )
    model = FineGrainedZS_SBIR(
        args=args, classnames=train_loader.dataset.all_categories
    )
    if os.path.isfile(args.ckpt_path):
        print(f"Resuming training from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.cache_teacher_features(
        train_loader.dataset,
        val_sketch_loader,
        val_photo_loader,
        batch_size=args.teacher_pretrain_batch_size,
        workers=args.workers,
        show_progress=args.progress,
    )
    if args.teacher_only:
        print(
            "[Teacher Only] staged teacher cache is ready; "
            "student training skipped."
        )
        return
    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])


if __name__ == "__main__":
    main()
