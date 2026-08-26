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
        description="Exact-instance fine-grained ZS-SBIR distillation."
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
        "--adapter_bottleneck",
        type=int,
        default=64,
        help="Student adapter compression width; 0 disables student adapters.",
    )
    parser.add_argument("--adapter_std", type=float, default=0.02)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--adapter_seed", type=int, default=None)
    parser.add_argument("--adapter_lr", type=float, default=None)
    parser.add_argument("--adapter_weight_decay", type=float, default=None)
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
        "--teacher_adapter_bottleneck",
        type=int,
        default=64,
        help="Teacher adapter compression width; 0 disables teacher adapters.",
    )
    parser.add_argument("--teacher_adapter_std", type=float, default=0.02)
    parser.add_argument("--teacher_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--teacher_adapter_scale", type=float, default=1.0)
    parser.add_argument("--teacher_adapter_seed", type=int, default=None)
    parser.add_argument("--teacher_adapter_lr", type=float, default=None)
    parser.add_argument(
        "--teacher_adapter_weight_decay", type=float, default=None
    )
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
    parser.add_argument("--exp_name", default="fine_grained_distillation")
    return parser


def validate_args(parser, args):
    if args.teacher_prompt_seed is None:
        args.teacher_prompt_seed = args.seed
    if args.adapter_seed is None:
        args.adapter_seed = args.seed + 30_000
    if args.teacher_adapter_seed is None:
        args.teacher_adapter_seed = args.teacher_prompt_seed + 10_000
    if args.adapter_lr is None:
        args.adapter_lr = args.lr
    if args.adapter_weight_decay is None:
        args.adapter_weight_decay = args.weight_decay
    if args.teacher_adapter_lr is None:
        args.teacher_adapter_lr = args.teacher_prompt_lr
    if args.teacher_adapter_weight_decay is None:
        args.teacher_adapter_weight_decay = args.teacher_weight_decay
    if args.photo_text_kd_temperature is None:
        args.photo_text_kd_temperature = args.image_text_kd_temperature
    if args.sketch_text_kd_temperature is None:
        args.sketch_text_kd_temperature = args.image_text_kd_temperature

    if args.n_ctx_visual < 1:
        parser.error("--n_ctx_visual must be at least 1.")
    if args.prompt_depth < 1:
        parser.error("--prompt_depth must be at least 1.")
    if args.adapter_bottleneck < 0:
        parser.error("--adapter_bottleneck must be non-negative.")
    if args.batch_size < 1 or args.test_batch_size < 1:
        parser.error("Batch sizes must be positive.")
    if args.teacher_pretrain_batch_size < 1:
        parser.error("--teacher_pretrain_batch_size must be positive.")
    if args.teacher_pretrain_epochs < 0:
        parser.error("--teacher_pretrain_epochs must be non-negative.")
    if args.teacher_pretrain_epochs > 0 and args.teacher_n_ctx_visual < 1:
        parser.error("Teacher prompt pretraining requires visual prompts.")
    if args.teacher_prompt_depth == 0 or args.teacher_prompt_depth < -1:
        parser.error("--teacher_prompt_depth must be -1 or greater than 0.")
    if args.teacher_adapter_bottleneck < 0:
        parser.error("--teacher_adapter_bottleneck must be non-negative.")
    positive_values = {
        "--lr": args.lr,
        "--adapter_lr": args.adapter_lr,
        "--teacher_prompt_lr": args.teacher_prompt_lr,
        "--teacher_adapter_lr": args.teacher_adapter_lr,
        "--teacher_prompt_std": args.teacher_prompt_std,
        "--adapter_std": args.adapter_std,
        "--adapter_scale": args.adapter_scale,
        "--teacher_adapter_std": args.teacher_adapter_std,
        "--teacher_adapter_scale": args.teacher_adapter_scale,
        "--teacher_scheduler_gamma": args.teacher_scheduler_gamma,
        "--scheduler_gamma": args.scheduler_gamma,
        "--teacher_instance_temperature": args.teacher_instance_temperature,
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
        "--adapter_dropout": args.adapter_dropout,
        "--adapter_weight_decay": args.adapter_weight_decay,
        "--teacher_momentum": args.teacher_momentum,
        "--teacher_weight_decay": args.teacher_weight_decay,
        "--teacher_adapter_dropout": args.teacher_adapter_dropout,
        "--teacher_adapter_weight_decay": (
            args.teacher_adapter_weight_decay
        ),
        "--lambda_teacher_retrieval": args.lambda_teacher_retrieval,
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
    if args.adapter_dropout >= 1:
        parser.error("--adapter_dropout must be less than 1.")
    if args.teacher_adapter_dropout >= 1:
        parser.error("--teacher_adapter_dropout must be less than 1.")
    if args.lambda_domain == 0 and args.lambda_modality == 0:
        parser.error("At least one student distillation loss must be active.")


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
    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])


if __name__ == "__main__":
    main()
