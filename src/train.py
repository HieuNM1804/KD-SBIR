import os

# Required by deterministic CUDA matrix multiplication. It must be set before
# importing torch and before CUDA is initialized.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import TrainDataset, ValidDataset, WorkerInvariantSampler
from src.data_config import UNSEEN_CLASSES
from src.model import ZS_SBIR


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
    val_sketch = ValidDataset(args, mode='sketch')
    val_photo = ValidDataset(args)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
    )

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
        "--n_ctx",
        type=int,
        default=4,
        help="Number of learnable visual prompt tokens for each modality.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, PyTorch, and DataLoader workers.",
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1.0,
        help="Weight for CE(photo, text) + CE(sketch, text).",
    )

    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--teacher_pretrain_epochs",
        type=int,
        default=3,
        help=(
            "Epochs used to train teacher adapters before student distillation; "
            "ignored when --teacher_adapter_ckpt is provided."
        ),
    )
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
    parser.add_argument("--teacher_adapter_bottleneck", type=int, default=64)
    parser.add_argument("--teacher_adapter_lr", type=float, default=2e-5)
    parser.add_argument(
        "--teacher_adapter_ckpt",
        type=str,
        default="",
        help=(
            "Standalone teacher-adapter checkpoint. When provided, adapter "
            "pretraining is skipped and the loaded adapter stays frozen."
        ),
    )
    parser.add_argument(
        "--lambda_teacher_retrieval",
        type=float,
        default=1.5,
        help="Weight for the teacher-adapter retrieval loss.",
    )
    parser.add_argument("--lambda_teacher_semantic", type=float, default=1.0)
    parser.add_argument("--teacher_temperature", type=float, default=0.07)
    parser.add_argument("--teacher_triplet_margin", type=float, default=0.2)
    parser.add_argument(
        "--lambda_kd",
        type=float,
        default=3.0,
        help="Weight for sketch-photo relational distillation.",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=0.07,
        help="Temperature for the sketch-photo similarity distribution.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="no_student_triplet_worker_invariant",
    )

    args = parser.parse_args()
    if args.teacher_pretrain_epochs < 0:
        parser.error("--teacher_pretrain_epochs must be non-negative.")

    adapter_ckpt = None
    if args.teacher_adapter_ckpt:
        if not os.path.isfile(args.teacher_adapter_ckpt):
            raise FileNotFoundError(
                f"Teacher adapter checkpoint not found: "
                f"{args.teacher_adapter_ckpt}"
            )
        adapter_ckpt = torch.load(
            args.teacher_adapter_ckpt,
            map_location="cpu",
        )
        args.teacher_adapter_bottleneck = int(
            adapter_ckpt.get(
                "bottleneck_dim",
                args.teacher_adapter_bottleneck,
            )
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath=f"saved_models/{args.exp_name}",
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    train_loader, val_sketch_loader, val_photo_loader = get_loaders(args)
    model = ZS_SBIR(args=args, classnames=train_loader.dataset.all_categories)
    if os.path.isfile(args.ckpt_path):
        print(f"Loading model weights from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    adapter_loaded = adapter_ckpt is not None
    if adapter_loaded:
        model.model.teacher_adapters.load_state_dict(
            adapter_ckpt["adapter_state_dict"],
            strict=True,
        )
        print(
            f"[Teacher Adapter] loaded {args.teacher_adapter_ckpt}; "
            "pretraining skipped"
        )

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        benchmark=False,
        deterministic=True,
        enable_progress_bar=args.progress,
    )

    if args.teacher_pretrain_epochs > 0 and not adapter_loaded:
        model.set_training_phase("adapter")
        adapter_logger = TensorBoardLogger(
            "tb_logs",
            name=args.exp_name,
            version="adapter_pretrain",
        )
        adapter_trainer = Trainer(
            **trainer_kwargs,
            max_epochs=args.teacher_pretrain_epochs,
            logger=adapter_logger,
            limit_val_batches=0,
            enable_checkpointing=False,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )
        adapter_trainer.fit(model, train_loader)

        adapter_dir = os.path.join("saved_models", args.exp_name)
        os.makedirs(adapter_dir, exist_ok=True)
        adapter_path = os.path.join(adapter_dir, "teacher_adapter_last.pt")
        adapter_state = {
            name: tensor.detach().cpu()
            for name, tensor in model.model.teacher_adapters.state_dict().items()
        }
        torch.save(
            {
                "adapter_state_dict": adapter_state,
                "epochs": args.teacher_pretrain_epochs,
                "bottleneck_dim": args.teacher_adapter_bottleneck,
            },
            adapter_path,
        )
        print(f"[Teacher Adapter] saved to {adapter_path}")

    # Reset sample order so student results do not depend on pretraining length.
    train_loader.sampler.epoch = 0
    model.set_training_phase("student")
    student_logger = TensorBoardLogger(
        "tb_logs",
        name=args.exp_name,
        version="student_distill",
    )
    student_trainer = Trainer(
        **trainer_kwargs,
        max_epochs=args.epochs,
        logger=student_logger,
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=20),
        ],
    )
    student_trainer.fit(
        model,
        train_loader,
        [val_sketch_loader, val_photo_loader],
    )
