import os

# Required by deterministic CUDA matrix multiplications. This must be set
# before importing torch and before the CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np
import random
import argparse
import hashlib
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.dataset import AdapterPCADataset, TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories
from src.data_config import UNSEEN_CLASSES


def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    if deterministic and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_datasets(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    
    train_dataset = TrainDataset(args)
    val_sketch = ValidDataset(args, mode='sketch')
    val_photo = ValidDataset(args)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,           # Transfer CPU→GPU nhanh hơn (non-blocking)
        persistent_workers=args.workers > 0,  # Giữ worker sống giữa các epoch
        prefetch_factor=4 if args.workers > 0 else None,  # Pre-load 4 batch trước
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Tránh batch lẻ ở cuối gây vấn đề với RKD (cần B>=2)
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


def get_adapter_pca_loader(args, modality, seed_offset):
    dataset = AdapterPCADataset(args, modality=modality)
    print(
        f"[Adapter PCA] calibration modality={modality}, "
        f"images={len(dataset):,}, workers={args.workers}, "
        f"batch_size={args.adapter_pca_batch_size}"
    )
    return DataLoader(
        dataset=dataset,
        batch_size=args.adapter_pca_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + seed_offset),
    )


def _canonicalize_pca_signs(basis):
    """Resolve the arbitrary eigenvector signs for reproducible initialization."""
    max_indices = basis.abs().argmax(dim=1)
    row_indices = torch.arange(basis.shape[0])
    signs = torch.sign(basis[row_indices, max_indices])
    signs[signs == 0] = 1
    return basis * signs.unsqueeze(1)


@torch.inference_mode()
def _modality_pca_basis(model, adapter, loader, modality):
    teacher = model.model_distill
    teacher_device = next(teacher.parameters()).device
    feature_dim = int(teacher.output_dim)
    rank = adapter.down.out_features

    weighted_sum = torch.zeros(feature_dim, device=teacher_device, dtype=torch.float32)
    weighted_second_moment = torch.zeros(
        feature_dim,
        feature_dim,
        device=teacher_device,
        dtype=torch.float32,
    )
    total_weight = torch.zeros((), device=teacher_device, dtype=torch.float32)
    sample_count = 0

    teacher.eval()
    for images, sample_weights in loader:
        images = images.to(teacher_device, non_blocking=True)
        teacher_features = teacher.encode_image(model.teacher_image_input(images))
        teacher_features = F.normalize(teacher_features.float(), dim=-1)
        down_inputs = adapter.norm(teacher_features)

        sample_weights = sample_weights.to(
            teacher_device,
            dtype=torch.float32,
            non_blocking=True,
        ).reshape(-1, 1)
        weighted_inputs = down_inputs * sample_weights
        weighted_sum += weighted_inputs.sum(dim=0)
        weighted_second_moment += down_inputs.t() @ weighted_inputs
        total_weight += sample_weights.sum()
        sample_count += down_inputs.shape[0]

    if sample_count < rank or total_weight.item() <= 0:
        raise RuntimeError(
            f"Not enough {modality} samples for rank-{rank} PCA: {sample_count}."
        )

    mean = weighted_sum / total_weight
    covariance = weighted_second_moment / total_weight - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.t())

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.cpu().double())
    basis = eigenvectors[:, -rank:].t().float().contiguous()
    basis = _canonicalize_pca_signs(basis)
    explained = eigenvalues[-rank:].clamp_min(0).sum()
    total = eigenvalues.clamp_min(0).sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    explained_ratio = float((explained / total).item())
    digest = hashlib.sha256(basis.numpy().tobytes()).hexdigest()[:16]

    print(
        f"[Adapter PCA] modality={modality}, samples={sample_count:,}, "
        f"rank={rank}, explained_variance={explained_ratio:.4f}, "
        f"basis_sha256={digest}"
    )
    return basis


@torch.no_grad()
def initialize_teacher_adapters_from_pca(lightning_model, args):
    model = lightning_model.model
    adapters = model.teacher_adapters
    if model.model_distill is None or adapters is None:
        raise RuntimeError("Adapter PCA initialization requires DFN5B and teacher adapters.")

    for modality, adapter, seed_offset in (
        ("sketch", adapters.sketch, 100),
        ("photo", adapters.photo, 101),
    ):
        loader = get_adapter_pca_loader(args, modality, seed_offset)
        basis = _modality_pca_basis(model, adapter, loader, modality)
        adapter.norm.weight.fill_(1)
        adapter.norm.bias.zero_()
        adapter.down.weight.copy_(
            basis.to(adapter.down.weight.device, dtype=adapter.down.weight.dtype)
        )
        adapter.down.bias.zero_()
        adapter.up.weight.zero_()
        adapter.up.bias.zero_()

    print("[Adapter PCA] initialized modality-specific down projections; up projections remain zero.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="dataset root containing sketch/ and photo/")
    parser.add_argument("--ckpt_path", type=str, default="", help="student checkpoint to resume")
    parser.add_argument("--dataset", type=str, default="sketchy_1",
                        choices=sorted(UNSEEN_CLASSES), help="zero-shot split")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho Python/NumPy/PyTorch/DataLoader workers.")
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Dùng deterministic CUDA algorithms để tái lập benchmark.')
    parser.add_argument('--no_deterministic', action='store_false', dest='deterministic',
                        help='Cho phép thuật toán CUDA không deterministic để ưu tiên tốc độ.')
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Trọng số cho classification loss: CE(photo,text)+CE(sketch,text).")
    parser.add_argument("--lambda_triplet", type=float, default=1.0,
                        help="Trọng số cho triplet loss sketch-photo-negative.")
    
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--progress', action='store_true', default=True,
                        help='Hiện tqdm progress bar trong lúc train')
    parser.add_argument('--no_progress', action='store_false', dest='progress',
                        help='Tắt tqdm progress bar.')
    parser.add_argument('--quantize_fp16', action='store_true', default=True,
                        help='Chạy DFN5B teacher ở FP16 để giảm VRAM và tăng tốc.')
    parser.add_argument('--no_quantize_fp16', action='store_false', dest='quantize_fp16',
                        help='Giữ DFN5B teacher ở FP32.')
    parser.add_argument('--teacher_adapter_ckpt', type=str, default='',
                        help='Checkpoint modality adapter đã fine-tune cho DFN5B.')
    parser.add_argument('--joint_teacher_adapter', action='store_true', default=True,
                        help='Train DFN5B sketch/photo adapters jointly with the student.')
    parser.add_argument('--no_joint_teacher_adapter', action='store_false', dest='joint_teacher_adapter',
                        help='Tắt joint teacher adapter để chạy ablation/no-teacher.')
    parser.add_argument('--teacher_adapter_bottleneck', type=int, default=64)
    parser.add_argument('--teacher_adapter_lr', type=float, default=2e-5)
    parser.add_argument('--adapter_pca_init', action='store_true', default=True,
                        help='Khởi tạo teacher-adapter down projections bằng PCA theo modality.')
    parser.add_argument('--no_adapter_pca_init', action='store_false', dest='adapter_pca_init',
                        help='Dùng Xavier initialization gốc thay cho PCA.')
    parser.add_argument('--adapter_pca_batch_size', type=int, default=64,
                        help='Batch size encode DFN5B cho PCA calibration pass.')
    parser.add_argument('--adapter_pca_samples_per_class', type=int, default=0,
                        help='Số mẫu tối đa mỗi class/modality cho PCA; 0 dùng toàn bộ.')
    parser.add_argument('--lambda_teacher_retrieval', type=float, default=1.0,
                        help='Trọng số teacher-adapter triplet loss trên nhánh ablation này.')
    parser.add_argument('--lambda_teacher_semantic', type=float, default=1.0)
    parser.add_argument('--teacher_temperature', type=float, default=0.07)
    parser.add_argument('--teacher_triplet_margin', type=float, default=0.2)
    parser.add_argument('--lambda_kd', type=float, default=3.0,
                        help='Trọng số relational KD sketch-photo.')
    parser.add_argument('--kd_temperature', type=float, default=0.07,
                        help='Temperature cho phân phối similarity sketch-photo.')
                        
    parser.add_argument('--exp_name', type=str, default='teacher_adapter_pca_init')


    
    args = parser.parse_args()
    if args.adapter_pca_init and args.teacher_adapter_ckpt:
        parser.error("--adapter_pca_init cannot be combined with --teacher_adapter_ckpt.")
    if args.adapter_pca_init and not args.joint_teacher_adapter:
        parser.error("--adapter_pca_init requires joint teacher-adapter training.")
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath='saved_models/%s'%args.exp_name,
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    from pytorch_lightning.callbacks import TQDMProgressBar
    progress_bar = TQDMProgressBar(refresh_rate=20)

    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=not args.deterministic,
        deterministic=args.deterministic,
        num_sanity_val_steps=0,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback, progress_bar]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ZS_SBIR(args=args, classname=classnames)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    if args.adapter_pca_init and ckpt_path is None:
        initialize_teacher_adapters_from_pca(model, args)
        # Calibration must not change the random stream used by joint training.
        seed_everything(args.seed, deterministic=args.deterministic)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
