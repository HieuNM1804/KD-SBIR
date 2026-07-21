import os

# Required by deterministic CUDA matrix multiplication. It must be set before
# importing torch and before CUDA is initialized.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.dataset import TrainDataset, ValidDataset, WorkerInvariantSampler
from src.model import ZS_SBIR
from src.data_config import UNSEEN_CLASSES


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_datasets(args):
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
    parser.add_argument("--root", type=str, required=True, help="dataset root containing sketch/ and photo/")
    parser.add_argument("--ckpt_path", type=str, default="", help="student checkpoint to resume")
    parser.add_argument("--dataset", type=str, default="sketchy_1",
                        choices=sorted(UNSEEN_CLASSES), help="zero-shot split")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Python, NumPy, PyTorch, and DataLoader workers.")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Weight for CE(photo, text) + CE(sketch, text).")
    
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of DataLoader workers; sample RNG is worker-count invariant.')
    parser.add_argument('--progress', action='store_true', default=True,
                        help='Show the tqdm training progress bar.')
    parser.add_argument('--no_progress', action='store_false', dest='progress',
                        help='Disable the tqdm progress bar.')
    parser.add_argument('--joint_teacher_adapter', action='store_true', default=True,
                        help='Train DFN5B sketch/photo adapters jointly with the student.')
    parser.add_argument('--no_joint_teacher_adapter', action='store_false', dest='joint_teacher_adapter',
                        help='Disable joint teacher-adapter training for ablations.')
    parser.add_argument('--teacher_adapter_bottleneck', type=int, default=64)
    parser.add_argument('--teacher_adapter_lr', type=float, default=2e-5)
    parser.add_argument('--lambda_teacher_retrieval', type=float, default=1.5,
                        help='Weight for the teacher-adapter retrieval loss.')
    parser.add_argument('--lambda_teacher_semantic', type=float, default=1.0)
    parser.add_argument('--teacher_temperature', type=float, default=0.07)
    parser.add_argument('--teacher_triplet_margin', type=float, default=0.2)
    parser.add_argument('--lambda_kd', type=float, default=3.0,
                        help='Weight for sketch-photo relational distillation.')
    parser.add_argument('--kd_temperature', type=float, default=0.07,
                        help='Temperature for the sketch-photo similarity distribution.')
                        
    parser.add_argument('--exp_name', type=str, default='no_student_triplet_worker_invariant')


    
    args = parser.parse_args()
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
        benchmark=False,
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback, progress_bar]
    )

    categories = set(os.listdir(os.path.join(args.root, "sketch")))
    classnames = sorted(
        categories
        - {".ipynb_checkpoints"}
        - set(UNSEEN_CLASSES[args.dataset])
    )
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ZS_SBIR(args=args, classname=classnames)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
