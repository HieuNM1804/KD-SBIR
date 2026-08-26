import hashlib
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset_fg import FineGrainedFullGalleryBatchSampler
from src.losses_fg import (
    fine_grained_distillation_loss,
    fine_grained_teacher_triplet_loss,
)
from src.model import (
    DFN5B_OUTPUT_DIM,
    CustomCLIP,
    _load_clip_model,
    _load_teacher,
    _teacher_training_config,
)


FG_CACHE_FORMAT_VERSION = 3


def better_acc1_acc5(acc1, acc5, best_acc1, best_acc5):
    """Use Acc@1 as the primary metric and Acc@5 only as a tie-breaker."""
    return acc1 > best_acc1 or (acc1 == best_acc1 and acc5 > best_acc5)


def fine_grained_accuracy(
    query_features,
    gallery_features,
    query_categories,
    gallery_categories,
    query_instances,
    gallery_instances,
):
    """Micro Acc@1/Acc@5 with a per-category exact-instance gallery."""
    query_features = F.normalize(query_features.float().cpu(), dim=-1)
    gallery_features = F.normalize(gallery_features.float().cpu(), dim=-1)
    query_categories = query_categories.long().cpu()
    gallery_categories = gallery_categories.long().cpu()
    query_instances = query_instances.long().cpu()
    gallery_instances = gallery_instances.long().cpu()

    top1 = 0
    top5 = 0
    query_count = 0
    for category in torch.unique(query_categories, sorted=True):
        query_mask = query_categories.eq(category)
        gallery_mask = gallery_categories.eq(category)
        current_queries = query_features[query_mask]
        current_gallery = gallery_features[gallery_mask]
        current_targets = query_instances[query_mask]
        current_gallery_ids = gallery_instances[gallery_mask]
        if len(current_gallery) != 100:
            raise RuntimeError(
                f"Category {int(category)} has {len(current_gallery)} gallery "
                "photos; expected exactly 100."
            )
        if len(torch.unique(current_gallery_ids)) != len(current_gallery_ids):
            raise RuntimeError(f"Category {int(category)} has duplicate photo IDs.")

        similarities = current_queries @ current_gallery.t()
        ranking = torch.argsort(
            similarities,
            dim=-1,
            descending=True,
            stable=True,
        )
        retrieved_ids = current_gallery_ids[ranking[:, :5]]
        matches = retrieved_ids.eq(current_targets[:, None])
        top1 += int(matches[:, 0].sum().item())
        top5 += int(matches.any(dim=-1).sum().item())
        query_count += len(current_queries)

    if query_count == 0:
        raise RuntimeError("Fine-grained validation contains no sketch queries.")
    return (
        torch.tensor(top1 / query_count, dtype=torch.float32),
        torch.tensor(top5 / query_count, dtype=torch.float32),
    )


def fine_grained_train_metric_ids(train_dataset):
    """Build exact-instance IDs for the seen sketch/photo retrieval split."""
    sketch_categories = torch.as_tensor(
        train_dataset.sample_category_ids,
        dtype=torch.long,
    )
    sketch_instances = torch.as_tensor(
        train_dataset.sample_local_photo_indices,
        dtype=torch.long,
    )
    photo_categories = torch.empty(
        len(train_dataset.all_photo_paths),
        dtype=torch.long,
    )
    photo_instances = torch.empty_like(photo_categories)

    for category, photo_indices in (
        train_dataset.category_to_photo_indices.items()
    ):
        if len(photo_indices) != 100:
            raise RuntimeError(
                f"Seen category {category} has {len(photo_indices)} photos; "
                "expected exactly 100."
            )
        indices = torch.as_tensor(photo_indices, dtype=torch.long)
        photo_categories[indices] = int(category)
        photo_instances[indices] = torch.arange(len(indices), dtype=torch.long)

    return (
        sketch_categories,
        photo_categories,
        sketch_instances,
        photo_instances,
    )


def _dataset_fingerprint(args, train_dataset):
    digest = hashlib.sha256()
    digest.update(b"fine-grained-exact-pair\0")
    digest.update(args.dataset.encode("utf-8"))
    digest.update(str(train_dataset.max_size).encode("utf-8"))
    all_paths = []
    for category in train_dataset.index.categories:
        all_paths.extend(train_dataset.index.sketches_by_category[category])
        all_paths.extend(train_dataset.index.photos_by_category[category])
    for path in sorted(all_paths):
        relative = os.path.relpath(path, args.root).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fg_teacher_config(args):
    config = _teacher_training_config(args)
    config.update(
        {
            "task": "fine_grained_exact_instance",
            "teacher_negative_scope": "full_100_photo_category_gallery",
            "checkpoint_selection": "best_unseen_acc1_then_acc5",
        }
    )
    return config


def default_teacher_cache_path(args, train_dataset):
    cache_key = {
        "format_version": FG_CACHE_FORMAT_VERSION,
        "dataset_fingerprint": _dataset_fingerprint(args, train_dataset),
        "teacher": _fg_teacher_config(args),
    }
    encoded = json.dumps(
        cache_key, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_hash = hashlib.sha256(encoded).hexdigest()[:16]
    cache_dir = args.teacher_cache_dir
    if not cache_dir:
        kaggle_working = Path("/kaggle/working")
        cache_dir = (
            kaggle_working / "teacher_cache"
            if kaggle_working.is_dir()
            else Path("teacher_cache")
        )
    return str(Path(cache_dir) / f"sketchy_fg_{config_hash}.pt")


class FineGrainedCustomCLIP(CustomCLIP):
    def _teacher_cache_metadata(self, train_dataset):
        return {
            "format_version": FG_CACHE_FORMAT_VERSION,
            "dataset": self.cfg.dataset,
            "task": "fine_grained_exact_instance",
            "max_size": train_dataset.max_size,
            "classnames": list(self.classnames),
            "sketch_count": len(train_dataset.all_sketches_path),
            "photo_count": len(train_dataset.all_photo_paths),
            "dataset_fingerprint": _dataset_fingerprint(
                self.cfg, train_dataset
            ),
            **_fg_teacher_config(self.cfg),
        }

    def _pretrain_teacher_prompts(
        self,
        train_dataset,
        val_sketch_loader,
        val_photo_loader,
        workers,
        show_progress,
    ):
        if self.teacher_prompts is None:
            raise RuntimeError(
                "Teacher prompt pretraining requires teacher_pretrain_epochs > 0."
            )

        cfg = self.cfg
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
        self.teacher_prompts.requires_grad_(True)
        sampler = FineGrainedFullGalleryBatchSampler(
            train_dataset,
            batch_size=cfg.teacher_pretrain_batch_size,
            seed=cfg.seed + 10_000,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=train_dataset.collate_full_gallery,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            generator=torch.Generator().manual_seed(cfg.seed + 10_000),
        )
        optimizer = torch.optim.SGD(
            self.teacher_prompts.parameters(),
            lr=cfg.teacher_prompt_lr,
            momentum=cfg.teacher_momentum,
            weight_decay=cfg.teacher_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.teacher_scheduler_step_size,
            gamma=cfg.teacher_scheduler_gamma,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=teacher_device.type == "cuda"
        )
        best_acc1 = -float("inf")
        best_acc5 = -float("inf")
        best_epoch = 0
        best_prompt_state = None
        self.teacher_train_metric_history = []
        self.teacher_unseen_metric_history = []

        for epoch in range(cfg.teacher_pretrain_epochs):
            retrieval_total = 0.0
            steps = 0
            batches = tqdm(
                loader,
                desc=f"Teacher FG pretrain {epoch + 1}/{cfg.teacher_pretrain_epochs}",
                disable=not show_progress,
            )
            with torch.enable_grad():
                for batch in batches:
                    photo, sketch, _, _, _, targets = batch
                    photo = photo.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    sketch = sketch.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    targets = targets.to(teacher_device, non_blocking=True)
                    with torch.amp.autocast(
                        "cuda",
                        dtype=torch.float16,
                        enabled=teacher_device.type == "cuda",
                    ):
                        photo_features = self._encode_teacher_image(
                            photo, "photo"
                        )
                        sketch_features = self._encode_teacher_image(
                            sketch, "sketch"
                        )
                        retrieval = fine_grained_teacher_triplet_loss(
                            sketch_features,
                            photo_features,
                            targets,
                            cfg.teacher_triplet_margin,
                        )
                        loss = cfg.lambda_teacher_retrieval * retrieval
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    retrieval_total += retrieval.detach().item()
                    steps += 1
                    if show_progress:
                        batches.set_postfix(T_TRI=f"{retrieval.item():.3f}")

            scheduler.step()
            if steps == 0:
                raise RuntimeError("Teacher pretraining produced no batches.")
            print(
                f"[Teacher Pretrain] epoch={epoch + 1}, "
                f"retrieval={retrieval_total / steps:.6f}"
            )
            train_acc1, train_acc5 = self._validate_teacher_train(
                train_dataset,
                epoch + 1,
                workers,
                show_progress,
            )
            self.teacher_train_metric_history.append(
                {
                    "epoch": epoch + 1,
                    "acc1": train_acc1,
                    "acc5": train_acc5,
                }
            )
            acc1, acc5 = self._validate_teacher_unseen(
                val_sketch_loader,
                val_photo_loader,
                epoch + 1,
                show_progress,
            )
            self.teacher_unseen_metric_history.append(
                {
                    "epoch": epoch + 1,
                    "acc1": acc1,
                    "acc5": acc5,
                }
            )
            improved = better_acc1_acc5(
                acc1, acc5, best_acc1, best_acc5
            )
            if improved:
                best_acc1 = acc1
                best_acc5 = acc5
                best_epoch = epoch + 1
                best_prompt_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.teacher_prompts.state_dict().items()
                }

        if best_prompt_state is None:
            raise RuntimeError("Teacher best-Acc@1 state was not created.")
        self.teacher_prompts.load_state_dict(best_prompt_state, strict=True)
        self.teacher_best_epoch = best_epoch
        self.teacher_best_acc1 = best_acc1
        self.teacher_best_acc5 = best_acc5
        print(
            "[Teacher Best] restored visual prompts from "
            f"epoch={best_epoch}, Acc@1={best_acc1:.4f}, Acc@5={best_acc5:.4f}"
        )
        self.teacher_prompts.requires_grad_(False)

    @torch.no_grad()
    def _validate_teacher_train(
        self,
        train_dataset,
        epoch,
        workers,
        show_progress,
    ):
        """Evaluate exact-instance retrieval on all seen training sketches."""
        batch_size = self.cfg.test_batch_size
        sketch_features = self._materialize_teacher_features(
            train_dataset.all_sketches_path,
            "sketch",
            batch_size,
            workers,
            show_progress,
            generator_seed=self.cfg.seed + 20_000 + epoch * 2,
        )
        photo_features = self._materialize_teacher_features(
            train_dataset.all_photo_paths,
            "photo",
            batch_size,
            workers,
            show_progress,
            generator_seed=self.cfg.seed + 20_001 + epoch * 2,
        )
        (
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        ) = fine_grained_train_metric_ids(train_dataset)
        acc1, acc5 = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        print(
            f"[Teacher Train Evaluation] epoch={epoch}, "
            f"Acc@1={acc1.item():.4f}, Acc@5={acc5.item():.4f}"
        )
        return acc1.item(), acc5.item()

    @torch.no_grad()
    def _validate_teacher_unseen(
        self,
        val_sketch_loader,
        val_photo_loader,
        epoch,
        show_progress,
    ):
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype

        def encode_loader(loader, modality):
            features = []
            categories = []
            instances = []
            batches = tqdm(
                loader,
                desc=f"Teacher FG {modality} epoch {epoch}",
                disable=not show_progress,
            )
            for images, current_categories, current_instances in batches:
                images = images.to(
                    teacher_device, dtype=teacher_dtype, non_blocking=True
                )
                current_features = self._encode_teacher_image(images, modality)
                features.append(current_features.float().cpu())
                categories.append(current_categories.cpu())
                instances.append(current_instances.cpu())
            return (
                torch.cat(features),
                torch.cat(categories),
                torch.cat(instances),
            )

        sketch_features, sketch_categories, sketch_instances = encode_loader(
            val_sketch_loader, "sketch"
        )
        photo_features, photo_categories, photo_instances = encode_loader(
            val_photo_loader, "photo"
        )
        acc1, acc5 = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        print(
            f"[Teacher Validation] epoch={epoch}, "
            f"Acc@1={acc1.item():.4f}, Acc@5={acc5.item():.4f}"
        )
        return acc1.item(), acc5.item()

    def _save_persistent_teacher_cache(
        self,
        train_dataset,
        sketch_features,
        photo_features,
        prompt_state,
    ):
        if not self.cfg.teacher_cache_path:
            return
        cache_path = Path(self.cfg.teacher_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(cache_path.name + ".tmp")
        payload = {
            "metadata": self._teacher_cache_metadata(train_dataset),
            "teacher_sketch_features": sketch_features.cpu(),
            "teacher_photo_features": photo_features.cpu(),
            "teacher_sketch_text": self._teacher_sketch_text.detach().cpu(),
            "teacher_photo_text": self._teacher_photo_text.detach().cpu(),
            "teacher_prompt_state_dict": prompt_state,
            "teacher_best_epoch": getattr(self, "teacher_best_epoch", None),
            "teacher_best_acc1": getattr(self, "teacher_best_acc1", None),
            "teacher_best_acc5": getattr(self, "teacher_best_acc5", None),
            "teacher_train_metric_history": getattr(
                self,
                "teacher_train_metric_history",
                [],
            ),
            "teacher_unseen_metric_history": getattr(
                self,
                "teacher_unseen_metric_history",
                [],
            ),
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
        print(
            f"[Teacher Cache] saved {cache_path} "
            f"({cache_path.stat().st_size / 1024**2:.1f} MB)."
        )


class FineGrainedZS_SBIR(pl.LightningModule):
    def __init__(self, args, classnames):
        super().__init__()
        self.args = args
        clip_model = _load_clip_model(args.backbone)
        teacher = _load_teacher(args)
        self.model = FineGrainedCustomCLIP(
            cfg=args,
            clip_model=clip_model,
            classnames=classnames,
            teacher=teacher,
        )
        self.best_acc1 = 0.0
        self.best_acc5 = 0.0
        self.val_step_outputs_sketch = []
        self.val_step_outputs_photo = []

    def cache_teacher_features(
        self,
        train_dataset,
        val_sketch_loader,
        val_photo_loader,
        batch_size,
        workers,
        show_progress,
    ):
        self.model.cache_teacher_features(
            train_dataset,
            val_sketch_loader,
            val_photo_loader,
            batch_size,
            workers,
            show_progress,
        )

    def configure_optimizers(self):
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        trainable = sum(parameter.numel() for parameter in parameters)
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum={self.args.momentum}, "
            f"weight_decay={self.args.weight_decay}, "
            f"trainable_params={trainable:,}"
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        photo, sketch, teacher_photo, teacher_sketch, categories, _ = batch
        features = self.model(
            (photo, sketch, teacher_photo, teacher_sketch, categories)
        )
        loss, loss_dict = fine_grained_distillation_loss(self.args, features)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "DOMAIN",
            loss_dict["domain_kd"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "MODALITY",
            loss_dict["modality_kd"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, categories, instances = batch
        modality = "sketch" if dataloader_idx == 0 else "photo"
        features = self.model.extract_feature(images, modality)
        output = (features.detach(), categories.detach(), instances.detach())
        if dataloader_idx == 0:
            self.val_step_outputs_sketch.append(output)
        else:
            self.val_step_outputs_photo.append(output)

    def on_validation_epoch_end(self):
        def combine(outputs):
            return tuple(
                torch.cat([output[index] for output in outputs]).cpu()
                for index in range(3)
            )

        sketch_features, sketch_categories, sketch_instances = combine(
            self.val_step_outputs_sketch
        )
        photo_features, photo_categories, photo_instances = combine(
            self.val_step_outputs_photo
        )
        acc1, acc5 = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        selection = acc1 + acc5 * 1e-6
        self.log("acc1", acc1, on_step=False, on_epoch=True)
        self.log("acc5", acc5, on_step=False, on_epoch=True)
        self.log("fg_selection", selection, on_step=False, on_epoch=True)

        if self.global_step > 0 and better_acc1_acc5(
            acc1.item(),
            acc5.item(),
            self.best_acc1,
            self.best_acc5,
        ):
            self.best_acc1 = acc1.item()
            self.best_acc5 = acc5.item()
        print(
            f"Acc@1: {acc1.item():.4f}, Acc@5: {acc5.item():.4f}, "
            f"Best Acc@1: {self.best_acc1:.4f}, "
            f"Best Acc@5: {self.best_acc5:.4f}"
        )
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
        self.val_step_outputs_sketch.clear()
        self.val_step_outputs_photo.clear()
