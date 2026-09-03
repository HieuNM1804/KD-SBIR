import hashlib
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip import clip
from src.dataset_fg import FineGrainedFullGalleryBatchSampler
from src.losses_fg import (
    fine_grained_distillation_loss,
    fine_grained_teacher_infonce_loss,
)
from src.model import (
    DFN5B_OUTPUT_DIM,
    CustomCLIP,
    _load_clip_model,
    _load_teacher,
    _reduce_on_plateau_patience,
    _teacher_optimizer_parameter_groups,
    _teacher_training_config,
)
from src.patch_prompts import (
    SharedImageConditionedPrompt,
    shared_prompt_infonce_loss,
)


FG_CACHE_FORMAT_VERSION = 9
FG_ACCURACY_KS = (1, 5, 10, 20, 30, 40, 50)


def better_acc1_acc5(acc1, acc5, best_acc1, best_acc5):
    """Use Acc@1 as the primary metric and Acc@5 only as a tie-breaker."""
    return acc1 > best_acc1 or (acc1 == best_acc1 and acc5 > best_acc5)


def format_fine_grained_accuracies(accuracies):
    return ", ".join(
        f"Acc@{top_k}={float(accuracies[top_k]):.4f}"
        for top_k in FG_ACCURACY_KS
    )


def fine_grained_metric_history_entry(epoch, accuracies):
    return {
        "epoch": epoch,
        **{
            f"acc{top_k}": float(accuracies[top_k])
            for top_k in FG_ACCURACY_KS
        },
    }


def fine_grained_accuracy(
    query_features,
    gallery_features,
    query_categories,
    gallery_categories,
    query_instances,
    gallery_instances,
):
    """Micro Acc@K with a per-category exact-instance gallery."""
    query_features = F.normalize(query_features.float().cpu(), dim=-1)
    gallery_features = F.normalize(gallery_features.float().cpu(), dim=-1)
    query_categories = query_categories.long().cpu()
    gallery_categories = gallery_categories.long().cpu()
    query_instances = query_instances.long().cpu()
    gallery_instances = gallery_instances.long().cpu()

    correct = {top_k: 0 for top_k in FG_ACCURACY_KS}
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
        retrieved_ids = current_gallery_ids[
            ranking[:, : max(FG_ACCURACY_KS)]
        ]
        matches = retrieved_ids.eq(current_targets[:, None])
        for top_k in FG_ACCURACY_KS:
            correct[top_k] += int(matches[:, :top_k].any(dim=-1).sum().item())
        query_count += len(current_queries)

    if query_count == 0:
        raise RuntimeError("Fine-grained validation contains no sketch queries.")
    return {
        top_k: torch.tensor(
            correct[top_k] / query_count,
            dtype=torch.float32,
        )
        for top_k in FG_ACCURACY_KS
    }


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
    config.pop("triplet_margin", None)
    config.update(
        {
            "task": "fine_grained_exact_instance",
            "teacher_negative_scope": "full_100_photo_category_gallery",
            "teacher_objective": "exact_instance_infonce",
            "scheduler_monitor": "best_unseen_acc1_then_acc5",
            "teacher_instance_temperature": (
                args.teacher_instance_temperature
            ),
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
    def __init__(
        self,
        cfg,
        clip_model,
        classnames,
        unseen_classnames,
        teacher=None,
    ):
        super().__init__(cfg, clip_model, classnames, teacher)
        self.unseen_classnames = tuple(unseen_classnames)
        self.shared_patch_prompt = None
        if cfg.patch_prompt_context_tokens > 0:
            visual_width = clip_model.visual.ln_pre.normalized_shape[0]
            self.shared_patch_prompt = SharedImageConditionedPrompt(
                text_model=clip_model,
                tokenizer=clip.tokenize,
                seen_classnames=self.classnames,
                unseen_classnames=self.unseen_classnames,
                visual_width=visual_width,
                latent_width=cfg.patch_prompt_latent_width,
                context_tokens=cfg.patch_prompt_context_tokens,
                heads=cfg.patch_prompt_heads,
                dropout=cfg.patch_prompt_dropout,
                gate_init=cfg.patch_prompt_gate_init,
                seed=cfg.patch_prompt_seed,
                encode_chunk_size=cfg.patch_prompt_encode_chunk_size,
            )
            print(
                "[Shared Patch Prompt] one projector for photo + sketch; "
                f"all patches -> M={cfg.patch_prompt_context_tokens} text "
                f"tokens (latent={cfg.patch_prompt_latent_width}, "
                f"heads={cfg.patch_prompt_heads}, "
                f"trainable_params="
                f"{self.shared_patch_prompt.trainable_parameter_count():,}); "
                "patches_detached=True"
            )

    @property
    def patch_prompt_active(self):
        return self.shared_patch_prompt is not None

    def encode_student_image(
        self,
        image,
        modality,
        return_patch_tokens=False,
    ):
        visual_prompt, compound_prompts = self.get_visual_prompt(modality)
        output = self.clip_model.visual(
            image.type(self.dtype),
            visual_prompt,
            compound_prompts,
            return_patch_tokens=return_patch_tokens,
        )
        if return_patch_tokens:
            features, patch_tokens = output
        else:
            features = output
        features = self.apply_student_output_adapter(features, modality)
        features = F.normalize(features.float(), dim=-1)
        if return_patch_tokens:
            return features, patch_tokens
        return features

    def prompt_features(self, patch_tokens, categories, split):
        if self.shared_patch_prompt is None:
            return None
        features, _ = self.shared_patch_prompt(
            self.clip_model,
            patch_tokens,
            categories,
            split=split,
        )
        return features

    def forward(self, x):
        (
            photo_tensor,
            sketch_tensor,
            teacher_photo_base,
            teacher_sketch_base,
            categories,
        ) = x
        if self.patch_prompt_active:
            photo_features, photo_patches = self.encode_student_image(
                photo_tensor,
                "photo",
                return_patch_tokens=True,
            )
            sketch_features, sketch_patches = self.encode_student_image(
                sketch_tensor,
                "sketch",
                return_patch_tokens=True,
            )
            photo_categories = categories[:1].expand(len(photo_tensor))
            photo_prompt_features = self.prompt_features(
                photo_patches,
                photo_categories,
                split="seen",
            )
            sketch_prompt_features = self.prompt_features(
                sketch_patches,
                categories,
                split="seen",
            )
        else:
            photo_features = self.encode_student_image(photo_tensor, "photo")
            sketch_features = self.encode_student_image(
                sketch_tensor, "sketch"
            )
            photo_prompt_features = None
            sketch_prompt_features = None

        student_photo_text = (
            F.normalize(self.get_student_text_features("photo"), dim=-1)
            if self.photo_text_active
            else None
        )
        student_sketch_text = (
            F.normalize(self.get_student_text_features("sketch"), dim=-1)
            if self.sketch_text_active
            else None
        )
        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            teacher_photo_features = teacher_photo_base
            teacher_sketch_features = teacher_sketch_base
            if self.image_text_kd_active:
                teacher_sketch_text, teacher_photo_text = (
                    self.get_teacher_text_features()
                )

        return (
            photo_features,
            sketch_features,
            teacher_photo_features,
            teacher_sketch_features,
            self.teacher_active,
            student_sketch_text,
            student_photo_text,
            teacher_sketch_text,
            teacher_photo_text,
            sketch_prompt_features,
            photo_prompt_features,
        )

    def extract_feature_and_prompt(
        self,
        image,
        modality,
        categories,
        split="unseen",
    ):
        if not self.patch_prompt_active:
            return self.encode_student_image(image, modality), None
        features, patches = self.encode_student_image(
            image,
            modality,
            return_patch_tokens=True,
        )
        return features, self.prompt_features(patches, categories, split)

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
            _teacher_optimizer_parameter_groups(self.teacher_prompts, cfg),
            lr=cfg.teacher_prompt_lr,
            momentum=cfg.teacher_momentum,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.teacher_scheduler_gamma,
            patience=_reduce_on_plateau_patience(
                cfg.teacher_scheduler_patience
            ),
            threshold=0.0,
            threshold_mode="abs",
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=teacher_device.type == "cuda"
        )
        best_acc1 = -float("inf")
        best_acc5 = -float("inf")
        best_epoch = 0
        best_prompt_state = None
        best_unseen_metrics = None
        self.teacher_train_metric_history = []
        self.teacher_unseen_metric_history = []

        for epoch in range(cfg.teacher_pretrain_epochs):
            self.teacher_prompts.train()
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
                        retrieval = fine_grained_teacher_infonce_loss(
                            sketch_features,
                            photo_features,
                            targets,
                            cfg.teacher_instance_temperature,
                        )
                        loss = cfg.lambda_teacher_retrieval * retrieval
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    retrieval_total += retrieval.detach().item()
                    steps += 1
                    if show_progress:
                        batches.set_postfix(T_NCE=f"{retrieval.item():.3f}")

            if steps == 0:
                raise RuntimeError("Teacher pretraining produced no batches.")
            print(
                f"[Teacher Pretrain] epoch={epoch + 1}, "
                f"instance_nce={retrieval_total / steps:.6f}"
            )
            train_metrics = self._validate_teacher_train(
                train_dataset,
                epoch + 1,
                workers,
                show_progress,
            )
            self.teacher_train_metric_history.append(
                fine_grained_metric_history_entry(epoch + 1, train_metrics)
            )
            unseen_metrics = self._validate_teacher_unseen(
                val_sketch_loader,
                val_photo_loader,
                epoch + 1,
                show_progress,
            )
            acc1 = unseen_metrics[1]
            acc5 = unseen_metrics[5]
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(acc1 + acc5 * 1e-6)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr != previous_lr:
                print(
                    "[Teacher LR] validation did not improve for "
                    f"{cfg.teacher_scheduler_patience} epochs; "
                    f"prompt_lr={current_lr:.3e}"
                )
            self.teacher_unseen_metric_history.append(
                fine_grained_metric_history_entry(epoch + 1, unseen_metrics)
            )
            improved = better_acc1_acc5(
                acc1, acc5, best_acc1, best_acc5
            )
            if improved:
                best_acc1 = acc1
                best_acc5 = acc5
                best_epoch = epoch + 1
                best_unseen_metrics = dict(unseen_metrics)
                best_prompt_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.teacher_prompts.state_dict().items()
                }

        if best_prompt_state is None or best_unseen_metrics is None:
            raise RuntimeError("Teacher best-Acc@1 state was not created.")
        self.teacher_prompts.load_state_dict(best_prompt_state, strict=True)
        self.teacher_prompts.eval()
        self.teacher_best_epoch = best_epoch
        self.teacher_best_acc1 = best_acc1
        self.teacher_best_acc5 = best_acc5
        self.teacher_best_metrics = best_unseen_metrics
        print(
            "[Teacher Best] restored visual prompts from "
            f"epoch={best_epoch}, "
            f"{format_fine_grained_accuracies(best_unseen_metrics)}"
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
        self.teacher_prompts.eval()
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
        accuracies = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        print(
            f"[Teacher Train Evaluation] epoch={epoch}, "
            f"{format_fine_grained_accuracies(accuracies)}"
        )
        return {
            top_k: accuracy.item()
            for top_k, accuracy in accuracies.items()
        }

    @torch.no_grad()
    def _validate_teacher_unseen(
        self,
        val_sketch_loader,
        val_photo_loader,
        epoch,
        show_progress,
    ):
        self.teacher_prompts.eval()
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
        accuracies = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        print(
            f"[Teacher Validation] epoch={epoch}, "
            f"{format_fine_grained_accuracies(accuracies)}"
        )
        return {
            top_k: accuracy.item()
            for top_k, accuracy in accuracies.items()
        }

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
            "teacher_best_metrics": getattr(
                self,
                "teacher_best_metrics",
                None,
            ),
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
    def __init__(self, args, classnames, unseen_classnames=()):
        super().__init__()
        self.args = args
        clip_model = _load_clip_model(args.backbone)
        teacher = _load_teacher(args)
        self.model = FineGrainedCustomCLIP(
            cfg=args,
            clip_model=clip_model,
            classnames=classnames,
            unseen_classnames=unseen_classnames,
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
        adapter_parameters = (
            list(self.model.student_adapters.parameters())
            if self.model.student_adapters is not None
            else []
        )
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        patch_prompt_parameters = (
            list(self.model.shared_patch_prompt.parameters())
            if getattr(self.model, "shared_patch_prompt", None) is not None
            else []
        )
        patch_prompt_ids = {
            id(parameter) for parameter in patch_prompt_parameters
        }
        prompt_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in adapter_ids
            and id(parameter) not in patch_prompt_ids
        ]
        param_groups = [
            {
                "params": prompt_parameters,
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "name": "prompts",
            }
        ]
        if adapter_parameters:
            param_groups.append(
                {
                    "params": adapter_parameters,
                    "lr": self.args.adapter_lr,
                    "weight_decay": self.args.adapter_weight_decay,
                    "name": "adapters",
                }
            )
        if patch_prompt_parameters:
            param_groups.append(
                {
                    "params": patch_prompt_parameters,
                    "lr": self.args.patch_prompt_lr,
                    "weight_decay": self.args.patch_prompt_weight_decay,
                    "name": "shared_patch_prompt",
                }
            )
        optimizer = torch.optim.SGD(
            param_groups,
            lr=self.args.lr,
            momentum=self.args.momentum,
        )
        prompt_trainable = sum(
            parameter.numel() for parameter in prompt_parameters
        )
        adapter_trainable = sum(
            parameter.numel() for parameter in adapter_parameters
        )
        patch_prompt_trainable = sum(
            parameter.numel() for parameter in patch_prompt_parameters
        )
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum={self.args.momentum}, "
            f"weight_decay={self.args.weight_decay}, "
            f"prompt_params={prompt_trainable:,}, "
            f"adapter_lr={self.args.adapter_lr}, "
            f"adapter_weight_decay={self.args.adapter_weight_decay}, "
            f"adapter_params={adapter_trainable:,}, "
            "patch_prompt_lr="
            f"{getattr(self.args, 'patch_prompt_lr', 0.0)}, "
            "patch_prompt_weight_decay="
            f"{getattr(self.args, 'patch_prompt_weight_decay', 0.0)}, "
            f"patch_prompt_params={patch_prompt_trainable:,}"
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.args.scheduler_gamma,
            patience=_reduce_on_plateau_patience(
                self.args.scheduler_patience
            ),
            threshold=0.0,
            threshold_mode="abs",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "fg_selection",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        (
            photo,
            sketch,
            teacher_photo,
            teacher_sketch,
            categories,
            targets,
        ) = batch
        features = self.model(
            (photo, sketch, teacher_photo, teacher_sketch, categories)
        )
        loss, loss_dict = fine_grained_distillation_loss(
            self.args, features[:9]
        )
        prompt_loss = loss.new_zeros(())
        sketch_prompt_features, photo_prompt_features = features[9:]
        if self.model.patch_prompt_active:
            prompt_loss = shared_prompt_infonce_loss(
                sketch_prompt_features,
                photo_prompt_features,
                targets,
                self.args.patch_prompt_temperature,
            )
            loss = loss + self.args.lambda_patch_prompt * prompt_loss
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
        self.log(
            "PROMPT_NCE",
            prompt_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=self.model.patch_prompt_active,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, categories, instances = batch
        modality = "sketch" if dataloader_idx == 0 else "photo"
        features, prompt_features = self.model.extract_feature_and_prompt(
            images,
            modality,
            categories,
            split="unseen",
        )
        output = (
            features.detach(),
            (
                prompt_features.detach()
                if prompt_features is not None
                else None
            ),
            categories.detach(),
            instances.detach(),
        )
        if dataloader_idx == 0:
            self.val_step_outputs_sketch.append(output)
        else:
            self.val_step_outputs_photo.append(output)

    def on_validation_epoch_end(self):
        def combine(outputs):
            visual = torch.cat([output[0] for output in outputs]).cpu()
            prompt = (
                torch.cat([output[1] for output in outputs]).cpu()
                if outputs[0][1] is not None
                else None
            )
            categories = torch.cat([output[2] for output in outputs]).cpu()
            instances = torch.cat([output[3] for output in outputs]).cpu()
            return visual, prompt, categories, instances

        (
            sketch_features,
            sketch_prompt_features,
            sketch_categories,
            sketch_instances,
        ) = combine(self.val_step_outputs_sketch)
        (
            photo_features,
            photo_prompt_features,
            photo_categories,
            photo_instances,
        ) = combine(self.val_step_outputs_photo)
        accuracies = fine_grained_accuracy(
            sketch_features,
            photo_features,
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        )
        acc1 = accuracies[1]
        acc5 = accuracies[5]
        selection = acc1 + acc5 * 1e-6
        for top_k, accuracy in accuracies.items():
            self.log(
                f"acc{top_k}",
                accuracy,
                on_step=False,
                on_epoch=True,
            )
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
            "[Student Visual Validation] "
            f"{format_fine_grained_accuracies(accuracies)}, "
            f"Best Acc@1: {self.best_acc1:.4f}, "
            f"Best Acc@5: {self.best_acc5:.4f}"
        )
        if sketch_prompt_features is not None:
            prompt_accuracies = fine_grained_accuracy(
                sketch_prompt_features,
                photo_prompt_features,
                sketch_categories,
                photo_categories,
                sketch_instances,
                photo_instances,
            )
            for top_k, accuracy in prompt_accuracies.items():
                self.log(
                    f"prompt_acc{top_k}",
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                )
            print(
                "[Student Prompt Validation] "
                f"{format_fine_grained_accuracies(prompt_accuracies)}"
            )
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
        self.val_step_outputs_sketch.clear()
        self.val_step_outputs_photo.clear()
