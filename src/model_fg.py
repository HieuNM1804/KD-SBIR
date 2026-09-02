import hashlib
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from clip import clip as openai_clip
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import TeacherFeatureDataset
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
from src.text_prompts import (
    MultiAspectPhotoTextPrompts,
    attention_diversity_loss,
    multi_aspect_infonce_loss,
    multi_aspect_similarity,
    relational_logits_kd_loss,
)


FG_CACHE_FORMAT_VERSION = 10
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


def fine_grained_multi_aspect_accuracy(
    query_features,
    gallery_aspects,
    query_categories,
    gallery_categories,
    query_instances,
    gallery_instances,
    aspect_temperature,
):
    """Micro Acc@K using multiple image-conditioned text features per photo."""
    query_features = query_features.float().cpu()
    gallery_aspects = gallery_aspects.float().cpu()
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
        current_gallery = gallery_aspects[gallery_mask]
        current_targets = query_instances[query_mask]
        current_gallery_ids = gallery_instances[gallery_mask]
        if len(current_gallery) != 100:
            raise RuntimeError(
                f"Category {int(category)} has {len(current_gallery)} gallery "
                "photos; expected exactly 100."
            )
        if len(torch.unique(current_gallery_ids)) != len(current_gallery_ids):
            raise RuntimeError(f"Category {int(category)} has duplicate photo IDs.")
        similarities = multi_aspect_similarity(
            current_queries,
            current_gallery,
            aspect_temperature,
        )
        ranking = torch.argsort(
            similarities, dim=-1, descending=True, stable=True
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
        top_k: torch.tensor(correct[top_k] / query_count, dtype=torch.float32)
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
            "text_prompt_aspects": getattr(args, "text_prompt_aspects", 0),
            "text_prompt_context_tokens": getattr(
                args, "text_prompt_context_tokens", 4
            ),
            "text_prompt_latent_width": getattr(
                args, "text_prompt_latent_width", 512
            ),
            "text_prompt_heads": getattr(args, "text_prompt_heads", 8),
            "text_prompt_dropout": getattr(args, "text_prompt_dropout", 0.1),
            "text_prompt_gate_init": getattr(
                args, "text_prompt_gate_init", 0.1
            ),
            "text_prompt_encode_chunk_size": getattr(
                args, "text_prompt_encode_chunk_size", 100
            ),
            "text_prompt_aspect_temperature": (
                getattr(args, "text_prompt_aspect_temperature", 0.1)
            ),
            "text_prompt_diversity_weight": (
                getattr(args, "text_prompt_diversity_weight", 0.01)
            ),
            "teacher_text_prompt_epochs": getattr(
                args, "teacher_text_prompt_epochs", 0
            ),
            "teacher_text_prompt_lr": getattr(
                args, "teacher_text_prompt_lr", 3e-4
            ),
            "teacher_text_prompt_weight_decay": (
                getattr(args, "teacher_text_prompt_weight_decay", 1e-4)
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
        super().__init__(cfg, clip_model, classnames, teacher=teacher)
        self.unseen_classnames = tuple(unseen_classnames)
        self.text_prompt_active = cfg.text_prompt_aspects > 0
        self.student_text_prompts = None
        self.teacher_text_prompts = None
        if not self.text_prompt_active:
            return

        student_visual_width = clip_model.visual.ln_pre.normalized_shape[0]
        self.student_text_prompts = MultiAspectPhotoTextPrompts(
            text_model=clip_model,
            tokenizer=openai_clip.tokenize,
            seen_classnames=self.classnames,
            unseen_classnames=self.unseen_classnames,
            visual_width=student_visual_width,
            latent_width=cfg.text_prompt_latent_width,
            aspects=cfg.text_prompt_aspects,
            context_tokens=cfg.text_prompt_context_tokens,
            heads=cfg.text_prompt_heads,
            dropout=cfg.text_prompt_dropout,
            gate_init=cfg.text_prompt_gate_init,
            seed=cfg.seed + 40_000,
            openclip=False,
            encode_chunk_size=cfg.text_prompt_encode_chunk_size,
        ).to(clip_model.visual.conv1.weight.device)
        if teacher is not None:
            teacher_visual_width = teacher.visual.conv1.out_channels
            self.teacher_text_prompts = MultiAspectPhotoTextPrompts(
                text_model=teacher,
                tokenizer=teacher.text_tokenizer,
                seen_classnames=self.classnames,
                unseen_classnames=self.unseen_classnames,
                visual_width=teacher_visual_width,
                latent_width=cfg.text_prompt_latent_width,
                aspects=cfg.text_prompt_aspects,
                context_tokens=cfg.text_prompt_context_tokens,
                heads=cfg.text_prompt_heads,
                dropout=cfg.text_prompt_dropout,
                gate_init=cfg.text_prompt_gate_init,
                seed=cfg.seed + 50_000,
                openclip=True,
                encode_chunk_size=cfg.text_prompt_encode_chunk_size,
            ).to(teacher.visual.conv1.weight.device)
        print(
            "[Multi-Aspect Text] photo patches -> "
            f"R={cfg.text_prompt_aspects} aspect prompts x "
            f"M={cfg.text_prompt_context_tokens} context tokens; "
            "visual/text encoders detached for this objective"
        )

    def _encode_student_text_source(self, images):
        pooled, patches = self.clip_model.visual(
            images.type(self.dtype),
            prompt=None,
            compound_prompts=None,
            return_patch_tokens=True,
        )
        return F.normalize(pooled.float(), dim=-1), patches.detach()

    def _encode_teacher_image(
        self, images, modality, return_patch_tokens=False
    ):
        if not return_patch_tokens:
            return super()._encode_teacher_image(images, modality)
        if self.teacher_prompts is not None:
            return self.teacher_prompts(
                images, modality, return_patch_tokens=True
            )

        visual = self._teacher.visual
        x = visual._embeds(images)
        x = visual.transformer(x)
        pooled, patches = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled, patches

    def _student_photo_text_aspects(self, images, categories, split):
        _, patches = self._encode_student_text_source(images)
        return self.student_text_prompts(
            self.clip_model,
            patches,
            categories,
            split=split,
        )

    def _teacher_photo_text_aspects(self, images, categories, split):
        _, patches = self._encode_teacher_image(
            images, "photo", return_patch_tokens=True
        )
        return self.teacher_text_prompts(
            self._teacher,
            patches,
            categories,
            split=split,
        )

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
                    photo, sketch, _, _, _, targets, _ = batch
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

    def _pretrain_teacher_text_prompts(
        self,
        train_dataset,
        val_sketch_loader,
        val_photo_loader,
        workers,
        show_progress,
    ):
        if self.teacher_text_prompts is None:
            raise RuntimeError("Teacher text prompt generator was not created.")
        cfg = self.cfg
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
        if self.teacher_prompts is not None:
            self.teacher_prompts.eval().requires_grad_(False)
        self.teacher_text_prompts.train().requires_grad_(True)
        sampler = FineGrainedFullGalleryBatchSampler(
            train_dataset,
            batch_size=cfg.teacher_pretrain_batch_size,
            seed=cfg.seed + 60_000,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=train_dataset.collate_full_gallery,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            generator=torch.Generator().manual_seed(cfg.seed + 60_000),
        )
        optimizer = torch.optim.AdamW(
            self.teacher_text_prompts.parameters(),
            lr=cfg.teacher_text_prompt_lr,
            weight_decay=cfg.teacher_text_prompt_weight_decay,
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
        best_state = None
        best_metrics = None

        for epoch in range(cfg.teacher_text_prompt_epochs):
            self.teacher_text_prompts.train()
            instance_total = 0.0
            diversity_total = 0.0
            steps = 0
            batches = tqdm(
                loader,
                desc=(
                    "Teacher text prompt "
                    f"{epoch + 1}/{cfg.teacher_text_prompt_epochs}"
                ),
                disable=not show_progress,
            )
            with torch.enable_grad():
                for batch in batches:
                    photo, sketch, _, _, categories, targets, _ = batch
                    photo = photo.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    sketch = sketch.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    categories = categories.to(teacher_device, non_blocking=True)
                    targets = targets.to(teacher_device, non_blocking=True)
                    gallery_categories = categories[:1].expand(len(photo))
                    with torch.no_grad():
                        sketch_features = self._encode_teacher_image(
                            sketch, "sketch"
                        ).float()
                        _, photo_patches = self._encode_teacher_image(
                            photo, "photo", return_patch_tokens=True
                        )
                    with torch.amp.autocast(
                        "cuda",
                        dtype=torch.float16,
                        enabled=teacher_device.type == "cuda",
                    ):
                        aspects, attention = self.teacher_text_prompts(
                            self._teacher,
                            photo_patches,
                            gallery_categories,
                            split="seen",
                        )
                        instance_loss, _ = multi_aspect_infonce_loss(
                            sketch_features,
                            aspects,
                            targets,
                            cfg.text_prompt_instance_temperature,
                            cfg.text_prompt_aspect_temperature,
                        )
                        diversity = attention_diversity_loss(attention)
                        loss = (
                            instance_loss
                            + cfg.text_prompt_diversity_weight * diversity
                        )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.teacher_text_prompts.parameters(),
                        cfg.text_prompt_gradient_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    instance_total += instance_loss.detach().item()
                    diversity_total += diversity.detach().item()
                    steps += 1
                    if show_progress:
                        batches.set_postfix(
                            TXT=f"{instance_loss.item():.3f}",
                            DIV=f"{diversity.item():.3f}",
                        )

            if steps == 0:
                raise RuntimeError("Teacher text pretraining produced no batches.")
            print(
                f"[Teacher Text Pretrain] epoch={epoch + 1}, "
                f"instance_nce={instance_total / steps:.6f}, "
                f"diversity={diversity_total / steps:.6f}"
            )
            metrics = self._validate_teacher_text_unseen(
                val_sketch_loader,
                val_photo_loader,
                epoch + 1,
                show_progress,
            )
            acc1 = metrics[1]
            acc5 = metrics[5]
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(acc1 + acc5 * 1e-6)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr != previous_lr:
                print(
                    "[Teacher Text LR] validation did not improve for "
                    f"{cfg.teacher_scheduler_patience} epochs; "
                    f"lr={current_lr:.3e}"
                )
            if better_acc1_acc5(acc1, acc5, best_acc1, best_acc5):
                best_acc1 = acc1
                best_acc5 = acc5
                best_epoch = epoch + 1
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.teacher_text_prompts.state_dict().items()
                }

        if best_state is None or best_metrics is None:
            raise RuntimeError("Teacher best text-prompt state was not created.")
        self.teacher_text_prompts.load_state_dict(best_state, strict=True)
        self.teacher_text_prompts.eval().requires_grad_(False)
        self.teacher_text_best_epoch = best_epoch
        self.teacher_text_best_metrics = best_metrics
        print(
            "[Teacher Text Best] restored generator from "
            f"epoch={best_epoch}, "
            f"{format_fine_grained_accuracies(best_metrics)}"
        )

    @torch.no_grad()
    def _validate_teacher_text_unseen(
        self,
        val_sketch_loader,
        val_photo_loader,
        epoch,
        show_progress,
    ):
        self.teacher_text_prompts.eval()
        if self.teacher_prompts is not None:
            self.teacher_prompts.eval()
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype

        sketch_features = []
        sketch_categories = []
        sketch_instances = []
        for images, categories, instances in tqdm(
            val_sketch_loader,
            desc=f"Teacher text sketch epoch {epoch}",
            disable=not show_progress,
        ):
            images = images.to(
                teacher_device, dtype=teacher_dtype, non_blocking=True
            )
            sketch_features.append(
                self._encode_teacher_image(images, "sketch").float().cpu()
            )
            sketch_categories.append(categories.cpu())
            sketch_instances.append(instances.cpu())

        photo_aspects = []
        photo_categories = []
        photo_instances = []
        for images, categories, instances in tqdm(
            val_photo_loader,
            desc=f"Teacher text photo epoch {epoch}",
            disable=not show_progress,
        ):
            images = images.to(
                teacher_device, dtype=teacher_dtype, non_blocking=True
            )
            categories = categories.to(teacher_device, non_blocking=True)
            aspects, _ = self._teacher_photo_text_aspects(
                images, categories, split="unseen"
            )
            photo_aspects.append(aspects.cpu())
            photo_categories.append(categories.cpu())
            photo_instances.append(instances.cpu())

        accuracies = fine_grained_multi_aspect_accuracy(
            torch.cat(sketch_features),
            torch.cat(photo_aspects),
            torch.cat(sketch_categories),
            torch.cat(photo_categories),
            torch.cat(sketch_instances),
            torch.cat(photo_instances),
            self.cfg.text_prompt_aspect_temperature,
        )
        print(
            f"[Teacher Text Validation] epoch={epoch}, "
            f"{format_fine_grained_accuracies(accuracies)}"
        )
        return {
            top_k: accuracy.item()
            for top_k, accuracy in accuracies.items()
        }

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

    @torch.no_grad()
    def _materialize_teacher_text_aspects(
        self,
        train_dataset,
        batch_size,
        workers,
        show_progress,
    ):
        self.teacher_text_prompts.eval()
        if self.teacher_prompts is not None:
            self.teacher_prompts.eval()
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
        photo_categories = fine_grained_train_metric_ids(train_dataset)[1]
        loader = DataLoader(
            TeacherFeatureDataset(
                train_dataset.all_photo_paths, self.cfg.max_size
            ),
            batch_size=min(batch_size, 100),
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if workers > 0 else None,
        )
        output = torch.empty(
            len(train_dataset.all_photo_paths),
            self.cfg.text_prompt_aspects,
            DFN5B_OUTPUT_DIM,
            dtype=torch.float16,
        )
        offset = 0
        for images in tqdm(
            loader,
            desc="Caching teacher photo text aspects",
            disable=not show_progress,
        ):
            end = offset + len(images)
            images = images.to(
                teacher_device, dtype=teacher_dtype, non_blocking=True
            )
            categories = photo_categories[offset:end].to(
                teacher_device, non_blocking=True
            )
            aspects, _ = self._teacher_photo_text_aspects(
                images, categories, split="seen"
            )
            output[offset:end].copy_(aspects.half().cpu())
            offset = end
        return output

    def _save_persistent_teacher_cache(
        self,
        train_dataset,
        sketch_features,
        photo_features,
        prompt_state,
        photo_text_aspects=None,
        text_prompt_state=None,
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
            "teacher_photo_text_aspects": (
                photo_text_aspects.cpu()
                if photo_text_aspects is not None
                else None
            ),
            "teacher_text_prompt_state_dict": text_prompt_state,
            "teacher_text_best_epoch": getattr(
                self, "teacher_text_best_epoch", None
            ),
            "teacher_text_best_metrics": getattr(
                self, "teacher_text_best_metrics", None
            ),
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

    def _load_persistent_teacher_cache(self, train_dataset):
        cache_path = Path(self.cfg.teacher_cache_path)
        try:
            payload = torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        expected = self._teacher_cache_metadata(train_dataset)
        actual = payload.get("metadata", {})
        mismatches = [
            key
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        ]
        required = ["teacher_sketch_features", "teacher_photo_features"]
        if self.text_prompt_active:
            required.append("teacher_photo_text_aspects")
        missing = [
            key
            for key in required
            if key not in payload or payload.get(key) is None
        ]
        if mismatches or missing:
            details = []
            if mismatches:
                details.append("metadata: " + ", ".join(mismatches))
            if missing:
                details.append("tensors: " + ", ".join(missing))
            raise RuntimeError(
                f"Teacher cache {cache_path} is incompatible "
                f"({'; '.join(details)}). Use another cache path or pass "
                "--rebuild_teacher_cache."
            )
        train_dataset.set_teacher_features(
            payload["teacher_sketch_features"],
            payload["teacher_photo_features"],
        )
        if self.text_prompt_active:
            train_dataset.set_teacher_text_aspects(
                payload["teacher_photo_text_aspects"]
            )
        self._teacher_sketch_text = payload.get("teacher_sketch_text")
        self._teacher_photo_text = payload.get("teacher_photo_text")
        self.teacher_active = True
        self.teacher_prompts = None
        self.teacher_text_prompts = None
        object.__setattr__(self, "_teacher", None)
        print(
            f"[Teacher Cache] loaded {cache_path} "
            f"({cache_path.stat().st_size / 1024**2:.1f} MB); "
            "restored pooled features and multi-aspect text targets."
        )

    def cache_teacher_features(
        self,
        train_dataset,
        val_sketch_loader,
        val_photo_loader,
        batch_size,
        workers,
        show_progress,
    ):
        if self.persistent_teacher_cache:
            self._load_persistent_teacher_cache(train_dataset)
            return
        if self._teacher is None:
            return

        if self.teacher_prompts is not None:
            self._pretrain_teacher_prompts(
                train_dataset,
                val_sketch_loader,
                val_photo_loader,
                workers,
                show_progress,
            )
            self.teacher_prompts.eval()

        if self.text_prompt_active:
            if self.cfg.teacher_text_prompt_epochs < 1:
                raise RuntimeError(
                    "A new multi-aspect teacher cache requires at least one "
                    "--teacher_text_prompt_epochs epoch."
                )
            self._pretrain_teacher_text_prompts(
                train_dataset,
                val_sketch_loader,
                val_photo_loader,
                workers,
                show_progress,
            )

        # Preserve the original fixed class text targets for compatibility with
        # the optional legacy loss and with existing cache inspection tools.
        self.get_teacher_text_features()
        sketch_features = self._materialize_teacher_features(
            train_dataset.all_sketches_path,
            "sketch",
            batch_size,
            workers,
            show_progress,
        )
        photo_features = self._materialize_teacher_features(
            train_dataset.all_photo_paths,
            "photo",
            batch_size,
            workers,
            show_progress,
        )
        train_dataset.set_teacher_features(sketch_features, photo_features)

        photo_text_aspects = None
        text_prompt_state = None
        if self.text_prompt_active:
            photo_text_aspects = self._materialize_teacher_text_aspects(
                train_dataset,
                batch_size,
                workers,
                show_progress,
            )
            train_dataset.set_teacher_text_aspects(photo_text_aspects)
            text_prompt_state = {
                key: value.detach().cpu()
                for key, value in self.teacher_text_prompts.state_dict().items()
            }
        prompt_state = (
            {
                key: value.detach().cpu()
                for key, value in self.teacher_prompts.state_dict().items()
            }
            if self.teacher_prompts is not None
            else {}
        )
        self._save_persistent_teacher_cache(
            train_dataset,
            sketch_features,
            photo_features,
            prompt_state,
            photo_text_aspects=photo_text_aspects,
            text_prompt_state=text_prompt_state,
        )

        teacher = self._teacher
        self.teacher_prompts = None
        self.teacher_text_prompts = None
        object.__setattr__(self, "_teacher", None)
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "[Teacher Cache] teacher released; student text training uses "
            "cached teacher sketch features and photo text aspects."
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
        if getattr(self.model, "text_prompt_active", False):
            parameters = list(self.model.student_text_prompts.parameters())
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.args.text_prompt_lr,
                weight_decay=self.args.text_prompt_weight_decay,
            )
            print(
                "[Text Optimizer] AdamW "
                f"lr={self.args.text_prompt_lr}, "
                f"weight_decay={self.args.text_prompt_weight_decay}, "
                f"params={sum(p.numel() for p in parameters):,}; "
                "domain/modality losses bypassed"
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

        adapter_parameters = (
            list(self.model.student_adapters.parameters())
            if self.model.student_adapters is not None
            else []
        )
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        prompt_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_ids
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
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum={self.args.momentum}, "
            f"weight_decay={self.args.weight_decay}, "
            f"prompt_params={prompt_trainable:,}, "
            f"adapter_lr={self.args.adapter_lr}, "
            f"adapter_weight_decay={self.args.adapter_weight_decay}, "
            f"adapter_params={adapter_trainable:,}"
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
            teacher_photo_text_aspects,
        ) = batch
        if self.model.text_prompt_active:
            if teacher_photo_text_aspects.numel() == 0:
                raise RuntimeError(
                    "Student text KD requires cached teacher photo aspects."
                )
            gallery_categories = categories[:1].expand(len(photo))
            with torch.no_grad():
                student_sketch, _ = self.model._encode_student_text_source(
                    sketch
                )
                _, student_photo_patches = (
                    self.model._encode_student_text_source(photo)
                )
            student_aspects, attention = self.model.student_text_prompts(
                self.model.clip_model,
                student_photo_patches,
                gallery_categories,
                split="seen",
            )
            instance_loss, student_logits = multi_aspect_infonce_loss(
                student_sketch,
                student_aspects,
                targets,
                self.args.text_prompt_instance_temperature,
                self.args.text_prompt_aspect_temperature,
            )
            teacher_logits = multi_aspect_similarity(
                teacher_sketch,
                teacher_photo_text_aspects,
                self.args.text_prompt_aspect_temperature,
            ) / self.args.text_prompt_instance_temperature
            kd_loss = relational_logits_kd_loss(
                student_logits,
                teacher_logits,
                self.args.text_prompt_kd_temperature,
            )
            diversity = attention_diversity_loss(attention)
            loss = (
                instance_loss
                + self.args.lambda_text_prompt_kd * kd_loss
                + self.args.text_prompt_diversity_weight * diversity
            )
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            self.log(
                "TEXT_NCE",
                instance_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                "TEXT_KD",
                kd_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return loss

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
        if self.model.text_prompt_active:
            if modality == "sketch":
                features, _ = self.model._encode_student_text_source(images)
            else:
                features, _ = self.model._student_photo_text_aspects(
                    images, categories, split="unseen"
                )
        else:
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
        if self.model.text_prompt_active:
            accuracies = fine_grained_multi_aspect_accuracy(
                sketch_features,
                photo_features,
                sketch_categories,
                photo_categories,
                sketch_instances,
                photo_instances,
                self.args.text_prompt_aspect_temperature,
            )
        else:
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
        metric_prefix = (
            "[Student Text Validation] "
            if self.model.text_prompt_active
            else ""
        )
        print(
            f"{metric_prefix}{format_fine_grained_accuracies(accuracies)}, "
            f"Best Acc@1: {self.best_acc1:.4f}, "
            f"Best Acc@5: {self.best_acc5:.4f}"
        )
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
        self.val_step_outputs_sketch.clear()
        self.val_step_outputs_photo.clear()
