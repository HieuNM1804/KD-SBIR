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
    fine_grained_prompt_infonce_loss,
    fine_grained_prompt_retrieval_accuracy,
    fine_grained_teacher_infonce_loss,
    image_conditioned_text_anchor_loss,
    teacher_semantic_refinement_loss,
    teacher_visual_refinement_control_loss,
)
from src.image_text_prompts import ImageConditionedTextPromptLearner
from src.teacher_prompts import build_teacher_prompt_controller
from src.teacher_refinement_report import write_teacher_refinement_report
from src.model import (
    DFN5B_OUTPUT_DIM,
    CustomCLIP,
    _load_clip_model,
    _load_teacher,
    _reduce_on_plateau_patience,
    _teacher_training_config,
)


FG_CACHE_FORMAT_VERSION = 9


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
    config.pop("triplet_margin", None)
    config.update(
        {
            "task": "fine_grained_exact_instance",
            "teacher_negative_scope": "full_100_photo_category_gallery",
            "teacher_objective": (
                "matched_control_staged_visual_text_semantic_visual_refinement"
            ),
            "teacher_auxiliary_objective": (
                "sketch_image_to_photo_text_and_sketch_text_to_photo_image"
            ),
            "scheduler_monitor": "best_unseen_acc1_then_acc5",
            "teacher_instance_temperature": (
                args.teacher_instance_temperature
            ),
            "teacher_n_ctx_text": args.teacher_n_ctx_text,
            "text_prompt_gate_init": args.text_prompt_gate_init,
            "teacher_text_prompt_seed": args.teacher_text_prompt_seed,
            "teacher_text_prompt_lr": args.teacher_text_prompt_lr,
            "teacher_text_prompt_weight_decay": (
                args.teacher_text_prompt_weight_decay
            ),
            "teacher_prompt_infonce_temperature": (
                args.teacher_prompt_infonce_temperature
            ),
            "lambda_teacher_prompt_infonce": (
                args.lambda_teacher_prompt_infonce
            ),
            "teacher_text_pretrain_epochs": args.teacher_text_pretrain_epochs,
            "teacher_semantic_refine_epochs": (
                args.teacher_semantic_refine_epochs
            ),
            "teacher_semantic_refine_lr": args.teacher_semantic_refine_lr,
            "teacher_semantic_warmup_epochs": (
                args.teacher_semantic_warmup_epochs
            ),
            "lambda_teacher_semantic_refine": (
                args.lambda_teacher_semantic_refine
            ),
            "lambda_teacher_visual_keep": args.lambda_teacher_visual_keep,
            "lambda_teacher_text_anchor": args.lambda_teacher_text_anchor,
            "teacher_training_schedule": (
                "visual_bootstrap_then_text_bootstrap_then_matched_control_"
                "and_semantic_visual_refine"
            ),
            "matched_phase_c_control": True,
            "checkpoint_selection": (
                "best_unseen_acc1_then_acc5_across_phase_a_control_semantic"
            ),
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
    def __init__(self, cfg, clip_model, classnames, teacher=None):
        super().__init__(cfg, clip_model, classnames, teacher)
        student_visual_width = clip_model.visual.ln_pre.normalized_shape[0]
        self.student_text_prompt_learner = ImageConditionedTextPromptLearner(
            text_model=clip_model,
            tokenizer=clip.tokenize,
            classnames=self.classnames,
            visual_width=student_visual_width,
            context_tokens=cfg.student_n_ctx_text,
            seed=cfg.text_prompt_seed,
            text_backend="openai",
            gate_init=cfg.text_prompt_gate_init,
            encode_chunk_size=cfg.text_prompt_encode_chunk_size,
            gradient_checkpointing=cfg.text_prompt_gradient_checkpointing,
        )

        self.teacher_text_prompt_learner = None
        if teacher is not None and cfg.teacher_pretrain_epochs > 0:
            teacher_visual_width = teacher.visual.conv1.out_channels
            self.teacher_text_prompt_learner = (
                ImageConditionedTextPromptLearner(
                    text_model=teacher,
                    tokenizer=teacher.text_tokenizer,
                    classnames=self.classnames,
                    visual_width=teacher_visual_width,
                    context_tokens=cfg.teacher_n_ctx_text,
                    seed=cfg.teacher_text_prompt_seed,
                    text_backend="open_clip",
                    gate_init=cfg.text_prompt_gate_init,
                    encode_chunk_size=cfg.teacher_text_prompt_encode_chunk_size,
                    gradient_checkpointing=(
                        cfg.text_prompt_gradient_checkpointing
                    ),
                ).to(teacher.visual.conv1.weight.device)
            )

        print(
            "[Image-Conditioned Text] independent teacher/student counts; "
            f"student_n_ctx_text={cfg.student_n_ctx_text}; "
            f"teacher_n_ctx_text={cfg.teacher_n_ctx_text}; "
            "patch_projection=True; "
            "modalities=photo+sketch; student_params="
            f"{self.student_text_prompt_learner.trainable_parameter_count():,}; "
            f"teacher_params={self._teacher_text_prompt_parameter_count():,}"
        )

    def _teacher_text_prompt_parameter_count(self):
        if self.teacher_text_prompt_learner is None:
            return 0
        return self.teacher_text_prompt_learner.trainable_parameter_count()

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
            return F.normalize(features.float(), dim=-1), patch_tokens
        return F.normalize(output.float(), dim=-1)

    def forward(self, x):
        (
            photo_tensor,
            sketch_tensor,
            teacher_photo_base,
            teacher_sketch_base,
            categories,
        ) = x
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
        student_photo_prompt_text, _ = self.student_text_prompt_learner(
            self.clip_model,
            photo_patches,
            photo_categories,
            "photo",
        )
        student_sketch_prompt_text, _ = self.student_text_prompt_learner(
            self.clip_model,
            sketch_patches,
            categories,
            "sketch",
        )

        # Fixed class text banks are retained only for the baseline modality
        # KD objective. Exact-instance prompt InfoNCE uses conditioned text.
        student_photo_text = F.normalize(
            self.get_student_text_features("photo"),
            dim=-1,
        )
        student_sketch_text = F.normalize(
            self.get_student_text_features("sketch"),
            dim=-1,
        )
        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            teacher_photo_features = teacher_photo_base
            teacher_sketch_features = teacher_sketch_base
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
            student_photo_prompt_text,
            student_sketch_prompt_text,
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
        if self.teacher_text_prompt_learner is None:
            raise RuntimeError(
                "Teacher pretraining requires its image-conditioned text "
                "prompt learner."
            )

        cfg = self.cfg
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
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
        scaler = torch.amp.GradScaler(
            "cuda", enabled=teacher_device.type == "cuda"
        )

        def clone_state(module):
            return {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }

        def move_batch(batch):
            photo, sketch, _, _, categories, targets = batch
            return (
                photo.to(
                    teacher_device, dtype=teacher_dtype, non_blocking=True
                ),
                sketch.to(
                    teacher_device, dtype=teacher_dtype, non_blocking=True
                ),
                categories.to(teacher_device, non_blocking=True),
                targets.to(teacher_device, non_blocking=True),
            )

        def evaluate_visual(phase, epoch):
            train_acc1, train_acc5 = self._validate_teacher_train(
                train_dataset,
                epoch,
                workers,
                show_progress,
            )
            val_acc1, val_acc5 = self._validate_teacher_unseen(
                val_sketch_loader,
                val_photo_loader,
                epoch,
                show_progress,
            )
            self.teacher_train_metric_history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "acc1": train_acc1,
                    "acc5": train_acc5,
                }
            )
            self.teacher_unseen_metric_history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "acc1": val_acc1,
                    "acc5": val_acc5,
                }
            )
            return val_acc1, val_acc5

        self.teacher_train_metric_history = []
        self.teacher_unseen_metric_history = []
        self.teacher_text_metric_history = []

        # The initial state is a real checkpoint candidate. A failed text
        # experiment must never force the final teacher below this state.
        self.teacher_prompts.eval().requires_grad_(False)
        self.teacher_text_prompt_learner.eval().requires_grad_(False)
        best_acc1, best_acc5 = evaluate_visual("initial", 0)
        best_epoch = 0
        best_phase = "initial"
        best_prompt_state = clone_state(self.teacher_prompts)

        # Phase A: obtain the strongest visual-only teacher before allowing
        # randomly initialized text prompts to influence visual retrieval.
        print(
            "[Teacher Phase A] visual bootstrap; "
            f"epochs={cfg.teacher_pretrain_epochs}"
        )
        self.teacher_prompts.train().requires_grad_(True)
        self.teacher_text_prompt_learner.eval().requires_grad_(False)
        visual_optimizer = torch.optim.SGD(
            self.teacher_prompts.parameters(),
            lr=cfg.teacher_prompt_lr,
            momentum=cfg.teacher_momentum,
            weight_decay=cfg.teacher_weight_decay,
        )
        visual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            visual_optimizer,
            mode="max",
            factor=cfg.teacher_scheduler_gamma,
            patience=_reduce_on_plateau_patience(
                cfg.teacher_scheduler_patience
            ),
            threshold=0.0,
            threshold_mode="abs",
        )
        for epoch in range(cfg.teacher_pretrain_epochs):
            retrieval_total = 0.0
            steps = 0
            batches = tqdm(
                loader,
                desc=(
                    "Teacher visual bootstrap "
                    f"{epoch + 1}/{cfg.teacher_pretrain_epochs}"
                ),
                disable=not show_progress,
            )
            for batch in batches:
                photo, sketch, _, targets = move_batch(batch)
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=teacher_device.type == "cuda",
                ):
                    photo_features = self._encode_teacher_image(photo, "photo")
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
                visual_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(visual_optimizer)
                scaler.update()
                retrieval_total += retrieval.detach().item()
                steps += 1
                if show_progress:
                    batches.set_postfix(T_VIS=f"{retrieval.item():.3f}")
            if steps == 0:
                raise RuntimeError("Teacher visual bootstrap produced no batches.")
            print(
                f"[Teacher Phase A] epoch={epoch + 1}, "
                f"visual_instance_nce={retrieval_total / steps:.6f}"
            )
            metric_epoch = epoch + 1
            acc1, acc5 = evaluate_visual("visual_bootstrap", metric_epoch)
            visual_scheduler.step(acc1 + acc5 * 1e-6)
            if better_acc1_acc5(acc1, acc5, best_acc1, best_acc5):
                best_acc1, best_acc5 = acc1, acc5
                best_epoch = metric_epoch
                best_phase = "visual_bootstrap"
                best_prompt_state = clone_state(self.teacher_prompts)

        self.teacher_prompts.load_state_dict(best_prompt_state, strict=True)
        visual_source_state = clone_state(self.teacher_prompts)
        visual_source_acc1, visual_source_acc5 = best_acc1, best_acc5
        visual_source_phase = best_phase
        visual_source_epoch = best_epoch
        self.teacher_phase_a_acc1 = visual_source_acc1
        self.teacher_phase_a_acc5 = visual_source_acc5
        self.teacher_phase_a_epoch = visual_source_epoch
        print(
            "[Teacher Phase A Best] "
            f"epoch={best_epoch}, Acc@1={best_acc1:.4f}, "
            f"Acc@5={best_acc5:.4f}"
        )

        # Phase B: train only the text prompt learner. Detaching teacher
        # image/patch features prevents the semantic branch from corrupting
        # the visual teacher while it is still learning its language bridge.
        print(
            "[Teacher Phase B] frozen-visual text bootstrap; "
            f"epochs={cfg.teacher_text_pretrain_epochs}"
        )
        self.teacher_prompts.eval().requires_grad_(False)
        self.teacher_text_prompt_learner.train().requires_grad_(True)
        text_optimizer = torch.optim.AdamW(
            self.teacher_text_prompt_learner.parameters(),
            lr=cfg.teacher_text_prompt_lr,
            weight_decay=cfg.teacher_text_prompt_weight_decay,
        )
        teacher_sketch_anchors, teacher_photo_anchors = (
            self.get_teacher_text_features()
        )
        best_text_loss = float("inf")
        best_text_prompt_state = clone_state(self.teacher_text_prompt_learner)
        for epoch in range(cfg.teacher_text_pretrain_epochs):
            prompt_total = 0.0
            anchor_total = 0.0
            metric_totals = {
                "sketch_to_photo_text_acc1": 0.0,
                "sketch_to_photo_text_acc5": 0.0,
                "sketch_text_to_photo_acc1": 0.0,
                "sketch_text_to_photo_acc5": 0.0,
            }
            query_count = 0
            steps = 0
            batches = tqdm(
                loader,
                desc=(
                    "Teacher text bootstrap "
                    f"{epoch + 1}/{cfg.teacher_text_pretrain_epochs}"
                ),
                disable=not show_progress,
            )
            for batch in batches:
                photo, sketch, categories, targets = move_batch(batch)
                photo_categories = categories[:1].expand(len(photo))
                with torch.no_grad(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=teacher_device.type == "cuda",
                ):
                    photo_features, photo_patches = self._encode_teacher_image(
                        photo, "photo", return_patch_tokens=True
                    )
                    sketch_features, sketch_patches = self._encode_teacher_image(
                        sketch, "sketch", return_patch_tokens=True
                    )
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=teacher_device.type == "cuda",
                ):
                    photo_prompt_text, _ = self.teacher_text_prompt_learner(
                        self._teacher,
                        photo_patches.detach(),
                        photo_categories,
                        "photo",
                    )
                    sketch_prompt_text, _ = self.teacher_text_prompt_learner(
                        self._teacher,
                        sketch_patches.detach(),
                        categories,
                        "sketch",
                    )
                    prompt_infonce, _ = fine_grained_prompt_infonce_loss(
                        sketch_features.detach(),
                        photo_features.detach(),
                        sketch_prompt_text,
                        photo_prompt_text,
                        targets,
                        cfg.teacher_prompt_infonce_temperature,
                    )
                    anchor = image_conditioned_text_anchor_loss(
                        sketch_prompt_text,
                        photo_prompt_text,
                        teacher_sketch_anchors[categories],
                        teacher_photo_anchors[photo_categories],
                    )
                    loss = (
                        cfg.lambda_teacher_prompt_infonce * prompt_infonce
                        + cfg.lambda_teacher_text_anchor * anchor
                    )
                prompt_metrics = fine_grained_prompt_retrieval_accuracy(
                    sketch_features,
                    photo_features,
                    sketch_prompt_text,
                    photo_prompt_text,
                    targets,
                )
                text_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(text_optimizer)
                scaler.update()
                prompt_total += prompt_infonce.detach().item()
                anchor_total += anchor.detach().item()
                current_queries = len(targets)
                for name, value in prompt_metrics.items():
                    metric_totals[name] += value.item() * current_queries
                query_count += current_queries
                steps += 1
                if show_progress:
                    batches.set_postfix(
                        T_PROMPT=f"{prompt_infonce.item():.3f}",
                        T_ANCHOR=f"{anchor.item():.3f}",
                    )
            if steps == 0:
                raise RuntimeError("Teacher text bootstrap produced no batches.")
            average_prompt = prompt_total / steps
            average_anchor = anchor_total / steps
            average_total = (
                cfg.lambda_teacher_prompt_infonce * average_prompt
                + cfg.lambda_teacher_text_anchor * average_anchor
            )
            average_metrics = {
                name: total / query_count
                for name, total in metric_totals.items()
            }
            self.teacher_text_metric_history.append(
                {
                    "epoch": epoch + 1,
                    "prompt_infonce": average_prompt,
                    "anchor": average_anchor,
                    "total": average_total,
                    **average_metrics,
                }
            )
            print(
                f"[Teacher Phase B] epoch={epoch + 1}, "
                f"prompt_infonce={average_prompt:.6f}, "
                f"semantic_anchor={average_anchor:.6f}, "
                "image_to_text_Acc@1="
                f"{average_metrics['sketch_to_photo_text_acc1']:.4f}, "
                "image_to_text_Acc@5="
                f"{average_metrics['sketch_to_photo_text_acc5']:.4f}, "
                "text_to_image_Acc@1="
                f"{average_metrics['sketch_text_to_photo_acc1']:.4f}, "
                "text_to_image_Acc@5="
                f"{average_metrics['sketch_text_to_photo_acc5']:.4f}"
            )
            if average_total < best_text_loss:
                best_text_loss = average_total
                best_text_prompt_state = clone_state(
                    self.teacher_text_prompt_learner
                )

        self.teacher_text_prompt_learner.load_state_dict(
            best_text_prompt_state, strict=True
        )
        self.teacher_text_prompt_learner.eval().requires_grad_(False)

        # Both Phase-C tracks start from the exact same Phase-A checkpoint and
        # consume the exact same deterministic batch order. The matched
        # visual-only continuation is required to distinguish a genuine text
        # contribution from the effect of simply training visual prompts for
        # more epochs.
        semantic_source = build_teacher_prompt_controller(
            teacher=self._teacher,
            n_ctx=cfg.teacher_n_ctx_visual,
            depth=cfg.teacher_prompt_depth,
            std=cfg.teacher_prompt_std,
            seed=cfg.teacher_prompt_seed,
        ).to(teacher_device)
        semantic_source.load_state_dict(visual_source_state, strict=True)
        semantic_source.eval().requires_grad_(False)

        refinement_sampler_epoch = sampler.epoch

        def run_visual_refinement(phase, use_semantic, metric_epoch_offset):
            # Reset both weights and sampler position so the semantic and
            # control tracks differ in exactly one term of the objective.
            sampler.epoch = refinement_sampler_epoch
            self.teacher_prompts.load_state_dict(
                visual_source_state, strict=True
            )
            self.teacher_prompts.train().requires_grad_(True)
            optimizer = torch.optim.SGD(
                self.teacher_prompts.parameters(),
                lr=cfg.teacher_semantic_refine_lr,
                momentum=cfg.teacher_momentum,
                weight_decay=cfg.teacher_weight_decay,
            )
            track_scaler = torch.amp.GradScaler(
                "cuda", enabled=teacher_device.type == "cuda"
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
            track_acc1, track_acc5 = visual_source_acc1, visual_source_acc5
            track_state = visual_source_state
            track_phase = visual_source_phase
            track_epoch = visual_source_epoch

            label = "semantic" if use_semantic else "matched control"
            print(
                f"[Teacher Phase C/{label}] "
                f"epochs={cfg.teacher_semantic_refine_epochs}; "
                f"same_start=True; same_batches=True"
            )
            for epoch in range(cfg.teacher_semantic_refine_epochs):
                semantic_weight = (
                    cfg.lambda_teacher_semantic_refine
                    * min(
                        1.0,
                        (epoch + 1) / cfg.teacher_semantic_warmup_epochs,
                    )
                    if use_semantic
                    else 0.0
                )
                totals = {"retrieval": 0.0, "keep": 0.0}
                if use_semantic:
                    totals["semantic"] = 0.0
                steps = 0
                batches = tqdm(
                    loader,
                    desc=(
                        f"Teacher {label} refine "
                        f"{epoch + 1}/{cfg.teacher_semantic_refine_epochs}"
                    ),
                    disable=not show_progress,
                )
                for batch in batches:
                    photo, sketch, categories, targets = move_batch(batch)
                    photo_categories = categories[:1].expand(len(photo))
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda",
                        dtype=torch.float16,
                        enabled=teacher_device.type == "cuda",
                    ):
                        if use_semantic:
                            source_photo, source_photo_patches = semantic_source(
                                photo, "photo", return_patch_tokens=True
                            )
                            source_sketch, source_sketch_patches = semantic_source(
                                sketch, "sketch", return_patch_tokens=True
                            )
                            fixed_photo_text, _ = (
                                self.teacher_text_prompt_learner(
                                    self._teacher,
                                    source_photo_patches,
                                    photo_categories,
                                    "photo",
                                )
                            )
                            fixed_sketch_text, _ = (
                                self.teacher_text_prompt_learner(
                                    self._teacher,
                                    source_sketch_patches,
                                    categories,
                                    "sketch",
                                )
                            )
                        else:
                            source_photo = semantic_source(photo, "photo")
                            source_sketch = semantic_source(sketch, "sketch")
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
                        if use_semantic:
                            loss, parts = teacher_semantic_refinement_loss(
                                sketch_features,
                                photo_features,
                                source_sketch,
                                source_photo,
                                fixed_sketch_text,
                                fixed_photo_text,
                                targets,
                                cfg.teacher_instance_temperature,
                                cfg.teacher_prompt_infonce_temperature,
                                cfg.lambda_teacher_retrieval,
                                semantic_weight,
                                cfg.lambda_teacher_visual_keep,
                            )
                        else:
                            loss, parts = (
                                teacher_visual_refinement_control_loss(
                                    sketch_features,
                                    photo_features,
                                    source_sketch,
                                    source_photo,
                                    targets,
                                    cfg.teacher_instance_temperature,
                                    cfg.lambda_teacher_retrieval,
                                    cfg.lambda_teacher_visual_keep,
                                )
                            )
                    optimizer.zero_grad(set_to_none=True)
                    track_scaler.scale(loss).backward()
                    track_scaler.step(optimizer)
                    track_scaler.update()
                    for name in totals:
                        totals[name] += parts[name].detach().item()
                    steps += 1
                    if show_progress:
                        postfix = {
                            "T_VIS": f"{parts['retrieval'].item():.3f}",
                            "T_KEEP": f"{parts['keep'].item():.3f}",
                        }
                        if use_semantic:
                            postfix["T_SEM"] = (
                                f"{parts['semantic'].item():.3f}"
                            )
                        batches.set_postfix(**postfix)
                if steps == 0:
                    raise RuntimeError(
                        f"Teacher {label} refinement produced no batches."
                    )
                message = (
                    f"[Teacher Phase C/{label}] epoch={epoch + 1}, "
                    f"visual_instance_nce="
                    f"{totals['retrieval'] / steps:.6f}, "
                    f"visual_keep={totals['keep'] / steps:.6f}"
                )
                if use_semantic:
                    message += (
                        f", semantic_infonce="
                        f"{totals['semantic'] / steps:.6f}, "
                        f"semantic_weight={semantic_weight:.6f}"
                    )
                print(message)
                metric_epoch = metric_epoch_offset + epoch + 1
                acc1, acc5 = evaluate_visual(phase, metric_epoch)
                scheduler.step(acc1 + acc5 * 1e-6)
                if better_acc1_acc5(
                    acc1, acc5, track_acc1, track_acc5
                ):
                    track_acc1, track_acc5 = acc1, acc5
                    track_epoch = metric_epoch
                    track_phase = phase
                    track_state = clone_state(self.teacher_prompts)

            return (
                track_state,
                track_acc1,
                track_acc5,
                track_epoch,
                track_phase,
            )

        control_result = run_visual_refinement(
            "matched_control",
            use_semantic=False,
            metric_epoch_offset=cfg.teacher_pretrain_epochs,
        )
        (
            control_state,
            control_acc1,
            control_acc5,
            control_epoch,
            control_phase,
        ) = control_result
        print(
            "[Teacher Matched Control Best] "
            f"phase={control_phase}, epoch={control_epoch}, "
            f"Acc@1={control_acc1:.4f}, Acc@5={control_acc5:.4f}"
        )

        semantic_result = run_visual_refinement(
            "semantic_refine",
            use_semantic=True,
            metric_epoch_offset=(
                cfg.teacher_pretrain_epochs
                + cfg.teacher_semantic_refine_epochs
            ),
        )
        (
            semantic_state,
            semantic_acc1,
            semantic_acc5,
            semantic_epoch,
            semantic_phase,
        ) = semantic_result
        print(
            "[Teacher Semantic Best] "
            f"phase={semantic_phase}, epoch={semantic_epoch}, "
            f"Acc@1={semantic_acc1:.4f}, Acc@5={semantic_acc5:.4f}"
        )

        best_prompt_state = control_state
        best_acc1, best_acc5 = control_acc1, control_acc5
        best_epoch, best_phase = control_epoch, control_phase
        if better_acc1_acc5(
            semantic_acc1,
            semantic_acc5,
            control_acc1,
            control_acc5,
        ):
            best_prompt_state = semantic_state
            best_acc1, best_acc5 = semantic_acc1, semantic_acc5
            best_epoch, best_phase = semantic_epoch, semantic_phase

        self.teacher_prompts.load_state_dict(best_prompt_state, strict=True)
        self.teacher_best_epoch = best_epoch
        self.teacher_best_phase = best_phase
        self.teacher_best_acc1 = best_acc1
        self.teacher_best_acc5 = best_acc5
        self.teacher_control_best_epoch = control_epoch
        self.teacher_control_best_phase = control_phase
        self.teacher_control_acc1 = control_acc1
        self.teacher_control_acc5 = control_acc5
        self.teacher_semantic_best_epoch = semantic_epoch
        self.teacher_semantic_best_phase = semantic_phase
        self.teacher_semantic_acc1 = semantic_acc1
        self.teacher_semantic_acc5 = semantic_acc5
        self.teacher_control_gain_acc1 = control_acc1 - visual_source_acc1
        self.teacher_control_gain_acc5 = control_acc5 - visual_source_acc5
        self.teacher_semantic_gain_acc1 = semantic_acc1 - visual_source_acc1
        self.teacher_semantic_gain_acc5 = semantic_acc5 - visual_source_acc5
        self.teacher_text_added_value_acc1 = semantic_acc1 - control_acc1
        self.teacher_text_added_value_acc5 = semantic_acc5 - control_acc5
        self.teacher_best_gain_acc1 = best_acc1 - visual_source_acc1
        self.teacher_best_gain_acc5 = best_acc5 - visual_source_acc5
        print(
            "[Teacher Best] restored visual prompts from "
            f"phase={best_phase}, epoch={best_epoch}, "
            f"Acc@1={best_acc1:.4f}, Acc@5={best_acc5:.4f}"
        )
        print(
            "[Teacher Semantic Gain] versus Phase-A best: "
            f"delta_Acc@1={self.teacher_semantic_gain_acc1:+.4f}, "
            f"delta_Acc@5={self.teacher_semantic_gain_acc5:+.4f}"
        )
        print(
            "[Teacher Text Added Value] versus matched visual-only control: "
            f"delta_Acc@1={self.teacher_text_added_value_acc1:+.4f}, "
            f"delta_Acc@5={self.teacher_text_added_value_acc5:+.4f}"
        )
        print(
            "[Teacher Best Gain] versus Phase-A best: "
            f"delta_Acc@1={self.teacher_best_gain_acc1:+.4f}, "
            f"delta_Acc@5={self.teacher_best_gain_acc5:+.4f}"
        )
        self.teacher_prompts.eval().requires_grad_(False)
        self.teacher_text_prompt_learner.eval().requires_grad_(False)
        del semantic_source

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
            "teacher_text_prompt_state_dict": (
                {
                    key: value.detach().cpu()
                    for key, value in (
                        self.teacher_text_prompt_learner.state_dict().items()
                    )
                }
                if self.teacher_text_prompt_learner is not None
                else None
            ),
            "teacher_best_epoch": getattr(self, "teacher_best_epoch", None),
            "teacher_best_phase": getattr(self, "teacher_best_phase", None),
            "teacher_phase_a_epoch": getattr(
                self, "teacher_phase_a_epoch", None
            ),
            "teacher_phase_a_acc1": getattr(
                self, "teacher_phase_a_acc1", None
            ),
            "teacher_phase_a_acc5": getattr(
                self, "teacher_phase_a_acc5", None
            ),
            "teacher_best_acc1": getattr(self, "teacher_best_acc1", None),
            "teacher_best_acc5": getattr(self, "teacher_best_acc5", None),
            "teacher_control_best_epoch": getattr(
                self, "teacher_control_best_epoch", None
            ),
            "teacher_control_best_phase": getattr(
                self, "teacher_control_best_phase", None
            ),
            "teacher_control_acc1": getattr(
                self, "teacher_control_acc1", None
            ),
            "teacher_control_acc5": getattr(
                self, "teacher_control_acc5", None
            ),
            "teacher_semantic_best_epoch": getattr(
                self, "teacher_semantic_best_epoch", None
            ),
            "teacher_semantic_best_phase": getattr(
                self, "teacher_semantic_best_phase", None
            ),
            "teacher_semantic_acc1": getattr(
                self, "teacher_semantic_acc1", None
            ),
            "teacher_semantic_acc5": getattr(
                self, "teacher_semantic_acc5", None
            ),
            "teacher_control_gain_acc1": getattr(
                self, "teacher_control_gain_acc1", None
            ),
            "teacher_control_gain_acc5": getattr(
                self, "teacher_control_gain_acc5", None
            ),
            "teacher_semantic_gain_acc1": getattr(
                self, "teacher_semantic_gain_acc1", None
            ),
            "teacher_semantic_gain_acc5": getattr(
                self, "teacher_semantic_gain_acc5", None
            ),
            "teacher_text_added_value_acc1": getattr(
                self, "teacher_text_added_value_acc1", None
            ),
            "teacher_text_added_value_acc5": getattr(
                self, "teacher_text_added_value_acc5", None
            ),
            "teacher_best_gain_acc1": getattr(
                self, "teacher_best_gain_acc1", None
            ),
            "teacher_best_gain_acc5": getattr(
                self, "teacher_best_gain_acc5", None
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
            "teacher_text_metric_history": getattr(
                self,
                "teacher_text_metric_history",
                [],
            ),
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
        report_path = write_teacher_refinement_report(
            cache_path,
            payload["metadata"],
            {
                "best_phase": getattr(self, "teacher_best_phase", None),
                "best_epoch": getattr(self, "teacher_best_epoch", None),
                "phase_a_epoch": getattr(self, "teacher_phase_a_epoch", None),
                "phase_a_acc1": getattr(self, "teacher_phase_a_acc1", None),
                "phase_a_acc5": getattr(self, "teacher_phase_a_acc5", None),
                "control_phase": getattr(
                    self, "teacher_control_best_phase", None
                ),
                "control_epoch": getattr(
                    self, "teacher_control_best_epoch", None
                ),
                "control_acc1": getattr(self, "teacher_control_acc1", None),
                "control_acc5": getattr(self, "teacher_control_acc5", None),
                "semantic_phase": getattr(
                    self, "teacher_semantic_best_phase", None
                ),
                "semantic_epoch": getattr(
                    self, "teacher_semantic_best_epoch", None
                ),
                "semantic_acc1": getattr(self, "teacher_semantic_acc1", None),
                "semantic_acc5": getattr(self, "teacher_semantic_acc5", None),
                "best_acc1": getattr(self, "teacher_best_acc1", None),
                "best_acc5": getattr(self, "teacher_best_acc5", None),
                "control_delta_vs_phase_a_acc1": getattr(
                    self, "teacher_control_gain_acc1", None
                ),
                "control_delta_vs_phase_a_acc5": getattr(
                    self, "teacher_control_gain_acc5", None
                ),
                "semantic_delta_vs_phase_a_acc1": getattr(
                    self, "teacher_semantic_gain_acc1", None
                ),
                "semantic_delta_vs_phase_a_acc5": getattr(
                    self, "teacher_semantic_gain_acc5", None
                ),
                "text_added_value_acc1": getattr(
                    self, "teacher_text_added_value_acc1", None
                ),
                "text_added_value_acc5": getattr(
                    self, "teacher_text_added_value_acc5", None
                ),
                "best_delta_vs_phase_a_acc1": getattr(
                    self, "teacher_best_gain_acc1", None
                ),
                "best_delta_vs_phase_a_acc5": getattr(
                    self, "teacher_best_gain_acc5", None
                ),
            },
            {
                "visual": payload["teacher_unseen_metric_history"],
                "text": payload["teacher_text_metric_history"],
            },
        )
        print(
            f"[Teacher Cache] saved {cache_path} "
            f"({cache_path.stat().st_size / 1024**2:.1f} MB)."
        )
        print(f"[Teacher Report] saved {report_path}")

    def cache_teacher_features(
        self,
        train_dataset,
        val_sketch_loader,
        val_photo_loader,
        batch_size,
        workers,
        show_progress,
    ):
        super().cache_teacher_features(
            train_dataset,
            val_sketch_loader,
            val_photo_loader,
            batch_size,
            workers,
            show_progress,
        )
        # Teacher text prompts have already shaped the cached teacher visual
        # features. Do not serialize them in every student checkpoint.
        self.teacher_text_prompt_learner = None


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
        text_parameters = list(
            self.model.student_text_prompt_learner.parameters()
        )
        text_ids = {id(parameter) for parameter in text_parameters}
        visual_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in text_ids
        ]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": visual_parameters,
                    "lr": self.args.lr,
                    "weight_decay": self.args.weight_decay,
                    "name": "visual_prompts",
                },
                {
                    "params": text_parameters,
                    "lr": self.args.text_prompt_lr,
                    "weight_decay": self.args.text_prompt_weight_decay,
                    "name": "image_conditioned_text_prompts",
                },
            ],
            lr=self.args.lr,
            momentum=self.args.momentum,
        )
        visual_trainable = sum(
            parameter.numel() for parameter in visual_parameters
        )
        text_trainable = sum(
            parameter.numel() for parameter in text_parameters
        )
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum={self.args.momentum}, "
            f"weight_decay={self.args.weight_decay}, "
            f"visual_params={visual_trainable:,}, "
            f"text_prompt_lr={self.args.text_prompt_lr}, "
            f"text_prompt_weight_decay={self.args.text_prompt_weight_decay}, "
            f"text_prompt_params={text_trainable:,}"
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
            self.args,
            features[:9],
        )
        student_photo_prompt_text, student_sketch_prompt_text = features[9:]
        visual_infonce = fine_grained_teacher_infonce_loss(
            features[1],
            features[0],
            targets,
            self.args.student_instance_temperature,
        )
        prompt_infonce, prompt_parts = fine_grained_prompt_infonce_loss(
            features[1],
            features[0],
            student_sketch_prompt_text,
            student_photo_prompt_text,
            targets,
            self.args.prompt_infonce_temperature,
        )
        loss = (
            loss
            + self.args.lambda_student_retrieval * visual_infonce
            + self.args.lambda_prompt_infonce * prompt_infonce
        )
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
            "VIS_NCE",
            visual_infonce,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "PROMPT_NCE",
            prompt_infonce,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "SKETCH_TO_PHOTO_TEXT_NCE",
            prompt_parts["sketch_to_photo_text"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "SKETCH_TEXT_TO_PHOTO_NCE",
            prompt_parts["sketch_text_to_photo"],
            on_step=False,
            on_epoch=True,
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
