import hashlib
import math
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
)
import open_clip
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip import clip
from clip.model import build_model
from src.dataset import (
    TeacherFeatureDataset,
    WorkerInvariantSampler,
)
from src.losses import (
    batch_hard_teacher_triplet_loss,
    loss_fn,
)
from src.teacher_prompts import build_teacher_prompt_controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DFN5B teacher loader
# ---------------------------------------------------------------------------
DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
DFN5B_OUTPUT_DIM = 1024
TEACHER_CACHE_FORMAT_VERSION = 6


def _retrieval_metrics(
    query_features,
    gallery_features,
    query_labels,
    gallery_labels,
    dataset,
):
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    ap = torch.zeros(len(query_features))
    precision_at_k = torch.zeros(len(query_features))
    if dataset == "sketchy_2":
        map_k = 200
        p_k = 200
    else:
        map_k = 0
        p_k = 200 if dataset == "quickdraw" else 100

    for index, query_feature in enumerate(query_features):
        cosine = F.cosine_similarity(
            query_feature.unsqueeze(0), gallery_features
        ).cpu()
        score = ((cosine + 1.0) * 0.5).clamp(
            min=torch.finfo(cosine.dtype).eps,
            max=1.0,
        )
        target = gallery_labels.eq(query_labels[index])
        if map_k:
            ap[index] = retrieval_average_precision(
                score,
                target,
                top_k=min(map_k, len(gallery_features)),
            )
        else:
            ap[index] = retrieval_average_precision(score, target)
        precision_at_k[index] = retrieval_precision(
            score,
            target,
            top_k=p_k,
        )

    return ap.mean(), precision_at_k.mean(), map_k, p_k


def _teacher_training_config(args):
    """Parameters that can change the prompt-tuned teacher targets."""
    return {
        "teacher_model": DFN5B_MODEL,
        "teacher_pretrained": DFN5B_PRETRAINED,
        "teacher_output_dim": DFN5B_OUTPUT_DIM,
        "teacher_precision": "fp16",
        "teacher_n_ctx_visual": args.teacher_n_ctx_visual,
        "teacher_prompt_depth": args.teacher_prompt_depth,
        "teacher_prompt_std": args.teacher_prompt_std,
        "teacher_prompt_seed": args.teacher_prompt_seed,
        "teacher_prompt_gradient_checkpointing": (
            args.teacher_prompt_gradient_checkpointing
        ),
        "teacher_prompt_lr": args.teacher_prompt_lr,
        "teacher_momentum": args.teacher_momentum,
        "teacher_weight_decay": args.teacher_weight_decay,
        "pretrain_epochs": args.teacher_pretrain_epochs,
        "pretrain_batch_size": args.teacher_pretrain_batch_size,
        "lambda_retrieval": args.lambda_teacher_retrieval,
        "triplet_margin": args.teacher_triplet_margin,
        "scheduler": "StepLR",
        "scheduler_step_size": args.teacher_scheduler_step_size,
        "scheduler_gamma": args.teacher_scheduler_gamma,
        "checkpoint_selection": "best_unseen_precision",
        "seed": args.seed,
    }


def default_teacher_cache_path(args, train_dataset):
    """Build a reusable cache path from the dataset and teacher configuration."""
    dataset_digest = hashlib.sha256()
    dataset_digest.update(args.dataset.encode("utf-8"))
    dataset_digest.update(str(train_dataset.max_size).encode("utf-8"))
    for classname in train_dataset.all_categories:
        dataset_digest.update(classname.encode("utf-8"))
        dataset_digest.update(b"\0")
    paths = train_dataset.all_sketches_path + train_dataset.all_photo_paths
    for path in paths:
        relative = os.path.relpath(path, args.root).replace("\\", "/")
        dataset_digest.update(relative.encode("utf-8"))
        dataset_digest.update(b"\0")

    cache_key = {
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
        "dataset": args.dataset,
        "dataset_fingerprint": dataset_digest.hexdigest(),
        "teacher": _teacher_training_config(args),
    }
    encoded = json.dumps(
        cache_key,
        sort_keys=True,
        separators=(",", ":"),
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
    return str(Path(cache_dir) / f"{args.dataset}_{config_hash}.pt")


def _image_text_kd_active(args):
    return args.lambda_modality > 0


def _persistent_teacher_cache_available(args):
    return (
        bool(args.teacher_cache_path)
        and not args.rebuild_teacher_cache
        and Path(args.teacher_cache_path).is_file()
    )


def _load_clip_model(backbone):
    model_path = clip.download_model(backbone)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return build_model(state_dict)


def _build_teacher_prompts(args, teacher):
    if teacher is None or args.teacher_pretrain_epochs == 0:
        return None

    controller = build_teacher_prompt_controller(
        teacher=teacher,
        n_ctx=args.teacher_n_ctx_visual,
        depth=args.teacher_prompt_depth,
        std=args.teacher_prompt_std,
        seed=args.teacher_prompt_seed,
    )
    print(
        "[Teacher Prompt] initialized for teacher pretraining "
        f"(n_ctx_visual={args.teacher_n_ctx_visual}, "
        f"depth={controller.depth}, std={args.teacher_prompt_std}, "
        f"trainable_params={controller.trainable_parameter_count():,})"
    )
    return controller


def _load_teacher(args):
    if _persistent_teacher_cache_available(args):
        print(
            "[Teacher Cache] persistent cache found; "
            "skipping DFN5B loading."
        )
        return None

    if (
        args.lambda_domain <= 0
        and not _image_text_kd_active(args)
        and args.teacher_pretrain_epochs == 0
    ):
        return None

    print(f"[Teacher] Loading {DFN5B_MODEL} in FP16...")
    teacher = open_clip.create_model(
        DFN5B_MODEL,
        pretrained=DFN5B_PRETRAINED,
        precision="fp16",
        device=device,
    )
    teacher.eval().requires_grad_(False)
    if args.teacher_pretrain_epochs > 0 or _image_text_kd_active(args):
        teacher.text_tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    teacher.output_dim = DFN5B_OUTPUT_DIM
    return teacher


def freeze_clip(clip_model):
    clip_model.requires_grad_(False)


def _random_parameter(rows, width, seed):
    if rows == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameter = torch.empty(rows, width)
    nn.init.normal_(parameter, std=0.02, generator=generator)
    return nn.Parameter(parameter)


class IndependentVisualPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx_visual,
        visual_width,
        seed,
        prompt_depth,
    ):
        super().__init__()
        self.ctx = _random_parameter(n_ctx_visual, visual_width, seed)
        self.compound_prompts = nn.ParameterList(
            [
                _random_parameter(
                    n_ctx_visual,
                    visual_width,
                    seed + layer_index,
                )
                for layer_index in range(1, prompt_depth)
                if n_ctx_visual > 0
            ]
        )

    def forward(self):
        return self.ctx, list(self.compound_prompts)


class CustomCLIP(nn.Module):
    def __init__(
        self,
        cfg,
        clip_model,
        classnames,
        teacher=None,
    ):
        super().__init__()
        self.cfg = cfg
        freeze_clip(clip_model)
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        visual_width = clip_model.visual.ln_pre.normalized_shape[0]
        prompt_depth = min(
            cfg.prompt_depth,
            clip_model.visual.transformer.layers,
        )
        self.classnames = tuple(classnames)
        self.image_text_kd_active = _image_text_kd_active(cfg)
        self.photo_text_active = self.image_text_kd_active
        self.sketch_text_active = self.image_text_kd_active
        self.photo_visual_prompt = IndependentVisualPromptLearner(
            cfg.n_ctx_visual,
            visual_width,
            cfg.seed + 201,
            prompt_depth,
        )
        self.sketch_visual_prompt = IndependentVisualPromptLearner(
            cfg.n_ctx_visual,
            visual_width,
            cfg.seed + 202,
            prompt_depth,
        )
        self.retrieval_head = getattr(cfg, "retrieval_head", "main")
        self.stroke_graph_head = None
        if self.retrieval_head == "sgcd":
            from src.stroke_graph import StrokeGraphEvidenceHead
            patch_count = clip_model.visual.positional_embedding.shape[0] - 1
            patch_grid = int(round(patch_count ** 0.5))
            if patch_grid * patch_grid != patch_count:
                raise ValueError("SGCD requires a square student patch lattice")
            cfg.sgcd_student_grid = patch_grid
            output_width = clip_model.visual.proj.shape[1]
            self.stroke_graph_head = StrokeGraphEvidenceHead(
                width=output_width,
                grid=patch_grid,
                bottleneck=cfg.sgcd_bottleneck,
                beta=cfg.sgcd_beta,
                temperature=cfg.sgcd_temperature,
                graph_steps=cfg.sgcd_graph_steps,
                graph_mix=cfg.sgcd_graph_mix,
            )
        photo_texts = [
            f"a photo of a {name.replace('_', ' ')}."
            for name in self.classnames
        ]
        sketch_texts = [
            f"a sketch of a {name.replace('_', ' ')}."
            for name in self.classnames
        ]
        self.register_buffer(
            "_student_photo_tokens",
            clip.tokenize(photo_texts),
            persistent=False,
        )
        self.register_buffer(
            "_student_sketch_tokens",
            clip.tokenize(sketch_texts),
            persistent=False,
        )
        self.register_buffer(
            "_student_photo_text_features",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_student_sketch_text_features",
            None,
            persistent=False,
        )

        # The pretrained teacher is reloaded when needed and must not be saved
        # inside every student checkpoint.
        object.__setattr__(self, "_teacher", teacher)
        self.persistent_teacher_cache = _persistent_teacher_cache_available(cfg)
        self.teacher_active = teacher is not None or self.persistent_teacher_cache
        self.teacher_prompts = _build_teacher_prompts(cfg, teacher)

        self.register_buffer("_teacher_sketch_text", None, persistent=False)
        self.register_buffer("_teacher_photo_text", None, persistent=False)

        print(
            "[Student] frozen text encoder with fixed modality templates; "
            "independent deep visual prompts; "
            f"n_ctx_visual={cfg.n_ctx_visual}, "
            f"prompt_depth={prompt_depth}"
        )
        print(
            "[Domain KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_domain}, "
            f"temperature={cfg.kd_temperature}"
        )
        print(
            "[Modality KD] photo-text + sketch-text -> "
            f"lambda={cfg.lambda_modality}, "
            f"photo_temperature={cfg.photo_text_kd_temperature}, "
            f"sketch_temperature={cfg.sketch_text_kd_temperature}"
        )
        if self.stroke_graph_head is not None:
            print(
                "[SGCD] photo-conditioned, causally verified stroke paths -> "
                f"grid={cfg.sgcd_student_grid}x{cfg.sgcd_student_grid}, "
                f"target={cfg.sgcd_target}, beta={cfg.sgcd_beta}, "
                f"lambda={cfg.lambda_sgcd}"
            )

    @staticmethod
    def _path_fingerprint(paths, root):
        digest = hashlib.sha256()
        for path in paths:
            relative = os.path.relpath(path, root).replace("\\", "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _teacher_cache_metadata(self, train_dataset):
        cfg = self.cfg
        return {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": cfg.dataset,
            "max_size": train_dataset.max_size,
            "classnames": list(self.classnames),
            "sketch_count": len(train_dataset.all_sketches_path),
            "photo_count": len(train_dataset.all_photo_paths),
            "sketch_fingerprint": self._path_fingerprint(
                train_dataset.all_sketches_path, cfg.root
            ),
            "photo_fingerprint": self._path_fingerprint(
                train_dataset.all_photo_paths, cfg.root
            ),
            **_teacher_training_config(cfg),
        }

    def _load_persistent_teacher_cache(self, train_dataset):
        cache_path = Path(self.cfg.teacher_cache_path)
        try:
            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
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
        required_tensors = (
            "teacher_sketch_features",
            "teacher_photo_features",
            "teacher_sketch_text",
            "teacher_photo_text",
        )
        missing = [key for key in required_tensors if key not in payload]
        if mismatches or missing:
            details = []
            if mismatches:
                details.append("metadata: " + ", ".join(mismatches))
            if missing:
                details.append("tensors: " + ", ".join(missing))
            raise RuntimeError(
                f"Teacher cache {cache_path} is incompatible ({'; '.join(details)}). "
                "Use another --teacher_cache_path or pass "
                "--rebuild_teacher_cache."
            )

        train_dataset.set_teacher_features(
            payload["teacher_sketch_features"],
            payload["teacher_photo_features"],
        )
        self._teacher_sketch_text = payload["teacher_sketch_text"]
        self._teacher_photo_text = payload["teacher_photo_text"]
        self.teacher_active = True
        self.teacher_prompts = None
        object.__setattr__(self, "_teacher", None)
        cache_size_mb = cache_path.stat().st_size / 1024**2
        print(
            f"[Teacher Cache] loaded {cache_path} ({cache_size_mb:.1f} MB); "
            "skipped DFN5B encoding and teacher pretraining."
        )

    def _encode_teacher_image(self, images, modality):
        if self.teacher_prompts is None:
            return self._teacher.encode_image(images)

        def encode(current_images):
            return self.teacher_prompts(current_images, modality)

        if (
            self.cfg.teacher_prompt_gradient_checkpointing
            and torch.is_grad_enabled()
        ):
            return checkpoint(encode, images, use_reentrant=False)
        return encode(images)

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
                "Teacher prompt pretraining requires "
                "teacher_pretrain_epochs > 0."
            )

        cfg = self.cfg
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
        self.teacher_prompts.requires_grad_(True)
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.teacher_pretrain_batch_size,
            shuffle=False,
            sampler=WorkerInvariantSampler(train_dataset, cfg.seed),
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            generator=torch.Generator().manual_seed(cfg.seed),
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
            "cuda",
            enabled=teacher_device.type == "cuda",
        )
        best_precision = -float("inf")
        best_epoch = 0
        best_prompt_state = None

        for epoch in range(cfg.teacher_pretrain_epochs):
            retrieval_total = 0.0
            steps = 0
            batches = tqdm(
                loader,
                desc=(
                    "Teacher prompt pretrain "
                    f"{epoch + 1}/{cfg.teacher_pretrain_epochs}"
                ),
                disable=not show_progress,
            )
            with torch.enable_grad():
                for photo, sketch, _, _, labels in batches:
                    photo = photo.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    sketch = sketch.to(
                        teacher_device, dtype=teacher_dtype, non_blocking=True
                    )
                    labels = labels.to(teacher_device, non_blocking=True)
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
                        retrieval = batch_hard_teacher_triplet_loss(
                            sketch_features,
                            photo_features,
                            labels,
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
                        batches.set_postfix(
                            T_TRI=f"{retrieval.item():.3f}",
                        )
            scheduler.step()
            if steps == 0:
                raise RuntimeError(
                    "Teacher pretraining produced no complete batches."
                )
            print(
                f"[Teacher Pretrain] epoch={epoch + 1}, "
                f"retrieval={retrieval_total / steps:.6f}"
            )
            precision = self._validate_teacher_unseen(
                val_sketch_loader,
                val_photo_loader,
                epoch + 1,
                show_progress,
            )
            if precision > best_precision:
                best_precision = precision
                best_epoch = epoch + 1
                best_prompt_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.teacher_prompts.state_dict().items()
                }

        if best_prompt_state is None:
            raise RuntimeError("Teacher best-precision state was not created.")
        self.teacher_prompts.load_state_dict(best_prompt_state, strict=True)
        self.teacher_best_precision = best_precision
        self.teacher_best_epoch = best_epoch
        print(
            "[Teacher Best] restored visual prompts from "
            f"epoch={best_epoch}, precision={best_precision:.6f}"
        )

        self.teacher_prompts.requires_grad_(False)

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
            labels = []
            batches = tqdm(
                loader,
                desc=f"Teacher unseen {modality} epoch {epoch}",
                disable=not show_progress,
            )
            for images, current_labels in batches:
                images = images.to(
                    teacher_device,
                    dtype=teacher_dtype,
                    non_blocking=True,
                )
                current_features = self._encode_teacher_image(
                    images, modality
                )
                features.append(current_features.float().cpu())
                labels.append(current_labels.cpu())
            return torch.cat(features), torch.cat(labels)

        sketch_features, sketch_labels = encode_loader(
            val_sketch_loader, "sketch"
        )
        photo_features, photo_labels = encode_loader(
            val_photo_loader, "photo"
        )
        mean_ap, precision, map_k, p_k = _retrieval_metrics(
            sketch_features,
            photo_features,
            sketch_labels,
            photo_labels,
            self.cfg.dataset,
        )
        map_name = f"mAP@{map_k}" if map_k else "mAP@all"
        print(
            f"[Teacher Validation] epoch={epoch}, "
            f"{map_name}={mean_ap.item():.4f}, "
            f"P@{p_k}={precision.item():.4f}"
        )
        return precision.item()

    @torch.no_grad()
    def _materialize_teacher_features(
        self,
        paths,
        modality,
        batch_size,
        workers,
        show_progress,
    ):
        teacher_parameter = self._teacher.visual.conv1.weight
        teacher_device = teacher_parameter.device
        teacher_dtype = teacher_parameter.dtype
        dataset = TeacherFeatureDataset(paths, self.cfg.max_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if workers > 0 else None,
        )
        output = torch.empty(
            len(paths),
            DFN5B_OUTPUT_DIM,
            dtype=torch.float16,
        )
        offset = 0
        batches = tqdm(
            loader,
            desc=f"Caching {modality} teacher features",
            disable=not show_progress,
        )
        for images in batches:
            images = images.to(
                teacher_device, dtype=teacher_dtype, non_blocking=True
            )
            features = self._encode_teacher_image(images, modality)
            end = offset + len(features)
            output[offset:end].copy_(features.to(dtype=torch.float16).cpu())
            offset = end
        return output

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
            "teacher_best_precision": getattr(
                self,
                "teacher_best_precision",
                None,
            ),
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
        cache_size_mb = cache_path.stat().st_size / 1024**2
        print(
            f"[Teacher Cache] saved {cache_path} ({cache_size_mb:.1f} MB)."
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

        image_count = (
            len(train_dataset.all_sketches_path)
            + len(train_dataset.all_photo_paths)
        )
        cache_size_mb = (
            image_count
            * DFN5B_OUTPUT_DIM
            * torch.tensor([], dtype=torch.float16).element_size()
            / 1024**2
        )

        if self.teacher_prompts is not None:
            self._pretrain_teacher_prompts(
                train_dataset,
                val_sketch_loader,
                val_photo_loader,
                workers,
                show_progress,
            )

        if self.image_text_kd_active or self.cfg.teacher_pretrain_epochs > 0:
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

        if self.cfg.teacher_pretrain_epochs > 0:
            prompt_state = {
                key: value.detach().cpu()
                for key, value in self.teacher_prompts.state_dict().items()
            }
            self._save_persistent_teacher_cache(
                train_dataset,
                sketch_features,
                photo_features,
                prompt_state,
            )

        teacher = self._teacher
        self.teacher_prompts = None
        object.__setattr__(self, "_teacher", None)
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "[Teacher Cache] materialized tuned seen-image features; "
            f"images={image_count:,}, memory={cache_size_mb:.1f} MB. "
            "DFN5B released."
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        if self._teacher is not None:
            self._teacher.eval()
        return self

    def get_teacher_text_features(self):
        if self._teacher_sketch_text is not None:
            return self._teacher_sketch_text, self._teacher_photo_text

        sketch_texts = [
            f"a sketch of a {name.replace('_', ' ')}."
            for name in self.classnames
        ]
        photo_texts = [
            f"a photo of a {name.replace('_', ' ')}."
            for name in self.classnames
        ]
        teacher_device = next(self._teacher.parameters()).device
        tokens = self._teacher.text_tokenizer(
            sketch_texts + photo_texts
        ).to(teacher_device)
        with torch.no_grad():
            text_features = F.normalize(
                self._teacher.encode_text(tokens).float(), dim=-1
            )
        class_count = len(self.classnames)
        self._teacher_sketch_text = text_features[:class_count]
        self._teacher_photo_text = text_features[class_count:]
        return (
            self._teacher_sketch_text,
            self._teacher_photo_text,
        )

    def get_visual_prompt(self, modality):
        if modality == "photo":
            return self.photo_visual_prompt()
        return self.sketch_visual_prompt()

    def get_student_text_features(self, modality):
        feature_name = f"_student_{modality}_text_features"
        features = getattr(self, feature_name)
        if features is None:
            tokens = getattr(self, f"_student_{modality}_tokens")
            with torch.no_grad():
                features = F.normalize(
                    self.clip_model.encode_text(tokens).float(),
                    dim=-1,
                )
            setattr(self, feature_name, features)
        return features

    def encode_student_image_details(self, image, modality):
        visual_prompt, compound_prompts = self.get_visual_prompt(modality)
        if self.stroke_graph_head is None or modality != "sketch":
            native = self.clip_model.visual(
                image.type(self.dtype), visual_prompt, compound_prompts
            )
            native = F.normalize(native.float(), dim=-1)
            return {
                "descriptor": native,
                "native": native,
                "dense": None,
                "weights": None,
                "evidence": native,
                "correction": torch.zeros_like(native),
                "ink_mass": None,
            }
        from src.stroke_graph import FinalBlockInputCapture, projected_patch_features, patch_ink_mass
        with FinalBlockInputCapture(self.clip_model.visual) as capture:
            native = self.clip_model.visual(
                image.type(self.dtype), visual_prompt, compound_prompts
            )
        native = F.normalize(native.float(), dim=-1)
        dense = projected_patch_features(self.clip_model.visual, capture.residual())
        ink = patch_ink_mass(
            image,
            self.cfg.sgcd_student_grid,
            self.cfg.sgcd_ink_threshold,
            self.cfg.sgcd_ink_softness,
        )
        return self.stroke_graph_head(native, dense, ink)

    def encode_student_image(self, image, modality):
        return self.encode_student_image_details(image, modality)["descriptor"]

    def _assemble_features(self, photo_features, sketch_features,
                           teacher_photo_base, teacher_sketch_base):
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
                teacher_sketch_text, teacher_photo_text = self.get_teacher_text_features()
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
        )

    def forward_with_stroke_graph(self, x):
        photo_tensor, sketch_tensor, teacher_photo_base, teacher_sketch_base, _label = x
        photo = self.encode_student_image_details(photo_tensor, "photo")
        sketch = self.encode_student_image_details(sketch_tensor, "sketch")
        features = self._assemble_features(
            photo["descriptor"], sketch["descriptor"],
            teacher_photo_base, teacher_sketch_base,
        )
        return features, sketch

    def forward(self, x):
        if self.stroke_graph_head is not None:
            return self.forward_with_stroke_graph(x)[0]
        photo_tensor, sketch_tensor, teacher_photo_base, teacher_sketch_base, _label = x
        photo_features = self.encode_student_image(photo_tensor, "photo")
        sketch_features = self.encode_student_image(sketch_tensor, "sketch")
        return self._assemble_features(
            photo_features, sketch_features, teacher_photo_base, teacher_sketch_base
        )

    def extract_feature(self, image, modality, return_details=False):
        output = self.encode_student_image_details(image, modality)
        return output if return_details else output["descriptor"]



class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classnames):
        super().__init__()
        self.args = args
        clip_model = _load_clip_model(args.backbone)

        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_precision = 0.0

        teacher = _load_teacher(args)
        self.model = CustomCLIP(
            cfg=args,
            clip_model=clip_model,
            classnames=classnames,
            teacher=teacher,
        )

        self.lambda_sgcd = getattr(args, "lambda_sgcd", 0.0)
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        self._sgcd_last_validation = {}
        self.save_hyperparameters({"args": dict(vars(args)), "classnames": list(classnames)})

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
        named = [(name, parameter) for name, parameter in self.model.named_parameters()
                 if parameter.requires_grad]
        head = [parameter for name, parameter in named if "stroke_graph_head." in name]
        base = [parameter for name, parameter in named if "stroke_graph_head." not in name]
        param_groups = []
        if base:
            param_groups.append({
                "params": base,
                "lr": self.args.lr,
                "momentum": self.args.momentum,
                "weight_decay": self.args.weight_decay,
            })
        if head:
            param_groups.append({
                "params": head,
                "lr": self.args.sgcd_head_lr,
                "momentum": self.args.momentum,
                "weight_decay": self.args.weight_decay,
            })
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
        )
        trainable = sum(parameter.numel() for _, parameter in named)
        print(
            "[Optimizer] SGD "
            f"prompt_lr={self.args.lr}, "
            f"head_lr={getattr(self.args, 'sgcd_head_lr', self.args.lr)}, "
            f"momentum={self.args.momentum}, weight_decay={self.args.weight_decay}, "
            f"trainable_params={trainable:,}"
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1,
        )
        return [optimizer], [scheduler]

    def forward(self, data):
        return self.model(data)
    
    def _sgcd_schedule_factor(self, batch_idx):
        if self.lambda_sgcd <= 0:
            return 0.0
        batches = max(1, int(getattr(self.trainer, "num_training_batches", 1)))
        progress = float(self.current_epoch) + batch_idx / batches
        warmup = self.args.sgcd_warmup_epochs
        factor = 1.0 if warmup <= 0 else min(1.0, progress / warmup)
        decay_start = self.args.sgcd_decay_start_epoch
        total = float(self.trainer.max_epochs)
        if progress > decay_start and total > decay_start:
            phase = min(1.0, (progress - decay_start) / (total - decay_start))
            factor *= 0.5 * (1.0 + math.cos(math.pi * phase))
        return factor

    def _select_sgcd_target(self, target):
        names = ("verified", "local", "random")
        name = self.args.sgcd_target
        shuffled = name == "shuffled"
        index = 0 if shuffled else names.index(name)
        selected = {
            "map": target["maps"][:, index],
            "mask_priority": target["mask_priorities"][:, index],
            "teacher_evidence": target["teacher_evidence"][:, index],
            "teacher_masked": target["teacher_masked"][:, index],
            "confidence": target["confidence"][:, index],
        }
        if shuffled:
            selected = {key: value.roll(1, 0) for key, value in selected.items()}
        return selected

    def stroke_graph_loss(self, batch, features, output):
        from src.stroke_graph import (
            counterfactual_field_alignment,
            centered_field_alignment,
            erase_by_patch_evidence,
            evidence_entropy,
            hellinger_loss,
        )
        if len(batch) != 6 or not isinstance(batch[5], dict):
            raise RuntimeError("SGCD targets are missing; prepare the complete cache before fitting")
        target = self._select_sgcd_target(batch[5])
        confidence = target["confidence"].to(self.device).float().clamp(0, 1.0)
        teacher_map = target["map"].to(self.device).float()
        where = hellinger_loss(output["weights"], teacher_map, confidence)
        what, what_cosine = centered_field_alignment(
            output["evidence"], features[0].detach(),
            target["teacher_evidence"], features[2], confidence,
        )
        anchor = (1 - F.cosine_similarity(
            output["descriptor"], output["native"].detach(), dim=-1
        )).mean()
        effect = output["descriptor"].sum() * 0
        effect_stats = {
            "effect_cosine": effect.detach(),
            "student_effect_rms": effect.detach(),
            "teacher_effect_rms": effect.detach(),
            "effect_magnitude_ratio": effect.detach(),
        }
        masked_output = None
        if self.args.lambda_sgcd_effect > 0:
            masked_images, removed = erase_by_patch_evidence(
                batch[1], target["mask_priority"].to(self.device).float(),
                self.args.sgcd_mask_fraction,
                self.args.sgcd_ink_threshold, self.args.sgcd_ink_softness,
            )
            masked_output = self.model.encode_student_image_details(masked_images, "sketch")
            effect, effect_stats = counterfactual_field_alignment(
                output["descriptor"], masked_output["descriptor"], features[0].detach(),
                features[3], target["teacher_masked"], features[2], confidence,
                self.args.sgcd_effect_magnitude_weight,
            )
            effect_stats["removed_ink_fraction"] = removed.mean().detach()
        values = {
            "where": where,
            "what": what,
            "effect": effect,
            "anchor": anchor,
        }
        total = (
            self.args.lambda_sgcd_where * where
            + self.args.lambda_sgcd_what * what
            + self.args.lambda_sgcd_effect * effect
            + self.args.lambda_sgcd_anchor * anchor
        )
        statistics = {
            **{name: value.detach() for name, value in values.items()},
            "what_cosine": what_cosine.detach(),
            "student_entropy": evidence_entropy(output["weights"]).mean().detach(),
            "teacher_entropy": evidence_entropy(teacher_map).mean().detach(),
            "map_cosine": F.cosine_similarity(output["weights"], teacher_map, dim=-1).mean().detach(),
            "descriptor_native_cosine": F.cosine_similarity(
                output["descriptor"], output["native"], dim=-1
            ).mean().detach(),
            "correction_norm": output["correction"].norm(dim=-1).mean().detach(),
            "confidence": confidence.mean().detach(),
            "confidence_nonzero": (confidence > 0).float().mean().detach(),
            **effect_stats,
        }
        return total, statistics, masked_output

    def training_step(self, batch, batch_idx):
        if self.model.stroke_graph_head is not None:
            features, output = self.model.forward_with_stroke_graph(batch[:5])
        else:
            features, output = self(batch[:5]), None
        main_loss, loss_dict = loss_fn(self.args, features)
        loss = main_loss
        if self.lambda_sgcd > 0:
            sgcd_loss, statistics, _masked = self.stroke_graph_loss(batch, features, output)
            schedule = self._sgcd_schedule_factor(batch_idx)
            weighted = self.lambda_sgcd * schedule * sgcd_loss
            loss = loss + weighted
            self.log("SGCD", sgcd_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("SGCD_WEIGHTED", weighted, on_step=False, on_epoch=True)
            self.log("SGCD_SCHEDULE", schedule, on_step=False, on_epoch=True)
            for name, value in statistics.items():
                self.log("SGCD_" + name, value, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("main_loss", main_loss, on_step=False, on_epoch=True)
        for key, bar_name in {"domain_kd": "DOMAIN", "modality_kd": "MODALITY"}.items():
            self.log(bar_name, loss_dict[key], on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        image_tensor, label = batch
        modality = "sketch" if dataloader_idx == 0 else "photo"
        output = self.model.extract_feature(image_tensor, modality, return_details=True)
        record = (output["descriptor"], output["native"], label)
        if dataloader_idx == 0:
            self.val_step_outputs_sk.append(record)
        else:
            self.val_step_outputs_ph.append(record)

    def on_validation_epoch_end(self):
        if not self.val_step_outputs_sk or not self.val_step_outputs_ph:
            return
        query_features = torch.cat([value[0] for value in self.val_step_outputs_sk])
        gallery_features = torch.cat([value[0] for value in self.val_step_outputs_ph])
        query_native = torch.cat([value[1] for value in self.val_step_outputs_sk])
        gallery_native = torch.cat([value[1] for value in self.val_step_outputs_ph])
        sketch_labels = torch.cat([value[2] for value in self.val_step_outputs_sk]).cpu()
        photo_labels = torch.cat([value[2] for value in self.val_step_outputs_ph]).cpu()
        mAP, precision, map_k, p_k = _retrieval_metrics(
            query_features, gallery_features, sketch_labels, photo_labels, self.args.dataset
        )
        native_mAP, native_precision, _native_map_k, _native_p_k = _retrieval_metrics(
            query_native, gallery_native, sketch_labels, photo_labels, self.args.dataset
        )
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log("precision", precision, on_step=False, on_epoch=True)
        self.log("native_mAP", native_mAP, on_step=False, on_epoch=True)
        self.log("native_precision", native_precision, on_step=False, on_epoch=True)
        self._sgcd_last_validation = {
            "mAP": mAP.item(),
            "precision": precision.item(),
            "native_mAP": native_mAP.item(),
            "native_precision": native_precision.item(),
            "descriptor_native_cosine_sketch": F.cosine_similarity(
                query_features.float(), query_native.float(), dim=-1
            ).mean().item(),
            "descriptor_native_cosine_photo": F.cosine_similarity(
                gallery_features.float(), gallery_native.float(), dim=-1
            ).mean().item(),
        }
        if self.global_step > 0:
            self.best_precision = max(self.best_precision, precision.item())
        label = f"mAP@{map_k}" if map_k else "mAP@all"
        print(
            f"{label}: {mAP.item()}, P@{p_k}: {precision}, Best P@{p_k}: {self.best_precision}; "
            f"native mAP: {native_mAP.item()}, native P@{p_k}: {native_precision}"
        )
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint.setdefault("hyper_parameters", {}).update(
            args=dict(vars(self.args)), classnames=list(self.model.classnames)
        )
        checkpoint["experiment_config"] = {
            "method": "SGCD" if self.model.stroke_graph_head is not None else "main",
            "retrieval_head": getattr(self.args, "retrieval_head", "main"),
            "args": dict(vars(self.args)),
            "stroke_graph_target_metadata": getattr(self.args, "sgcd_target_metadata", None),
        }
