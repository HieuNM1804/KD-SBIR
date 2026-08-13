import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
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
    TeacherAdapterDataset,
    TeacherFeatureDataset,
    WorkerInvariantSampler,
)
from src.text_encoder import TextEncoder
from src.losses import (
    batch_hard_teacher_triplet_loss,
    loss_fn,
    teacher_semantic_loss,
)
from src.teacher_adapters import ModalityAdapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DFN5B teacher loader
# ---------------------------------------------------------------------------
DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
DFN5B_OUTPUT_DIM = 1024
TEACHER_CACHE_FORMAT_VERSION = 2


def _teacher_training_config(args):
    """Parameters that can change the cached adapted teacher targets."""
    return {
        "teacher_model": DFN5B_MODEL,
        "teacher_pretrained": DFN5B_PRETRAINED,
        "teacher_output_dim": DFN5B_OUTPUT_DIM,
        "teacher_precision": "fp16",
        "joint_teacher_adapter": args.joint_teacher_adapter,
        "adapter_bottleneck": args.teacher_adapter_bottleneck,
        "adapter_lr": args.teacher_adapter_lr,
        "teacher_momentum": args.teacher_momentum,
        "teacher_weight_decay": args.teacher_weight_decay,
        "pretrain_epochs": args.teacher_pretrain_epochs,
        "pretrain_batch_size": args.teacher_pretrain_batch_size,
        "lambda_retrieval": args.lambda_teacher_retrieval,
        "lambda_semantic": args.lambda_teacher_semantic,
        "temperature": args.teacher_temperature,
        "triplet_margin": args.teacher_triplet_margin,
        "scheduler": "StepLR",
        "scheduler_step_size": 5,
        "scheduler_gamma": 0.1,
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
    return (
        args.lambda_photo_text_kd > 0
        or args.lambda_sketch_text_kd > 0
    )


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


def _build_teacher_adapters(args, teacher):
    if not args.joint_teacher_adapter or teacher is None:
        return None

    feature_dim = int(teacher.output_dim)
    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=args.teacher_adapter_bottleneck,
    )
    print(
        "[Teacher Adapter] initialized for joint training "
        f"(feature_dim={feature_dim}, "
        f"bottleneck={args.teacher_adapter_bottleneck})"
    )
    return adapters


def _load_teacher(args):
    if _persistent_teacher_cache_available(args):
        print(
            "[Teacher Cache] persistent cache found; "
            "skipping DFN5B loading."
        )
        return None

    if (
        args.lambda_kd <= 0
        and not args.joint_teacher_adapter
        and not _image_text_kd_active(args)
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
    if args.joint_teacher_adapter or _image_text_kd_active(args):
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


class IndependentTextPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx_text,
        text_width,
        classnames,
        token_embedding,
        modality,
        seed,
        prompt_depth,
    ):
        super().__init__()
        self.n_ctx = n_ctx_text
        modality_name = "photo" if modality == "photo" else "sketch"
        classnames = [name.replace("_", " ") for name in classnames]

        if n_ctx_text == 0:
            ctx = None
        elif n_ctx_text <= 3:
            init_phrase = f"a {modality_name} of"
            init_tokens = clip.tokenize(init_phrase)
            with torch.no_grad():
                embeddings = token_embedding(init_tokens).detach()
            ctx = embeddings[0, 1 : 1 + n_ctx_text].clone()
        else:
            ctx = _random_parameter(
                n_ctx_text,
                text_width,
                seed,
            ).detach()

        self.ctx = nn.Parameter(ctx) if ctx is not None else None
        self.compound_prompts = nn.ParameterList()
        for layer_index in range(1, prompt_depth):
            if n_ctx_text == 0:
                break
            if n_ctx_text <= 3:
                deep_ctx = ctx.clone()
            else:
                deep_ctx = _random_parameter(
                    n_ctx_text,
                    text_width,
                    seed + layer_index,
                ).detach()
            self.compound_prompts.append(nn.Parameter(deep_ctx))

        base_prompts = [
            f"a {modality_name} of a {name}" for name in classnames
        ]
        if n_ctx_text <= 3:
            raw_prompts = [f"{base}." for base in base_prompts]
        else:
            placeholders = " ".join(["X"] * n_ctx_text)
            raw_prompts = [
                f"{base} {placeholders}." for base in base_prompts
            ]
        try:
            tokenized_prompts = clip.tokenize(raw_prompts)
        except RuntimeError as error:
            raise ValueError(
                f"n_ctx_text={n_ctx_text} exceeds CLIP's text context length."
            ) from error

        if 0 < n_ctx_text <= 3:
            prompt_positions = torch.arange(
                1, 1 + n_ctx_text
            ).expand(len(classnames), -1)
        elif n_ctx_text > 3:
            base_tokens = clip.tokenize(base_prompts)
            prompt_starts = base_tokens.argmax(dim=-1)
            prompt_positions = prompt_starts[:, None] + torch.arange(
                n_ctx_text
            )[None, :]
            eot_positions = tokenized_prompts.argmax(dim=-1)
            if torch.any(prompt_positions[:, -1] >= eot_positions):
                raise ValueError(
                    f"n_ctx_text={n_ctx_text} leaves no room for the end-of-text token."
                )
        else:
            prompt_positions = torch.empty(
                len(classnames), 0, dtype=torch.long
            )

        with torch.no_grad():
            prompt_embeddings = token_embedding(tokenized_prompts).detach()
        self.register_buffer(
            "tokenized_prompts",
            tokenized_prompts,
            persistent=False,
        )
        self.register_buffer(
            "prompt_embeddings",
            prompt_embeddings,
            persistent=False,
        )
        self.register_buffer(
            "prompt_positions",
            prompt_positions,
            persistent=False,
        )

    def forward(self):
        if self.ctx is None:
            return (
                self.tokenized_prompts,
                self.prompt_embeddings,
                [],
                self.prompt_positions,
            )

        prompts = self.prompt_embeddings.clone()
        context = self.ctx.to(dtype=prompts.dtype)
        batch_indices = torch.arange(
            prompts.shape[0], device=prompts.device
        )[:, None]
        prompts[batch_indices, self.prompt_positions] = context.unsqueeze(0)
        return (
            self.tokenized_prompts,
            prompts,
            list(self.compound_prompts),
            self.prompt_positions,
        )


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
        self.dtype = clip_model.dtype

        self.visual_encoder = clip_model.visual
        visual_width = self.visual_encoder.ln_pre.normalized_shape[0]
        text_width = clip_model.ln_final.normalized_shape[0]
        prompt_depth = min(
            cfg.prompt_depth,
            clip_model.visual.transformer.layers,
            clip_model.transformer.layers,
        )
        self.classnames = tuple(classnames)
        self.classification_active = cfg.lambda_cls > 0
        self.photo_text_active = (
            self.classification_active or cfg.lambda_photo_text_kd > 0
        )
        self.sketch_text_active = (
            self.classification_active or cfg.lambda_sketch_text_kd > 0
        )
        self.image_text_kd_active = _image_text_kd_active(cfg)
        self.photo_text_prompt = (
            IndependentTextPromptLearner(
                cfg.n_ctx_text,
                text_width,
                self.classnames,
                clip_model.token_embedding,
                "photo",
                cfg.seed + 101,
                prompt_depth,
            )
            if self.photo_text_active
            else None
        )
        self.sketch_text_prompt = (
            IndependentTextPromptLearner(
                cfg.n_ctx_text,
                text_width,
                self.classnames,
                clip_model.token_embedding,
                "sketch",
                cfg.seed + 102,
                prompt_depth,
            )
            if self.sketch_text_active
            else None
        )
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
        self.text_encoder = (
            TextEncoder(clip_model)
            if self.photo_text_active or self.sketch_text_active
            else None
        )
        self.logit_scale = clip_model.logit_scale

        # The pretrained teacher is reloaded when needed and must not be saved
        # inside every student checkpoint.
        object.__setattr__(self, "_teacher", teacher)
        self.persistent_teacher_cache = _persistent_teacher_cache_available(cfg)
        self.teacher_active = teacher is not None or self.persistent_teacher_cache
        self.joint_teacher_adapter = (
            cfg.joint_teacher_adapter and not self.persistent_teacher_cache
        )
        self.teacher_adapters = _build_teacher_adapters(cfg, teacher)

        self.register_buffer("_teacher_sketch_text", None, persistent=False)
        self.register_buffer("_teacher_photo_text", None, persistent=False)

        print(
            "[Student] frozen CLIP with independent deep text and visual prompts; "
            f"n_ctx_text={cfg.n_ctx_text}, "
            f"n_ctx_visual={cfg.n_ctx_visual}, "
            f"prompt_depth={prompt_depth}; no cross-modal projection"
        )
        print(
            "[Relational KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_kd}, "
            f"temperature={cfg.kd_temperature}"
        )
        print(
            "[Classification] "
            f"active={self.classification_active}, lambda={cfg.lambda_cls}"
        )
        print(
            "[Image-Text KD] "
            f"photo_lambda={cfg.lambda_photo_text_kd}, "
            f"sketch_lambda={cfg.lambda_sketch_text_kd}, "
            f"photo_temperature={cfg.photo_text_kd_temperature}, "
            f"sketch_temperature={cfg.sketch_text_kd_temperature}"
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
        self.joint_teacher_adapter = False
        self.teacher_adapters = None
        object.__setattr__(self, "_teacher", None)
        cache_size_mb = cache_path.stat().st_size / 1024**2
        print(
            f"[Teacher Cache] loaded {cache_path} ({cache_size_mb:.1f} MB); "
            "skipped DFN5B encoding and teacher pretraining."
        )

    def _pretrain_teacher_adapters(
        self,
        train_dataset,
        workers,
        show_progress,
    ):
        if self.teacher_adapters is None:
            raise RuntimeError(
                "Teacher pretraining requires --joint_teacher_adapter."
            )

        cfg = self.cfg
        teacher_device = next(self._teacher.parameters()).device
        self.teacher_adapters.to(device=teacher_device, dtype=torch.float32)
        self.teacher_adapters.requires_grad_(True).train()
        teacher_sketch_text, teacher_photo_text = (
            self.get_teacher_text_features()
        )
        adapter_dataset = TeacherAdapterDataset(train_dataset)
        loader = DataLoader(
            adapter_dataset,
            batch_size=cfg.teacher_pretrain_batch_size,
            shuffle=False,
            sampler=WorkerInvariantSampler(adapter_dataset, cfg.seed),
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        optimizer = torch.optim.SGD(
            self.teacher_adapters.parameters(),
            lr=cfg.teacher_adapter_lr,
            momentum=cfg.teacher_momentum,
            weight_decay=cfg.teacher_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
        )

        for epoch in range(cfg.teacher_pretrain_epochs):
            retrieval_total = 0.0
            semantic_total = 0.0
            steps = 0
            batches = tqdm(
                loader,
                desc=(
                    "Teacher adapter pretrain "
                    f"{epoch + 1}/{cfg.teacher_pretrain_epochs}"
                ),
                disable=not show_progress,
            )
            with torch.enable_grad():
                for photo_base, sketch_base, labels in batches:
                    photo_base = photo_base.to(
                        teacher_device, dtype=torch.float32, non_blocking=True
                    )
                    sketch_base = sketch_base.to(
                        teacher_device, dtype=torch.float32, non_blocking=True
                    )
                    labels = labels.to(teacher_device, non_blocking=True)
                    photo_features = self.adapt_teacher_feature(
                        photo_base, "photo"
                    )
                    sketch_features = self.adapt_teacher_feature(
                        sketch_base, "sketch"
                    )
                    retrieval = batch_hard_teacher_triplet_loss(
                        sketch_features,
                        photo_features,
                        labels,
                        cfg.teacher_triplet_margin,
                    )
                    semantic = teacher_semantic_loss(
                        sketch_features,
                        photo_features,
                        labels,
                        teacher_sketch_text,
                        teacher_photo_text,
                        cfg.teacher_temperature,
                    )
                    loss = (
                        cfg.lambda_teacher_retrieval * retrieval
                        + cfg.lambda_teacher_semantic * semantic
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    retrieval_total += retrieval.detach().item()
                    semantic_total += semantic.detach().item()
                    steps += 1
                    if show_progress:
                        batches.set_postfix(
                            T_TRI=f"{retrieval.item():.3f}",
                            T_SEM=f"{semantic.item():.3f}",
                        )
            scheduler.step()
            if steps == 0:
                raise RuntimeError(
                    "Teacher pretraining produced no complete batches."
                )
            print(
                f"[Teacher Pretrain] epoch={epoch + 1}, "
                f"retrieval={retrieval_total / steps:.6f}, "
                f"semantic={semantic_total / steps:.6f}"
            )

        self.teacher_adapters.eval().requires_grad_(False)

    @torch.no_grad()
    def _materialize_adapted_features(self, features, modality):
        teacher_device = next(self._teacher.parameters()).device
        output = torch.empty_like(features, dtype=torch.float16)
        batch_size = self.cfg.teacher_pretrain_batch_size
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = features[start:end].to(
                teacher_device, dtype=torch.float32, non_blocking=True
            )
            adapted = self.adapt_teacher_feature(batch, modality)
            output[start:end].copy_(adapted.to(dtype=torch.float16).cpu())
        return output

    def _save_persistent_teacher_cache(
        self,
        train_dataset,
        sketch_features,
        photo_features,
        adapter_state,
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
            "adapter_state_dict": adapter_state,
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
        batch_size,
        workers,
        show_progress,
    ):
        if self.persistent_teacher_cache:
            self._load_persistent_teacher_cache(train_dataset)
            return
        if self._teacher is None:
            return

        sketch_count = len(train_dataset.all_sketches_path)
        paths = (
            train_dataset.all_sketches_path
            + train_dataset.all_photo_paths
        )
        feature_dataset = TeacherFeatureDataset(
            paths,
            train_dataset.max_size,
        )
        loader = DataLoader(
            feature_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if workers > 0 else None,
        )

        feature_cache = torch.empty(
            len(paths),
            DFN5B_OUTPUT_DIM,
            dtype=torch.float16,
        )
        cache_size_mb = (
            feature_cache.numel()
            * feature_cache.element_size()
            / 1024**2
        )
        teacher_device = next(self._teacher.parameters()).device
        offset = 0
        batches = tqdm(
            loader,
            desc="Caching DFN5B features",
            disable=not show_progress,
        )
        with torch.no_grad():
            for images in batches:
                images = images.to(
                    device=teacher_device,
                    dtype=torch.float16,
                    non_blocking=True,
                )
                features = self._teacher.encode_image(images)
                end = offset + len(features)
                feature_cache[offset:end].copy_(features.cpu())
                offset = end

        train_dataset.set_teacher_features(
            feature_cache[:sketch_count],
            feature_cache[sketch_count:],
        )
        if self.joint_teacher_adapter or self.image_text_kd_active:
            self.get_teacher_text_features()

        if self.cfg.teacher_pretrain_epochs > 0:
            self._pretrain_teacher_adapters(
                train_dataset,
                workers,
                show_progress,
            )
            adapter_state = {
                key: value.detach().cpu()
                for key, value in self.teacher_adapters.state_dict().items()
            }
            adapted_sketch = self._materialize_adapted_features(
                feature_cache[:sketch_count], "sketch"
            )
            adapted_photo = self._materialize_adapted_features(
                feature_cache[sketch_count:], "photo"
            )
            train_dataset.set_teacher_features(
                adapted_sketch,
                adapted_photo,
            )
            self._save_persistent_teacher_cache(
                train_dataset,
                adapted_sketch,
                adapted_photo,
                adapter_state,
            )
            self.joint_teacher_adapter = False
            self.teacher_adapters = None
            del feature_cache

        teacher = self._teacher
        object.__setattr__(self, "_teacher", None)
        del images, features
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "[Teacher Cache] encoded each seen image once; "
            f"images={len(paths):,}, memory={cache_size_mb:.1f} MB. "
            "DFN5B released."
        )

    def train(self, mode=True):
        super().train(mode)
        if self._teacher is not None:
            self._teacher.eval()
        if self.teacher_adapters is not None:
            self.teacher_adapters.train(mode and self.joint_teacher_adapter)
        return self

    def adapt_teacher_feature(self, feature, modality):
        if self.teacher_adapters is None:
            return feature
        feature = F.normalize(feature.float(), dim=-1)
        adapter = (
            self.teacher_adapters.photo
            if modality == "photo"
            else self.teacher_adapters.sketch
        )
        return adapter(feature)

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

    def get_text_prompt(self, modality):
        if modality == "photo":
            return self.photo_text_prompt
        return self.sketch_text_prompt

    def get_visual_prompt(self, modality):
        if modality == "photo":
            return self.photo_visual_prompt()
        return self.sketch_visual_prompt()

    def get_student_text_features(self, modality):
        tokenized_prompts, text_prompts, compound_prompts, prompt_positions = (
            self.get_text_prompt(modality)()
        )
        return self.text_encoder(
            tokenized_prompts,
            text_prompts,
            compound_prompts,
            prompt_positions,
        )

    def encode_student_image(self, image, modality):
        visual_prompt, compound_prompts = self.get_visual_prompt(modality)
        features = self.visual_encoder(
            image.type(self.dtype),
            visual_prompt,
            compound_prompts,
        )
        return features / features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        (
            photo_tensor,
            sk_tensor,
            teacher_photo_base,
            teacher_sketch_base,
            label,
        ) = x
        photo_features = self.encode_student_image(photo_tensor, "photo")
        sketch_features = self.encode_student_image(sk_tensor, "sketch")
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
        photo_logits = (
            self.logit_scale.exp()
            * photo_features
            @ student_photo_text.t()
            if self.classification_active
            else None
        )
        sketch_logits = (
            self.logit_scale.exp()
            * sketch_features
            @ student_sketch_text.t()
            if self.classification_active
            else None
        )

        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            teacher_photo_features = self.adapt_teacher_feature(
                teacher_photo_base, "photo"
            )
            teacher_sketch_features = self.adapt_teacher_feature(
                teacher_sketch_base, "sketch"
            )
            if self.joint_teacher_adapter or self.image_text_kd_active:
                teacher_sketch_text, teacher_photo_text = (
                    self.get_teacher_text_features()
                )

        return (
            photo_features,
            sketch_features,
            teacher_photo_features,
            teacher_sketch_features,
            label,
            photo_logits,
            sketch_logits,
            self.teacher_active,
            self.joint_teacher_adapter,
            student_sketch_text,
            student_photo_text,
            teacher_sketch_text,
            teacher_photo_text,
        )

    def extract_feature(self, image, modality):
        return self.encode_student_image(image, modality)


class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classnames):
        super().__init__()
        self.args = args
        clip_model = _load_clip_model(args.backbone)

        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3

        teacher = _load_teacher(args)
        self.model = CustomCLIP(
            cfg=args,
            clip_model=clip_model,
            classnames=classnames,
            teacher=teacher,
        )

        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []

    def cache_teacher_features(
        self,
        train_dataset,
        batch_size,
        workers,
        show_progress,
    ):
        self.model.cache_teacher_features(
            train_dataset,
            batch_size,
            workers,
            show_progress,
        )
        
    def configure_optimizers(self):
        adapter_params = (
            [
                parameter
                for parameter in self.model.teacher_adapters.parameters()
                if parameter.requires_grad
            ]
            if self.model.teacher_adapters is not None
            else []
        )
        adapter_param_ids = {id(parameter) for parameter in adapter_params}
        student_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in adapter_param_ids
        ]
        param_groups = [
            {
                "params": student_params,
                "lr": self.args.lr,
                "momentum": self.args.momentum,
                "weight_decay": self.args.weight_decay,
            }
        ]
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": self.args.teacher_adapter_lr,
                    "momentum": self.args.teacher_momentum,
                    "weight_decay": self.args.teacher_weight_decay,
                }
            )
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
        )
        trainable = sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        )
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum={self.args.momentum}, "
            f"weight_decay={self.args.weight_decay}, "
            f"teacher_adapter_lr="
            f"{self.args.teacher_adapter_lr if adapter_params else 'off'}, "
            f"teacher_momentum="
            f"{self.args.teacher_momentum if adapter_params else 'off'}, "
            f"teacher_weight_decay="
            f"{self.args.teacher_weight_decay if adapter_params else 'off'}, "
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
    
    def training_step(self, batch, batch_idx):
        features = self(batch)
        loss, loss_dict = loss_fn(self.args, features)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        bar_names = {
            "cls": "CE",
            "kd_sketch_photo": "KD_II",
            "image_text_kd": "KD_IT",
        }
        for key, bar_name in bar_names.items():
            self.log(
                bar_name,
                loss_dict[key],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, "sketch")
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, "photo")
            self.val_step_outputs_ph.append((feat, label))

    def on_validation_epoch_end(self):
        query_features = torch.cat(
            [features for features, _ in self.val_step_outputs_sk]
        )
        gallery_features = torch.cat(
            [features for features, _ in self.val_step_outputs_ph]
        )
        sketch_labels = torch.cat(
            [labels for _, labels in self.val_step_outputs_sk]
        ).cpu()
        photo_labels = torch.cat(
            [labels for _, labels in self.val_step_outputs_ph]
        ).cpu()

        ap = torch.zeros(len(query_features))
        precision_at_k = torch.zeros(len(query_features))
        if self.args.dataset == "sketchy_2":
            map_k = 200
            p_k = 200
        else:
            map_k = 0
            p_k = 200 if self.args.dataset == "quickdraw" else 100

        for idx, sketch_feature in enumerate(query_features):
            cosine = self.distance_fn(
                sketch_feature.unsqueeze(0), gallery_features
            ).cpu()
            # TorchMetrics treats non-positive predictions as non-relevant.
            # Map cosine from [-1, 1] to (0, 1] without changing its ranking.
            score = ((cosine + 1.0) * 0.5).clamp(
                min=torch.finfo(cosine.dtype).eps,
                max=1.0,
            )
            target = photo_labels.eq(sketch_labels[idx])

            if map_k:
                top_k = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(
                    score, target, top_k=top_k
                )
            else:
                ap[idx] = retrieval_average_precision(score, target)

            precision_at_k[idx] = retrieval_precision(
                score, target, top_k=p_k
            )

        mAP = ap.mean()
        precision = precision_at_k.mean()
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        if map_k:
            print(
                f"mAP@{map_k}: {mAP.item()}, P@{p_k}: {precision}, "
                f"Best mAP: {self.best_metric}"
            )
        else:
            print(
                f"mAP@all: {mAP.item()}, P@{p_k}: {precision}, "
                f"Best mAP: {self.best_metric}"
            )
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
