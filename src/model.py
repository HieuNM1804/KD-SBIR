import copy
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
from src.dataset import TeacherFeatureDataset
from src.text_encoder import TextEncoder
from src.losses import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DFN5B teacher loader
# ---------------------------------------------------------------------------
DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
DFN5B_OUTPUT_DIM = 1024


def _load_clip_model(backbone):
    model_path = clip.download_model(backbone)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return build_model(state_dict)


def _load_teacher(args):
    if args.lambda_kd <= 0:
        return None

    print(f"[Teacher] Loading {DFN5B_MODEL} in FP16...")
    teacher = open_clip.create_model(
        DFN5B_MODEL,
        pretrained=DFN5B_PRETRAINED,
        precision="fp16",
        device=device,
    )
    teacher.eval().requires_grad_(False)
    teacher.output_dim = DFN5B_OUTPUT_DIM
    return teacher


def freeze_clip_except_layer_norm(clip_model):
    clip_model.requires_grad_(False)
    for module in clip_model.modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)


def _random_parameter(rows, width, seed):
    if rows == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameter = torch.empty(rows, width)
    nn.init.normal_(parameter, std=0.02, generator=generator)
    return nn.Parameter(parameter)


class TextTailPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx_text,
        text_width,
        classnames,
        token_embedding,
        modality,
        seed,
    ):
        super().__init__()
        self.ctx = _random_parameter(n_ctx_text, text_width, seed)
        modality_name = "photo" if modality == "photo" else "sketch"
        class_phrases = [
            f"a {modality_name} of a {name.replace('_', ' ')}"
            for name in classnames
        ]
        placeholders = " ".join(["X"] * n_ctx_text)
        raw_prompts = [
            f"{phrase} {placeholders}." if placeholders else f"{phrase}."
            for phrase in class_phrases
        ]
        try:
            tokenized_prompts = clip.tokenize(raw_prompts)
        except RuntimeError as error:
            raise ValueError(
                f"n_ctx_text={n_ctx_text} exceeds CLIP's text context length."
            ) from error

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
        if n_ctx_text > 0:
            context_starts = [
                int(clip.tokenize(phrase).argmax())
                for phrase in class_phrases
            ]
            self.register_buffer(
                "context_starts",
                torch.tensor(context_starts, dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer(
                "context_starts",
                None,
                persistent=False,
            )

    def forward(self):
        if self.ctx is None:
            return self.tokenized_prompts, self.prompt_embeddings

        prompts = self.prompt_embeddings.clone()
        offsets = torch.arange(self.ctx.shape[0], device=prompts.device)
        context_indices = self.context_starts[:, None] + offsets[None, :]
        batch_indices = torch.arange(
            prompts.shape[0], device=prompts.device
        )[:, None]
        context = self.ctx.to(dtype=prompts.dtype)
        prompts[batch_indices, context_indices] = context.unsqueeze(0)
        return self.tokenized_prompts, prompts


class VisualPromptLearner(nn.Module):
    def __init__(self, n_ctx_visual, visual_width, seed):
        super().__init__()
        self.ctx = _random_parameter(n_ctx_visual, visual_width, seed)

    def forward(self):
        return self.ctx


class CustomCLIP(nn.Module):
    def __init__(
        self,
        cfg,
        clip_model,
        classnames,
        teacher=None,
    ):
        super().__init__()
        freeze_clip_except_layer_norm(clip_model)
        self.dtype = clip_model.dtype

        self.ph_encoder = clip_model.visual
        self.sk_encoder = copy.deepcopy(clip_model.visual)
        visual_width = self.ph_encoder.ln_pre.normalized_shape[0]
        text_width = clip_model.ln_final.normalized_shape[0]
        self.classnames = tuple(classnames)
        self.photo_text_prompt = TextTailPromptLearner(
            cfg.n_ctx_text,
            text_width,
            self.classnames,
            clip_model.token_embedding,
            "photo",
            cfg.seed + 101,
        )
        self.sketch_text_prompt = TextTailPromptLearner(
            cfg.n_ctx_text,
            text_width,
            self.classnames,
            clip_model.token_embedding,
            "sketch",
            cfg.seed + 102,
        )
        self.photo_visual_prompt = VisualPromptLearner(
            cfg.n_ctx_visual,
            visual_width,
            cfg.seed + 201,
        )
        self.sketch_visual_prompt = VisualPromptLearner(
            cfg.n_ctx_visual,
            visual_width,
            cfg.seed + 202,
        )
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        # The pretrained teacher is reloaded when needed and must not be saved
        # inside every student checkpoint.
        object.__setattr__(self, "_teacher", teacher)
        self.teacher_active = teacher is not None

        print(
            "[Student] independent random tail-text and visual prompts; "
            f"n_ctx_text={cfg.n_ctx_text}, "
            f"n_ctx_visual={cfg.n_ctx_visual}"
        )
        print(
            "[Relational KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_kd}, "
            f"temperature={cfg.kd_temperature}"
        )

    @torch.no_grad()
    def cache_teacher_features(
        self,
        train_dataset,
        batch_size,
        workers,
        show_progress,
    ):
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
        teacher_device = next(self._teacher.parameters()).device
        offset = 0
        batches = tqdm(
            loader,
            desc="Caching DFN5B features",
            disable=not show_progress,
        )
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

        teacher = self._teacher
        object.__setattr__(self, "_teacher", None)
        del images, features
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cache_size_mb = (
            feature_cache.numel()
            * feature_cache.element_size()
            / 1024**2
        )
        print(
            "[Teacher Cache] encoded each seen image once; "
            f"images={len(paths):,}, memory={cache_size_mb:.1f} MB. "
            "DFN5B released."
        )

    def train(self, mode=True):
        super().train(mode)
        if self._teacher is not None:
            self._teacher.eval()
        return self

    def get_text_prompt(self, modality):
        if modality == "photo":
            return self.photo_text_prompt
        return self.sketch_text_prompt

    def get_visual_prompt(self, modality):
        if modality == "photo":
            return self.photo_visual_prompt()
        return self.sketch_visual_prompt()

    def get_student_text_features(self, modality):
        tokenized_prompts, text_prompts = self.get_text_prompt(modality)()
        return self.text_encoder(tokenized_prompts, text_prompts)

    def encode_student_image(self, image, modality):
        if modality == "photo":
            image_encoder = self.ph_encoder
        else:
            image_encoder = self.sk_encoder
        visual_prompt = self.get_visual_prompt(modality)
        features = image_encoder(image.type(self.dtype), visual_prompt)
        return features / features.norm(dim=-1, keepdim=True)

    def get_logits(self, image, modality):
        text_features = self.get_student_text_features(modality)
        text_features = F.normalize(text_features, dim=-1)
        image_features = self.encode_student_image(image, modality)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features

    def forward(self, x):
        (
            photo_tensor,
            sk_tensor,
            teacher_photo_base,
            teacher_sketch_base,
            label,
        ) = x
        photo_logits, photo_features = self.get_logits(
            photo_tensor, "photo"
        )
        sk_logits, sketch_features = self.get_logits(
            sk_tensor, "sketch"
        )

        return (
            photo_features,
            sketch_features,
            teacher_photo_base,
            teacher_sketch_base,
            label,
            photo_logits,
            sk_logits,
            self.teacher_active,
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
        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(
            params=trainable_params,
            lr=self.args.lr,
            weight_decay=1e-3,
            momentum=0.9,
        )
        trainable = sum(parameter.numel() for parameter in trainable_params)
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum=0.9, weight_decay=1e-3, "
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
        for k, v in loss_dict.items():
            bar_names = {
                "kd_sketch_photo": "KD_SP",
            }
            show_on_bar = k in bar_names
            bar_name = bar_names.get(k, k)
            self.log(bar_name, v, on_step=True, on_epoch=False, prog_bar=show_on_bar)
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
            distance = self.distance_fn(
                sketch_feature.unsqueeze(0), gallery_features
            ).cpu()
            target = photo_labels.eq(sketch_labels[idx])

            if map_k:
                top_k = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(
                    distance, target, top_k=top_k
                )
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision_at_k[idx] = retrieval_precision(
                distance, target, top_k=p_k
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
