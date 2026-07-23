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

from clip import clip
from clip.model import build_model
from src.text_encoder import TextEncoder
from src.losses import loss_fn
from src.teacher_adapters import ModalityAdapters

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


def _build_teacher_adapters(args, teacher):
    if not args.joint_teacher_adapter:
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
    if args.lambda_kd <= 0 and not args.joint_teacher_adapter:
        return None

    print(f"[Teacher] Loading {DFN5B_MODEL} in FP16...")
    teacher = open_clip.create_model(
        DFN5B_MODEL,
        pretrained=DFN5B_PRETRAINED,
        precision="fp16",
        device=device,
    )
    teacher.eval().requires_grad_(False)
    if args.joint_teacher_adapter:
        teacher.text_tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    teacher.output_dim = DFN5B_OUTPUT_DIM
    return teacher


def freeze_clip_except_layer_norm(clip_model):
    clip_model.requires_grad_(False)
    for module in clip_model.modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)


class MultiModalPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx,
        text_width,
        visual_width,
        classnames,
        token_embedding,
        modality,
        seed,
    ):
        super().__init__()
        if n_ctx <= 0:
            raise ValueError(f"n_ctx must be positive, got {n_ctx}.")

        prompt_prefix = (
            "a photo of a" if modality == "photo" else "a sketch of a"
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        context = torch.empty(n_ctx, text_width)
        nn.init.normal_(context, std=0.02, generator=generator)
        with torch.no_grad():
            prefix_tokens = clip.tokenize(prompt_prefix)
            prefix_embedding = token_embedding(prefix_tokens).float()
            initialized_tokens = min(n_ctx, 4)
            context[:initialized_tokens] = prefix_embedding[
                0, 1 : 1 + initialized_tokens
            ]
        self.ctx = nn.Parameter(context)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            self.projection = nn.Linear(text_width, visual_width)

        placeholders = " ".join(["X"] * n_ctx)
        raw_prompts = [
            f"{placeholders} {name.replace('_', ' ')}."
            for name in classnames
        ]
        tokenized_prompts = clip.tokenize(raw_prompts)
        with torch.no_grad():
            embeddings = token_embedding(tokenized_prompts).detach()
        self.register_buffer(
            "tokenized_prompts",
            tokenized_prompts,
            persistent=False,
        )
        self.register_buffer(
            "token_prefix",
            embeddings[:, :1],
            persistent=False,
        )
        self.register_buffer(
            "token_suffix",
            embeddings[:, 1 + n_ctx :],
            persistent=False,
        )

    def text_prompts(self):
        context = self.ctx.to(dtype=self.token_prefix.dtype)
        context = context.unsqueeze(0).expand(
            self.token_prefix.shape[0], -1, -1
        )
        prompts = torch.cat(
            (self.token_prefix, context, self.token_suffix),
            dim=1,
        )
        return self.tokenized_prompts, prompts

    def visual_prompt(self):
        return self.projection(self.ctx)

    def forward(self):
        tokenized_prompts, text_prompts = self.text_prompts()
        return tokenized_prompts, text_prompts, self.visual_prompt()


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
        self.photo_prompt_learner = MultiModalPromptLearner(
            cfg.n_ctx,
            text_width,
            visual_width,
            self.classnames,
            clip_model.token_embedding,
            "photo",
            cfg.seed,
        )
        self.sketch_prompt_learner = MultiModalPromptLearner(
            cfg.n_ctx,
            text_width,
            visual_width,
            self.classnames,
            clip_model.token_embedding,
            "sketch",
            cfg.seed + 1,
        )
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        # The pretrained teacher is reloaded when needed and must not be saved
        # inside every student checkpoint.
        object.__setattr__(self, "_teacher", teacher)
        self.teacher_active = teacher is not None
        self.joint_teacher_adapter = cfg.joint_teacher_adapter
        self.teacher_adapters = _build_teacher_adapters(cfg, teacher)

        self.register_buffer("_teacher_sketch_text", None, persistent=False)
        self.register_buffer("_teacher_photo_text", None, persistent=False)

        print(
            "[Student] modality-specific text prompts projected to visual; "
            f"{cfg.n_ctx} learnable context tokens per modality"
        )
        print(
            "[Relational KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_kd}, "
            f"temperature={cfg.kd_temperature}"
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

    def get_prompt_learner(self, modality):
        if modality == "photo":
            return self.photo_prompt_learner
        return self.sketch_prompt_learner

    def get_student_text_features(self, modality):
        prompt_learner = self.get_prompt_learner(modality)
        tokenized_prompts, text_prompts = (
            prompt_learner.text_prompts()
        )
        return self.text_encoder(tokenized_prompts, text_prompts)

    def encode_student_image(self, image, modality):
        if modality == "photo":
            image_encoder = self.ph_encoder
        else:
            image_encoder = self.sk_encoder
        visual_prompt = self.get_prompt_learner(
            modality
        ).visual_prompt()

        features = image_encoder(image.type(self.dtype), visual_prompt)
        return features / features.norm(dim=-1, keepdim=True)

    def get_logits(self, image, modality):
        text_features = self.get_student_text_features(modality)
        text_features = F.normalize(text_features, dim=-1)
        image_features = self.encode_student_image(image, modality)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features

    def forward(self, x):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, label = x
        photo_logits, photo_features = self.get_logits(
            photo_tensor, "photo"
        )
        sk_logits, sketch_features = self.get_logits(
            sk_tensor, "sketch"
        )

        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            with torch.no_grad():
                teacher_photo_base = self._teacher.encode_image(
                    photo_aug_tensor.half()
                )
                teacher_sketch_base = self._teacher.encode_image(
                    sk_aug_tensor.half()
                )
            teacher_photo_features = self.adapt_teacher_feature(
                teacher_photo_base, "photo"
            )
            teacher_sketch_features = self.adapt_teacher_feature(
                teacher_sketch_base, "sketch"
            )
            if self.joint_teacher_adapter:
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
            sk_logits,
            self.teacher_active,
            self.joint_teacher_adapter,
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
        # Intentionally unused: preserve the historical RNG consumption.
        text_clip_model = _load_clip_model(args.backbone)

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
        
    def configure_optimizers(self):
        adapter_params = (
            [p for p in self.model.teacher_adapters.parameters() if p.requires_grad]
            if self.model.teacher_adapters is not None
            else []
        )
        adapter_param_ids = {id(p) for p in adapter_params}
        student_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in adapter_param_ids
        ]
        param_groups = [{"params": student_params, "lr": self.args.lr}]
        if adapter_params:
            param_groups.append(
                {"params": adapter_params, "lr": self.args.teacher_adapter_lr}
            )
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=self.args.lr,
            weight_decay=1e-3,
            momentum=0.9,
        )
        trainable = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
            if p.requires_grad
        )
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum=0.9, weight_decay=1e-3, "
            f"teacher_adapter_lr={self.args.teacher_adapter_lr if adapter_params else 'off'}, "
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
                "teacher_triplet": "T_TRI",
                "teacher_semantic": "T_SEM",
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
