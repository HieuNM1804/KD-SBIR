import copy
import numpy as np
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
from src.prompt_learner import TextEncoder
from src.losses import loss_fn
from src.teacher_adapters import ModalityAdapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DFN5B teacher loader
# ---------------------------------------------------------------------------
DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
DFN5B_OUTPUT_DIM = 1024


def _load_clip_model(cfg):
    model_path = clip._download(clip._MODELS[cfg.backbone])
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return clip.build_model(state_dict)


def _build_teacher_adapters(args, strong_teacher):
    if not args.joint_teacher_adapter:
        return None

    feature_dim = int(strong_teacher.output_dim)
    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=args.teacher_adapter_bottleneck,
    )
    print(
        "[Teacher Adapter] initialized for joint training "
        f"(feature_dim={feature_dim}, "
        f"bottleneck={args.teacher_adapter_bottleneck})"
    )
    return adapters.to(device=device, dtype=torch.float32)


def _load_teacher():
    print(f"[Teacher] Loading {DFN5B_MODEL} in FP16...")
    teacher = open_clip.create_model(
        DFN5B_MODEL,
        pretrained=DFN5B_PRETRAINED,
        precision="fp16",
        device=device,
    )
    teacher.eval().requires_grad_(False)
    teacher.text_tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    teacher.output_dim = DFN5B_OUTPUT_DIM
    return teacher

# ---------------------------------------------------------------------------

def freeze_all_but_ln(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill, strong_teacher=None
    ):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_ln)
        clip_model_distill.apply(freeze_all_but_ln)
        self.dtype = clip_model.dtype
        prompt_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.sk_prompt = nn.Parameter(torch.randn(cfg.n_ctx, prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(cfg.n_ctx, prompt_dim))
        
        self.ph_encoder = copy.deepcopy(clip_model.visual)
        self.sk_encoder = copy.deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model_distill)
        self.logit_scale = clip_model.logit_scale
        
        self.model_distill = strong_teacher
        self.teacher_active = strong_teacher is not None
        self.joint_teacher_adapter = getattr(cfg, "joint_teacher_adapter", False)
        self.teacher_adapters = _build_teacher_adapters(cfg, strong_teacher)
        self._student_text_token_cache = {}
        self._teacher_text_cache = {}
        print(
            "[Student Prompt] fixed text template, Sketch-LVM-style random "
            f"visual tokens (n_ctx={cfg.n_ctx})"
        )
        print(
            "[Relational KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_kd}, "
            f"temperature={cfg.kd_temperature}"
        )

    def train(self, mode=True):
        super().train(mode)
        if self.model_distill is not None:
            self.model_distill.eval()
        if self.teacher_adapters is not None:
            self.teacher_adapters.train(mode and self.joint_teacher_adapter)
        return self
    
    def teacher_image_input(self, image):
        return image.half()

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

    def get_teacher_text_features(self, classnames):
        cache_key = tuple(classnames)
        if cache_key in self._teacher_text_cache:
            return self._teacher_text_cache[cache_key]

        sketch_prompts = [
            f"a sketch of a {name.replace('_', ' ')}." for name in classnames
        ]
        photo_prompts = [
            f"a photo of a {name.replace('_', ' ')}." for name in classnames
        ]
        tokens = self.model_distill.text_tokenizer(
            sketch_prompts + photo_prompts
        ).to(device)
        with torch.no_grad():
            text_features = F.normalize(
                self.model_distill.encode_text(tokens).float(), dim=-1
            )
        class_count = len(classnames)
        result = (
            text_features[:class_count],
            text_features[class_count:],
        )
        self._teacher_text_cache[cache_key] = result
        return result

    def get_student_text_features(self, classnames):
        cache_key = tuple(classnames)
        if cache_key not in self._student_text_token_cache:
            prompts = [
                f"a photo/sketch of {name.replace('_', ' ')}."
                for name in classnames
            ]
            self._student_text_token_cache[cache_key] = torch.cat(
                [clip.tokenize(prompt) for prompt in prompts]
            )
        text_device = self.text_encoder.positional_embedding.device
        tokens = self._student_text_token_cache[cache_key].to(text_device)
        return self.text_encoder(tokens)

    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            image_encoder = self.ph_encoder
            visual_prompt = self.img_prompt
        else:
            image_encoder = self.sk_encoder
            visual_prompt = self.sk_prompt
            
        logit_scale = self.logit_scale.exp()
        text_features = self.get_student_text_features(classnames)

        image_features = image_encoder(
            img_tensor.type(self.dtype), visual_prompt
        )
        
        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize
        
    def forward(self, x, classnames):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, label = x
        pos_logits, photo_features = self.get_logits(photo_tensor, classnames)
        sk_logits, sketch_features = self.get_logits(
            sk_tensor, classnames, type='sketch'
        )

        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            with torch.no_grad():
                teacher_photo_base = self.model_distill.encode_image(
                    self.teacher_image_input(photo_aug_tensor)
                )
                teacher_sketch_base = self.model_distill.encode_image(
                    self.teacher_image_input(sk_aug_tensor)
                )
            teacher_photo_features = self.adapt_teacher_feature(
                teacher_photo_base, "photo"
            )
            teacher_sketch_features = self.adapt_teacher_feature(
                teacher_sketch_base, "sketch"
            )
            if self.joint_teacher_adapter:
                teacher_sketch_text, teacher_photo_text = (
                    self.get_teacher_text_features(classnames)
                )

        return (
            photo_features,
            sketch_features,
            teacher_photo_features,
            teacher_sketch_features,
            label,
            pos_logits,
            sk_logits,
            self.teacher_active,
            self.joint_teacher_adapter,
            teacher_sketch_text,
            teacher_photo_text,
        )
        
    def extract_feature(self, image, classname, type='photo'):
        _, feature = self.get_logits(image, classnames=classname, type=type)
        return feature


class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = _load_clip_model(args)
        
        clip_model_distill = _load_clip_model(args)
        
        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3

        strong_teacher = _load_teacher()
        self.model = CustomCLIP(
            cfg=args,
            clip_model=clip_model,
            clip_model_distill=clip_model_distill,
            strong_teacher=strong_teacher,
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
        trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"] if p.requires_grad)
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum=0.9, weight_decay=1e-3, "
            f"teacher_adapter_lr={self.args.teacher_adapter_lr if adapter_params else 'off'}, "
            f"trainable_params={trainable:,}"
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data, classname):
        return self.model(data, classname)
    
    def training_step(self, batch, batch_idx):
        features = self.forward(batch, self.classname)
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
            feat = self.model.extract_feature(image_tensor, classname=self.classname, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, classname=self.classname, type='photo')
            self.val_step_outputs_ph.append((feat, label))

    def on_validation_epoch_end(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)

        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])

        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))

        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision = torch.zeros(len(query_feat_all))
        if self.args.dataset == "sketchy_2":
            map_k = 200
            p_k = 200
        else:
            map_k = 0
            if self.args.dataset == "quickdraw":
                p_k = 200
            else:
                p_k = 100

        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            distance = self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_photo_category == category)] = True

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery))
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())

            precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)

        mAP = torch.mean(ap)
        precision = torch.mean(precision)
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        if map_k != 0:
            print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
        else:
            print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
