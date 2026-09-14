"""Teacher-free deployed 512-D descriptor with explicit decision-space KD."""
from argparse import Namespace
import json

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from src.model import CustomCLIP, _load_clip_model
from src.semantic_region import SemanticRegionHead
from src.region_correspondence import correspondence_losses, retrieval_metrics


class CorrespondenceModule(LightningModule):
    def __init__(self, args, classnames, backbone=None):
        super().__init__()
        self.args = args
        backbone = backbone if backbone is not None else _load_clip_model(args.backbone)
        self.model = CustomCLIP(args, backbone, classnames, teacher=None)
        self.model.teacher_active = False
        width = backbone.visual.proj.shape[1]
        grid = int((backbone.visual.positional_embedding.shape[0] - 1) ** .5)
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 810)
            self.model.region_head = SemanticRegionHead(width, args.max_size, grid,
                                                       args.region_grid, args.region_bottleneck, args.region_beta)
        if not args.region_train_prompts:
            self.model.photo_visual_prompt.requires_grad_(False)
            self.model.sketch_visual_prompt.requires_grad_(False)
        self.save_hyperparameters({'args': dict(vars(args)), 'classnames': list(classnames)})
        self.validation_outputs = {'sketch': [], 'photo': []}
        print(f'[Correspondence] protocol={args.retrieval_protocol}, mode={args.correspondence_mode}, '
              f'student_dim={width}, teacher_dim=1024, alignment=None, teacher_in_state=False')

    def configure_optimizers(self):
        groups = [{'params': [p for p in self.model.region_head.parameters() if p.requires_grad],
                   'lr': self.args.region_head_lr}]
        prompts = [p for n, p in self.model.named_parameters() if p.requires_grad and 'region_head.' not in n]
        if prompts:
            groups.append({'params': prompts, 'lr': self.args.lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step_size,
                                                    gamma=self.args.scheduler_gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        outputs = {m: self.model.encode_region_image(batch[m], m) for m in ('sketch', 'photo')}
        loss, values = correspondence_losses(outputs['sketch'], outputs['photo'], batch, self.args)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite correspondence loss')
        size = len(batch['sketch'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=size, prog_bar=True)
        for key, value in values.items():
            self.log('train_' + key, value, on_step=False, on_epoch=True, batch_size=size,
                     prog_bar=key in ('teacher_acceptance', 'evidence_fraction'))
        for modality, output in outputs.items():
            drift = 1 - F.cosine_similarity(output['descriptor'].float(), output['native'].float()).mean()
            self.log(modality + '_descriptor_drift', drift, on_step=False, on_epoch=True, batch_size=size)
            self.log(modality + '_correction_norm', output['correction'].float().norm(dim=-1).mean(),
                     on_step=False, on_epoch=True, batch_size=size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        modality = 'sketch' if dataloader_idx == 0 else 'photo'
        images, labels = batch
        output = self.model.encode_region_image(images, modality)
        self.validation_outputs[modality].append((output['descriptor'].detach().float().cpu(),
                                                 output['native'].detach().float().cpu(), labels.cpu()))

    def on_validation_epoch_end(self):
        if not all(self.validation_outputs.values()):
            return
        data = {m: [torch.cat([v[i] for v in parts]) for i in range(3)]
                for m, parts in self.validation_outputs.items()}
        results = {}
        for name, index in [('deployed', 0), ('native', 1)]:
            results[name] = retrieval_metrics(data['sketch'][index].to(self.device), data['photo'][index].to(self.device),
                                              data['sketch'][2], data['photo'][2], self.args.retrieval_protocol,
                                              self.args.precision_k, self.args.metric_chunk_size, self.args.fg_gallery)
        for key in ('primary', 'mAP', 'precision', 'Acc1', 'Acc5'):
            self.log('seen_val_' + key, results['deployed'][key], prog_bar=key == 'primary')
            self.log('seen_native_' + key, results['native'][key])
        print('[Correspondence Seen Validation]', json.dumps({'epoch': self.current_epoch, 'results': results}), flush=True)
        for m in self.validation_outputs:
            self.validation_outputs[m].clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['experiment_config'] = {'retrieval_head': 'region_correspondence',
                                          'args': dict(vars(self.args)),
                                          'teacher_target_metadata': getattr(self.args, 'teacher_target_metadata', None),
                                          'source_sha256': getattr(self.args, 'training_source_sha256', {})}


def load_correspondence_checkpoint(path, device='cpu'):
    """Load backbone/head/prompts from a trusted checkpoint, no weights download."""
    from clip.model import build_model
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved.get('experiment_config', {})
    if config.get('retrieval_head') != 'region_correspondence':
        raise ValueError('Expected a region-correspondence checkpoint')
    prefix = 'model.clip_model.'
    weights = {k[len(prefix):]: v for k, v in saved['state_dict'].items() if k.startswith(prefix)}
    if not weights:
        raise ValueError('Checkpoint has no frozen backbone')
    with torch.random.fork_rng():
        backbone = build_model(dict(weights))
        if weights['visual.conv1.weight'].dtype == torch.float32:
            backbone.float()
        module = CorrespondenceModule(Namespace(**config['args']), saved['hyper_parameters']['classnames'], backbone)
    module.load_state_dict(saved['state_dict'], strict=True)
    return module.to(device).eval().requires_grad_(False)
