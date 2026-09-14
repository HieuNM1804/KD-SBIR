"""Load a deployed region descriptor from checkpoint, with no teacher or downloads."""
from argparse import Namespace
from unittest.mock import patch
import torch


def load_region_checkpoint(path, device='cpu'):
    from clip.model import build_model
    import src.model as source
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved.get('experiment_config', {})
    if config.get('retrieval_head') != 'semantic_region':
        raise ValueError('Expected semantic-region checkpoint')
    args = Namespace(**config['args'])
    prefix = 'model.clip_model.'
    backbone = {k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)}
    if not backbone:
        raise ValueError('Checkpoint has no frozen CLIP state')
    with torch.random.fork_rng(), patch.object(source, '_load_teacher', return_value=None), \
         patch.object(source, '_load_clip_model', side_effect=lambda _:build_model(dict(backbone))):
        module = source.ZS_SBIR(args, saved['hyper_parameters']['classnames'])
    module.load_state_dict(saved['state_dict'], strict=True)
    return module.to(device).eval().requires_grad_(False)
