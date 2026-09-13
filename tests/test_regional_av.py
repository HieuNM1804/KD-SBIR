import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from PIL import Image

from src.attention_output_kd import (region_patch_weights, patch_attention_output,
                                     PatchOutputCapture, regional_cosine_kd, feature_cosine_kd)


class RegionalAVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_normalized_regions_cover_mismatched_patch_grids(self):
        for side in (7, 16):
            weights = region_patch_weights(side**2, 2)
            torch.testing.assert_close(weights.sum(0), torch.ones(side**2))
            torch.testing.assert_close(weights.sum(1), torch.full((4,), side**2 / 4))
            torch.testing.assert_close(weights[0, 0], torch.tensor(1.))
            torch.testing.assert_close(weights[-1, -1], torch.tensor(1.))
        # Central student patch overlaps all four image quadrants equally.
        torch.testing.assert_close(region_patch_weights(49, 2)[:, 24], torch.full((4,), .25))
        for count, grid in ((5, 2), (49, 8), (49, 0)):
            with self.assertRaises(ValueError):
                region_patch_weights(count, grid)

    def test_explicit_contributions_forward_and_gradients(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            attn = torch.nn.MultiheadAttention(32, 4, batch_first=True).to(device).eval()
            x = torch.randn(2, 53, 32, device=device, requires_grad=True)
            _, a = attn(x, x, x, need_weights=True, average_attn_weights=False)
            v = F.linear(x, attn.in_proj_weight[64:], attn.in_proj_bias[64:])
            v = v.reshape(2, 53, 4, 8).transpose(1, 2)[:, :, 1:50]
            # Independent geometric reference in integer units: patch width=2,
            # region width=7 in a 14x14 square.
            expected = []
            for ry in range(2):
                for rx in range(2):
                    contribution = torch.zeros(2, 4, 8, device=device)
                    for y in range(7):
                        for z in range(7):
                            oy = max(0, min(2*y+2, 7*ry+7)-max(2*y, 7*ry))/2
                            ox = max(0, min(2*z+2, 7*rx+7)-max(2*z, 7*rx))/2
                            j = 7*y+z
                            contribution = contribution + ox*oy*a[:, :, 0, 1+j, None]*v[:, :, j]
                    expected.append(F.linear(contribution.flatten(1), attn.out_proj.weight, None))
            expected = torch.stack(expected, 1)
            actual = patch_attention_output(attn, x, x, x, 49, region_grid=2)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            g1 = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
            g2 = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
            torch.testing.assert_close(g1, g2, atol=2e-6, rtol=2e-5)
            global_av = patch_attention_output(attn, x, x, x, 49)
            torch.testing.assert_close(actual.sum(1), global_av, atol=2e-6, rtol=2e-5)
            one = patch_attention_output(attn, x, x, x, 49, region_grid=1)
            torch.testing.assert_close(one[:, 0], global_av)
            target = torch.randn(2, 4, 48, device=device, requires_grad=True)
            projector = torch.nn.Linear(32, 48, bias=False).to(device)
            loss = regional_cosine_kd(actual, target, projector)
            reference = torch.stack([feature_cosine_kd(actual[:, r], target[:, r], projector) for r in range(4)]).mean()
            torch.testing.assert_close(loss, reference)
            loss.backward()
            self.assertIsNone(target.grad)
            self.assertGreater(projector.weight.grad.norm().item(), 0)
            with self.assertRaises(ValueError):
                regional_cosine_kd(actual, target[:, :2], projector)

    def test_production_loss_gradient_routing_and_checkpoint(self):
        from clip.model import CLIP
        from src.model import ZS_SBIR
        from src.losses import loss_fn
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, TensorDataset
        with tempfile.TemporaryDirectory() as tmp:
            teacher = Path(tmp)/'teacher.pt'; teacher.touch()
            args = Namespace(backbone='ViT-B/32', seed=42, n_ctx_visual=3, prompt_depth=3,
                             lambda_domain=3., lambda_modality=1., kd_temperature=.07,
                             photo_text_kd_temperature=.15, sketch_text_kd_temperature=.02,
                             teacher_cache_path=str(teacher), rebuild_teacher_cache=False,
                             teacher_pretrain_epochs=1, lambda_av=1., lambda_global_feature=0.,
                             av_objective='regional_cosine', av_modality='sketch_only', av_region_grid=2,
                             lr=.01, momentum=.9, weight_decay=.0005)
            tiny = CLIP(32, 16, 3, 64, 4, 16, 128, 64, 1, 1).eval().requires_grad_(False)
            with patch('src.model._load_clip_model', return_value=tiny):
                module = ZS_SBIR(args, ['cat', 'dog', 'bird']).eval()
            for mod in ('photo', 'sketch'):
                setattr(module.model, '_student_'+mod+'_text_features', F.normalize(torch.randn(3, 32), dim=-1))
                setattr(module.model, '_teacher_'+mod+'_text', F.normalize(torch.randn(3, 1024), dim=-1))
            batch = [torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16),
                     torch.randn(4, 1024), torch.randn(4, 1024), torch.tensor([0, 1, 2, 0]),
                     torch.empty(4, 0), torch.randn(4, 4, 1280)]
            reference_features = module(batch[:5])
            with PatchOutputCapture(module.model.clip_model.visual, region_grid=2) as cap:
                features = module(batch[:5])
            for i in (0, 1):
                torch.testing.assert_close(reference_features[i], features[i], atol=0, rtol=0)
            main, _ = loss_fn(args, features)
            av = .5*regional_cosine_kd(cap.values[1], batch[6], module.model.av_projector)
            named = [(n,p) for n,p in module.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(av, [p for _,p in named], retain_graph=True, allow_unused=True)
            self.assertTrue(all(g is None for (n,p),g in zip(named,grads) if 'photo_visual_prompt.' in n))
            for group in ('sketch_visual_prompt.', 'av_projector.'):
                self.assertGreater(sum(g.norm().item() for (n,p),g in zip(named,grads) if group in n and g is not None), 0)
            with patch.object(module, 'log'):
                total = module.training_step(batch, 0)
            torch.testing.assert_close(total, main+av)
            manual = deepcopy(module)
            with patch.object(manual, 'log'):
                manual_loss = manual.training_step(batch, 0)
            opt = manual.configure_optimizers()[0][0]
            manual_loss.backward(); opt.step()
            trainer = pl.Trainer(accelerator='cpu', max_epochs=1, logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False,
                                 enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
            trainer.fit(module, DataLoader(TensorDataset(*batch), batch_size=4))
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, manual.state_dict()[key])
            path = Path(tmp)/'regional.ckpt'; trainer.save_checkpoint(path)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['experiment_config']['av_objective'], 'regional_cosine')
            self.assertEqual(saved['experiment_config']['av_region_grid'], 2)
            self.assertEqual(saved['global_step'], 1)
            # The checkpoint reconstructs the regional objective without runtime patching.
            with patch('src.model._load_clip_model', return_value=deepcopy(tiny)):
                restored = ZS_SBIR(saved['hyper_parameters']['args'], saved['hyper_parameters']['classnames'])
            restored.load_state_dict(saved['state_dict'], strict=True)
            self.assertEqual(restored.av_region_grid, 2)

    def test_regional_cache_build_load_and_reject_wrong_definition(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION
        from src.attention_output_cache import prepare_av_cache
        from src.dataset import TrainDataset
        teacher = torch.nn.Module()
        teacher.visual = VisionTransformer(8, 4, 1280, 1, 20, 1, output_dim=1024).eval()
        prompts = TeacherPromptController(teacher.visual, 2, 1, .02, 42)
        meta = dict(format_version=TEACHER_CACHE_FORMAT_VERSION, dataset='sketchy_2', max_size=8,
                    teacher_model=DFN5B_MODEL, teacher_pretrained=DFN5B_PRETRAINED, pretrain_epochs=1,
                    teacher_n_ctx_visual=2, teacher_prompt_depth=1, teacher_prompt_std=.02, teacher_prompt_seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mod in ('photo', 'sketch'):
                (root/mod/'cat').mkdir(parents=True)
                for i in range(2): Image.new('RGB', (8,8), (i*80,20,50)).save(root/mod/'cat'/f'{i}.png')
            args = Namespace(root=tmp, dataset='sketchy_2', seed=42, max_size=8, workers=0,
                             teacher_cache_path=str(root/'teacher.pt'), av_cache_path='', av_teacher_batch_size=2,
                             av_objective='regional_cosine', av_modality='sketch_only', av_region_grid=2)
            torch.save(dict(metadata=meta, teacher_prompt_state_dict=prompts.state_dict()), args.teacher_cache_path)
            ds = TrainDataset(args)
            with patch('open_clip.create_model', side_effect=lambda *a,**kw: teacher.to(kw['device']).eval().requires_grad_(False)):
                prepare_av_cache(args, ds)
            self.assertEqual(ds.attention_sketch_features.shape, (2,4,1280))
            self.assertEqual(ds.attention_photo_features.shape, (2,0))
            self.assertEqual(ds[0][5].shape, (0,))
            self.assertEqual(ds[0][6].shape, (4,1280))
            restored = TrainDataset(args)
            with patch('open_clip.create_model', side_effect=AssertionError('must reuse cache')):
                prepare_av_cache(args, restored)
            torch.testing.assert_close(ds.attention_sketch_features, restored.attention_sketch_features)
            for grid in (1, 3):
                args.av_region_grid = grid
                with self.assertRaises(ValueError): prepare_av_cache(args, restored)
            args.av_objective = 'cosine'
            with self.assertRaises(ValueError): prepare_av_cache(args, restored)


if __name__ == '__main__':
    unittest.main()
