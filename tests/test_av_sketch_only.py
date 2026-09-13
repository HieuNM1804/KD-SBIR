import importlib.util
import inspect
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location('sketch_only_cell',
    Path(__file__).resolve().parents[1] / 'test/kaggle_av_sketch_only_cell.py')
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)
exec(compile(cell.RUNNER_SOURCE, 'sketch_only_runner', 'exec'), cell.__dict__)


class SketchOnlyTests(unittest.TestCase):
    def test_reject_ambiguous_training_method(self):
        for source in ('def training_step(self, batch, i):\n    return 0',
                       'def training_step(self, batch, i):\n    av_loss = 1\n    av_loss = 2'):
            with self.assertRaises(ValueError):
                cell.sketch_only_step(source, {})

    def test_real_sketch_loss_and_gradient_routing(self):
        from clip.model import CLIP
        import src.model as production
        from src.losses import loss_fn
        from src.attention_output_kd import PatchOutputCapture, feature_cosine_kd
        torch.set_num_threads(1)
        torch.manual_seed(42)
        tiny = CLIP(32, 16, 3, 64, 4, 16, 128, 64, 1, 1).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            teacher = Path(tmp) / 'teacher.pt'
            teacher.touch()
            args = Namespace(backbone='ViT-B/32', seed=42, n_ctx_visual=3, prompt_depth=3,
                             lambda_domain=3., lambda_modality=1., kd_temperature=.07,
                             photo_text_kd_temperature=.15, sketch_text_kd_temperature=.02,
                             teacher_cache_path=str(teacher), rebuild_teacher_cache=False,
                             teacher_pretrain_epochs=1, lambda_av=1., lambda_global_feature=0.,
                             av_objective='cosine', av_modality='sketch_only', av_temperature=.07,
                             lr=.01, momentum=.9, weight_decay=.0005)
            with patch('src.model._load_clip_model', return_value=tiny):
                module = production.ZS_SBIR(args, ['cat', 'dog', 'bird']).eval()
            for mod in ('photo', 'sketch'):
                setattr(module.model, '_student_' + mod + '_text_features', F.normalize(torch.randn(3, 32), dim=-1))
                setattr(module.model, '_teacher_' + mod + '_text', F.normalize(torch.randn(3, 1024), dim=-1))
            batch = [torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16),
                     torch.randn(4, 1024), torch.randn(4, 1024), torch.tensor([0, 1, 2, 0]),
                     torch.randn(4, 1280), torch.randn(4, 1280)]
            changed, _ = cell.sketch_only_step(inspect.getsource(production.ZS_SBIR.training_step), vars(production))
            with patch.object(module, 'log'):
                actual = changed(module, batch, 0)
                direct = module.training_step(batch, 0)
            torch.testing.assert_close(direct, actual)
            checkpoint = {}
            module.on_save_checkpoint(checkpoint)
            self.assertEqual(checkpoint['experiment_config']['av_modality'], 'sketch_only')
            self.assertEqual(checkpoint['experiment_config']['args']['av_modality'], 'sketch_only')
            with PatchOutputCapture(module.model.clip_model.visual) as capture:
                features = module(batch[:5])
            main, _ = loss_fn(args, features)
            sketch = .5 * feature_cosine_kd(capture.values[1], batch[6], module.model.av_projector)
            torch.testing.assert_close(actual, main + sketch)
            named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            params = [p for n, p in named]
            av_grads = torch.autograd.grad(sketch, params, allow_unused=True, retain_graph=True)
            for (name, _), grad in zip(named, av_grads):
                if 'photo_visual_prompt.' in name:
                    self.assertTrue(grad is None or not grad.any())
            for group in ('sketch_visual_prompt.', 'av_projector.'):
                self.assertGreater(sum(g.abs().sum().item() for (n, _), g in zip(named, av_grads)
                                       if group in n and g is not None), 0)
            expected_grads = torch.autograd.grad(main + sketch, params)
            actual_grads = torch.autograd.grad(actual, params)
            direct_grads = torch.autograd.grad(direct, params)
            for a, b, c in zip(actual_grads, expected_grads, direct_grads):
                torch.testing.assert_close(a, b)
                torch.testing.assert_close(c, b)
            # Photo AV targets must have no influence on this objective.
            alternate = list(batch)
            alternate[5] = torch.full_like(batch[5], float('nan'))
            with patch.object(module, 'log'):
                unaffected = changed(module, alternate, 0)
            torch.testing.assert_close(unaffected, actual)


if __name__ == '__main__':
    unittest.main()
