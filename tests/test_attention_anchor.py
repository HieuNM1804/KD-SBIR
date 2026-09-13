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

from src.attention_anchor import (token_evidence, AnchorCapture, AnchorRouter, anchor_distribution,
                                   anchor_kd, hellinger_descriptor, fit_vocabulary)


class AttentionAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_capture_matches_attention_and_projected_values(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for layout in (True, False):
                attn = torch.nn.MultiheadAttention(32, 4, batch_first=layout).to(device).eval()
                x = torch.randn(2, 8, 32, device=device, requires_grad=True)
                qkv = x if layout else x.transpose(0, 1)
                _, weights = attn(qkv, qkv, qkv, need_weights=True, average_attn_weights=False)
                tokens, a = token_evidence(attn, qkv, qkv, qkv, 4)
                expected_a = weights[:, :, 0, 1:5].mean(1)
                expected_a = expected_a/expected_a.sum(-1, keepdim=True)
                expected_v = F.linear(F.linear(x[:, 1:5], attn.in_proj_weight[64:], attn.in_proj_bias[64:]),
                                      attn.out_proj.weight, None)
                torch.testing.assert_close(a, expected_a)
                torch.testing.assert_close(tokens, expected_v)
                first = torch.autograd.grad(tokens.square().sum()+a.square().sum(), x, retain_graph=True)[0]
                second = torch.autograd.grad(expected_v.square().sum()+expected_a.square().sum(), x)[0]
                torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    def test_distribution_permutation_descriptor_and_teacher_stop_gradient(self):
        rng = torch.get_rng_state().clone()
        head = AnchorRouter(12, 8, .1, seed=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        tokens = torch.randn(3, 9, 12, requires_grad=True)
        logits = torch.randn(3, 9, requires_grad=True)
        a = logits.softmax(-1)
        p = head(tokens, a)
        order = torch.randperm(9)
        torch.testing.assert_close(p, head(tokens[:,order], a[:,order]))
        torch.testing.assert_close(p.sum(-1), torch.ones(3))
        z = hellinger_descriptor(p)
        torch.testing.assert_close(z.norm(dim=-1), torch.ones(3))
        torch.testing.assert_close(z @ z.T, (p[:,None,:]*p[None,:,:]).sqrt().sum(-1))
        target = torch.randn(3, 8).softmax(-1).requires_grad_()
        loss = anchor_kd(p, target)
        expected = (target*(target.log()-p.log())).sum(-1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(target.grad)
        for gradient in (tokens.grad, logits.grad, head.anchors.grad, head.center.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)
        uniform = anchor_distribution(tokens, a, head.anchors, head.center, .1, 'uniform')
        torch.testing.assert_close(uniform, anchor_distribution(tokens, a.flip(-1), head.anchors, head.center, .1, 'uniform'))
        with self.assertRaises(ValueError): anchor_kd(p, target[:,:4])

    def test_vocabulary_fitting_is_deterministic(self):
        samples = torch.randn(64, 12)
        rng = torch.get_rng_state().clone()
        a, center, info = fit_vocabulary(samples, 8, 5, 1)
        b, other, repeated = fit_vocabulary(samples, 8, 5, 1)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(center,other,atol=0,rtol=0)
        torch.testing.assert_close(a.norm(dim=-1),torch.ones(8))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(info,repeated)
        with self.assertRaises(ValueError): fit_vocabulary(samples[:2],8)

    def test_standalone_training_inference_reload_and_main_baseline(self):
        from clip.model import CLIP
        from src.model import ZS_SBIR
        from src.losses import loss_fn
        from torch.utils.data import TensorDataset, DataLoader
        import pytorch_lightning as pl
        with tempfile.TemporaryDirectory() as tmp:
            teacher_path = Path(tmp)/'teacher.pt';teacher_path.touch()
            args = Namespace(backbone='ViT-B/32', seed=42, n_ctx_visual=3, prompt_depth=3,
                             lambda_domain=0.,lambda_modality=0.,kd_temperature=.07,
                             photo_text_kd_temperature=.15,sketch_text_kd_temperature=.02,
                             teacher_cache_path=str(teacher_path),rebuild_teacher_cache=False,
                             teacher_pretrain_epochs=1,lr=.01,momentum=.9,weight_decay=.0005,
                             retrieval_head='attention_anchor',anchor_count=8,anchor_temperature=.1,
                             anchor_pooling='attention',lambda_anchor=1.)
            tiny = CLIP(32,16,3,64,4,16,128,64,1,1).eval().requires_grad_(False)
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                module = ZS_SBIR(args,['cat','dog']).eval()
            batch = [torch.randn(4,3,16,16),torch.randn(4,3,16,16),torch.randn(4,1024),torch.randn(4,1024),
                     torch.tensor([0,1,0,1]),torch.randn(4,8).softmax(-1),torch.randn(4,8).softmax(-1)]
            with AnchorCapture(module.model.clip_model.visual) as capture:
                features = module(batch[:5])
            ph = module.model.anchor_router(*capture.values[0]);sk = module.model.anchor_router(*capture.values[1])
            torch.testing.assert_close(module.model.extract_feature(batch[0],'photo'),hellinger_descriptor(ph))
            torch.testing.assert_close(module.model.extract_feature(batch[1],'sketch'),hellinger_descriptor(sk))
            expected = .5*(anchor_kd(ph,batch[5])+anchor_kd(sk,batch[6]))
            with patch.object(module,'log'):
                actual = module.training_step(batch,0)
            torch.testing.assert_close(actual,expected)
            actual.backward()
            for group in ('photo_visual_prompt.','sketch_visual_prompt.','anchor_router.'):
                self.assertGreater(sum(p.grad.norm().item() for n,p in module.named_parameters() if group in n and p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in module.model.clip_model.parameters()))
            module.zero_grad(set_to_none=True)
            before = module.model.extract_feature(batch[0],'photo').detach()
            trainer = pl.Trainer(accelerator='cpu', max_epochs=1,logger=False,enable_checkpointing=False,
                                 enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0,num_sanity_val_steps=0)
            trainer.fit(module,DataLoader(TensorDataset(*batch),batch_size=4))
            after = module.model.extract_feature(batch[0],'photo').detach()
            self.assertFalse(torch.equal(before,after))
            path=Path(tmp)/'anchor.ckpt';trainer.save_checkpoint(path)
            saved=torch.load(path,weights_only=False)
            self.assertEqual(saved['experiment_config']['retrieval_head'],'attention_anchor')
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                restored=ZS_SBIR(saved['hyper_parameters']['args'],saved['hyper_parameters']['classnames']).eval()
            restored.load_state_dict(saved['state_dict'],strict=True)
            torch.testing.assert_close(restored.model.extract_feature(batch[0],'photo'),after)
            # Main mode has no new trainable head, and keeps the exact original
            # feature extraction / loss_fn behavior for the baseline.
            baseline_args=deepcopy(args);baseline_args.retrieval_head='main';baseline_args.lambda_domain=3.
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                baseline=ZS_SBIR(baseline_args,['cat','dog']).eval()
            self.assertFalse(hasattr(baseline.model,'anchor_router'))
            torch.testing.assert_close(baseline.model.extract_feature(batch[0],'photo'),
                                       baseline.model.encode_student_image(batch[0],'photo'),atol=0,rtol=0)
            with patch.object(baseline,'log'):actual=baseline.training_step(batch[:5],0)
            expected,_=loss_fn(baseline_args,baseline(batch[:5]))
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)


if __name__ == '__main__':unittest.main()
