import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from argparse import Namespace
import unittest
import torch
from torch.nn import functional as F
from clip.model import VisionTransformer
from src.semantic_region import (area_prior, content_prior, crop_regions, semantic_weights,
                                 fit_alignment, DenseRegionCapture, SemanticRegionHead, region_losses)

torch.set_num_threads(4)


def loss_args(mode='semantic'):
    return Namespace(region_mode=mode,region_spread_fraction=.5,lambda_descriptor=1.,lambda_region=1.,
                     lambda_gate=.1,lambda_reference=.1,lambda_spread=1.)


class RegionTests(unittest.TestCase):
    def test_overlap_and_crop_geometry(self):
        p=area_prior(224,7,2)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(4)))
        self.assertEqual(p.shape,(4,49))
        self.assertEqual((p[0]>0).sum().item(),16)
        self.assertAlmostEqual(p[0,24].item()/p[0,0].item(),.25)
        x=torch.arange(16.).reshape(1,1,4,4).repeat(1,3,1,1)
        crops=crop_regions(x,2)
        self.assertEqual(crops.shape,(1,4,3,4,4))
        self.assertGreater(crops[0,3].mean(),crops[0,0].mean())

    def test_ink_and_empty_sketch_prior(self):
        mean=torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None]
        std=torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None]
        rgb=torch.ones(2,3,8,8);rgb[0,:,:4,:4]=0
        prior=content_prior((rgb-mean)/std,'sketch',2)
        self.assertTrue(torch.equal(prior[0],torch.tensor([1.,0,0,0])))
        self.assertTrue(torch.equal(prior[1],torch.full((4,),.25)))

    def test_semantic_gate(self):
        full=torch.tensor([[1.,0]])
        crops=torch.tensor([[[1.,0],[0,1],[-1,0],[1,0]]])
        prior=torch.tensor([[.5,.25,.25,0]])
        w=semantic_weights(full,crops,prior,.1)
        self.assertGreater(w[0,0],.99)
        self.assertEqual(w[0,3].item(),0)
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(1)))

    def test_fixed_alignment_recovery(self):
        torch.manual_seed(31)
        q,_=torch.linalg.qr(torch.randn(5,3,dtype=torch.double))
        s=F.normalize(torch.randn(100,3,dtype=torch.double),dim=-1)
        fitted=fit_alignment(s@q.T,s)
        self.assertTrue(torch.allclose(fitted.double(),q,atol=1e-5))
        self.assertTrue(torch.allclose(fitted.T@fitted,torch.eye(3),atol=1e-5))

    def test_capture_does_not_change_cls_and_excludes_prompts(self):
        torch.manual_seed(32)
        visual=VisionTransformer(32,8,64,2,1,16).eval().requires_grad_(False)
        x=torch.randn(2,3,32,32);prompt=torch.randn(3,64,requires_grad=True)
        before=visual(x,prompt,[prompt])
        with DenseRegionCapture(visual) as capture:
            after=visual(x,prompt,[prompt])
        dense=capture.dense_features()
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(dense.shape,(2,16,16))
        dense.square().mean().backward()
        self.assertGreater(prompt.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in visual.parameters()))
        self.assertEqual(len(visual.transformer.resblocks[-1]._forward_pre_hooks),0)
        with self.assertRaises(RuntimeError):
            with DenseRegionCapture(visual):raise RuntimeError('test')
        self.assertEqual(len(visual.transformer.resblocks[-1]._forward_pre_hooks),0)

    def test_initial_descriptor_and_gradient_paths(self):
        torch.manual_seed(33)
        head=SemanticRegionHead(16,32,4,2,8,.5)
        native=F.normalize(torch.randn(4,16),dim=-1)
        dense=torch.randn(4,16,16)
        prior=torch.full((4,4),.25)
        teacher=F.normalize(torch.randn(4,16),dim=-1)
        crops=F.normalize(torch.randn(4,4,16),dim=-1)
        gate=torch.softmax(torch.randn(4,4),dim=-1)
        output=head(native,dense,prior,'semantic')
        self.assertTrue(torch.allclose(output['descriptor'],native,atol=1e-7))
        self.assertTrue(torch.equal(output['correction'],torch.zeros_like(native)))
        self.assertTrue(torch.allclose(output['attention'].sum(-1),torch.ones(4,4)))
        self.assertTrue((output['attention'][:,head.prior==0]==0).all())
        total,_=region_losses(output,teacher,crops,gate,prior,native,loss_args())
        total.backward()
        for p in (head.queries,head.region_adapter.up.weight,head.gate[-1].weight,head.fusion.up.weight):
            self.assertGreater(p.grad.norm().item(),0)
        torch.optim.AdamW(head.parameters(),lr=.01).step()
        self.assertFalse(torch.allclose(head(native,dense,prior,'semantic')['descriptor'],native))

    def test_gate_controls_share_local_loss_weights(self):
        head=SemanticRegionHead(8,32,4,2,4,.5)
        native=F.normalize(torch.randn(3,8),dim=-1);prior=torch.full((3,4),.25)
        output=head(native,torch.randn(3,16,8),prior,'semantic')
        crops=F.normalize(torch.randn(3,4,8),dim=-1)
        _,semantic=region_losses(output,native,crops,torch.tensor([[.7,.1,.1,.1]]).expand(3,-1),prior,native,loss_args())
        _,uniform=region_losses(output,native,crops,prior,prior,native,loss_args('uniform'))
        self.assertEqual(semantic['region'].item(),uniform['region'].item())
        _,global_values=region_losses(output,native,crops,prior,prior,native,loss_args('global'))
        self.assertEqual(global_values['region'].item(),0)
        self.assertEqual(global_values['gate'].item(),0)


if __name__=='__main__':unittest.main()
