import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from argparse import Namespace
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.mask_guided import (TeacherAttention, make_masks, apply_mask, RetrievalProjection,
                              embedding_loss, response_terms)


class MaskGuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_attention_matches_mha_and_hook_is_removed(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            for layout in (True,False):
                mha=torch.nn.MultiheadAttention(32,4,batch_first=layout).to(device).eval()
                visual=Namespace(transformer=Namespace(resblocks=[Namespace(attn=mha)]),
                                 positional_embedding=torch.empty(5,32))
                x=torch.randn(2,7,32,device=device);x=x if layout else x.transpose(0,1)
                with TeacherAttention(visual) as capture:
                    _,a=mha(x,x,x,average_attn_weights=False)
                expected=a[:,:,0,1:5].mean(1);expected/=expected.sum(-1,keepdim=True)
                torch.testing.assert_close(capture.values[0],expected)
                self.assertFalse(mha._forward_pre_hooks)
                with self.assertRaises(RuntimeError):
                    with TeacherAttention(visual):raise RuntimeError('fixture')
                self.assertFalse(mha._forward_pre_hooks)

    def test_masks_equal_area_deterministic_path_sampling_and_image_alignment(self):
        scores=torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]])
        rng=torch.get_rng_state().clone()
        _,guided,random=make_masks(scores,['photo/cat/a','sketch/cat/a'],2,.25,42)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(guided[0].flatten().nonzero().item(),3)
        self.assertTrue((guided.sum((-1,-2))==random.sum((-1,-2))).all())
        _,g2,r2=make_masks(scores.flip(0),['sketch/cat/a','photo/cat/a'],2,.25,42)
        torch.testing.assert_close(random,r2.flip(0));torch.testing.assert_close(guided,g2.flip(0))
        images=torch.ones(2,3,8,8,requires_grad=True)
        result=apply_mask(images,guided)
        self.assertTrue((result[0,:,-4:,-4:]==0).all())
        self.assertEqual((result==0).sum().item(),2*3*16)
        result.sum().backward()
        self.assertTrue(torch.equal(images.grad,result.detach()))
        torch.testing.assert_close(apply_mask(images[0],guided[0]),result[0])
        self.assertTrue((images.detach()==1).all())

    def test_response_formula_stopgrad_and_no_forced_invariance(self):
        head=RetrievalProjection(8,12)
        x=torch.randn(3,8,requires_grad=True);xm=torch.randn(3,8,requires_grad=True)
        z,zm=head(x),head(xm)
        t=torch.randn(3,12,requires_grad=True);tm=torch.randn(3,12,requires_grad=True)
        loss,stats=response_terms(z,zm,t,tm)
        dt=F.normalize(t,dim=-1)-F.normalize(tm,dim=-1)
        torch.testing.assert_close(loss,((z-zm)-dt).square().sum(-1).mean())
        (loss+embedding_loss(z,t)).backward()
        self.assertIsNone(t.grad);self.assertIsNone(tm.grad)
        self.assertGreater(x.grad.norm(),0);self.assertGreater(head.weight.grad.norm(),0)
        exact,_=response_terms(F.normalize(t.detach(),dim=-1),F.normalize(tm.detach(),dim=-1),t,tm)
        self.assertLess(exact.item(),1e-12)
        invariant,_=response_terms(z,z,t,tm)
        self.assertGreater(invariant.item(),0)
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_half_precision_teacher_capture_and_response_on_cuda(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        visual=VisionTransformer(image_size=8,patch_size=4,width=64,layers=2,heads=4,
                                 mlp_ratio=2,output_dim=1024).cuda().half().eval().requires_grad_(False)
        controller=TeacherPromptController(visual,2,2,.02,42).eval().requires_grad_(False)
        images=torch.randn(2,3,8,8,device='cuda',dtype=torch.float16)
        with torch.no_grad(),TeacherAttention(visual) as capture:teacher=controller(images,'sketch')
        _,guided,_=make_masks(capture.values[0],['a','b'],2,.25,42)
        with torch.no_grad():masked=controller(apply_mask(images,guided),'sketch')
        head=RetrievalProjection(32).cuda()
        x=torch.randn(2,32,device='cuda',dtype=torch.float16,requires_grad=True)
        xm=torch.randn_like(x,requires_grad=True)
        z,zm=head(x),head(xm)
        loss,_=response_terms(z,zm,teacher,masked)
        (loss+embedding_loss(z,teacher)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(head.weight.grad.norm().item(),0)
        self.assertTrue(torch.isfinite(capture.values[0]).all())

    def test_real_training_diagnostics_inference_reload_and_main(self):
        from clip.model import CLIP
        from src.model import ZS_SBIR
        from src.losses import loss_fn
        from src.mask_guided_diagnostics import MaskDiagnostics
        from torch.utils.data import TensorDataset,DataLoader
        import pytorch_lightning as pl
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);teacher=root/'teacher.pt';teacher.touch()
            args=Namespace(backbone='ViT-B/32',seed=42,n_ctx_visual=3,prompt_depth=3,
                           lambda_domain=0.,lambda_modality=0.,kd_temperature=.07,
                           photo_text_kd_temperature=.15,sketch_text_kd_temperature=.02,
                           teacher_cache_path=str(teacher),rebuild_teacher_cache=False,teacher_pretrain_epochs=1,
                           lr=.01,momentum=.9,weight_decay=.0005,retrieval_head='mask_guided',mask_strategy='attention',
                           lambda_embedding=1.,lambda_response=1.,mask_gradient_audit=True,dataset='sketchy_1')
            tiny=CLIP(32,16,3,64,4,16,128,64,1,1).eval().requires_grad_(False)
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                module=ZS_SBIR(args,['cat','dog']).eval()
            ph,sk=torch.randn(4,3,16,16),torch.randn(4,3,16,16)
            masks=torch.zeros(4,4,4,dtype=torch.bool);masks[:,:2,:2]=True
            batch=[ph,sk,F.normalize(torch.randn(4,1024),dim=-1),F.normalize(torch.randn(4,1024),dim=-1),
                   torch.tensor([0,1,0,1]),apply_mask(ph,masks),apply_mask(sk,masks),
                   F.normalize(torch.randn(4,1024),dim=-1),F.normalize(torch.randn(4,1024),dim=-1)]
            features=module(batch[:5]);expected=0
            for feat,full,masked,tm,mod in ((features[0],batch[2],batch[5],batch[7],'photo'),(features[1],batch[3],batch[6],batch[8],'sketch')):
                z=module.model.retrieval_projection(feat);zm=module.model.extract_feature(masked,mod)
                torch.testing.assert_close(z,module.model.extract_feature(ph if mod=='photo' else sk,mod))
                expected+=.5*(embedding_loss(z,full)+response_terms(z,zm,full,tm)[0])
            with patch.object(module,'log'):actual=module.training_step(batch,0)
            torch.testing.assert_close(actual,expected)
            actual.backward()
            for tag in ('photo_visual_prompt.','sketch_visual_prompt.','retrieval_projection.'):
                self.assertGreater(sum(p.grad.norm().item() for n,p in module.named_parameters() if tag in n and p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in module.model.clip_model.parameters()))
            module.zero_grad(set_to_none=True);module.mask_gradient_rows=[]
            before=module.model.extract_feature(ph,'photo').detach()
            diagnostic=MaskDiagnostics(root/'diagnostics')
            trainer=pl.Trainer(accelerator='cpu',max_epochs=1,logger=False,enable_checkpointing=False,
                               enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
                               callbacks=[diagnostic])
            train=DataLoader(TensorDataset(*batch),batch_size=4)
            # Main metric helper is exercised separately in the repository; here
            # use enough gallery items for P@100 to check real Lightning lifecycle.
            labels=torch.arange(104)%2
            val=[DataLoader(TensorDataset(sk.repeat(26,1,1,1),labels),batch_size=26),
                 DataLoader(TensorDataset(ph.repeat(26,1,1,1),labels),batch_size=26)]
            trainer.validate(module,val,verbose=False)
            self.assertTrue((root/'diagnostics/initial_validation.json').exists())
            initial_bytes=(root/'diagnostics/initial_validation.json').read_bytes()
            trainer.fit(module,train,val)
            self.assertEqual((root/'diagnostics/initial_validation.json').read_bytes(),initial_bytes)
            self.assertFalse(torch.equal(before,module.model.extract_feature(ph,'photo')))
            row=list(csv.DictReader((root/'diagnostics/epochs.csv').read_text().splitlines()))[0]
            self.assertEqual(int(row['gallery']),104);self.assertEqual(int(row['epoch']),0)
            self.assertAlmostEqual(float(row['total_loss']),float(row['embedding_loss'])+float(row['response_loss']),places=5)
            self.assertEqual(len(list(csv.DictReader((root/'diagnostics/gradient_audit.csv').read_text().splitlines()))),3)
            self.assertTrue((root/'diagnostics/training_diagnostics.png').exists())
            path=root/'model.ckpt';trainer.save_checkpoint(path)
            saved=torch.load(path,weights_only=False)
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                restored=ZS_SBIR(saved['hyper_parameters']['args'],saved['hyper_parameters']['classnames']).eval()
            restored.load_state_dict(saved['state_dict'],strict=True)
            torch.testing.assert_close(restored.model.extract_feature(ph,'photo'),module.model.extract_feature(ph,'photo'))
            # Embedding-only uses exactly the same inference head and needs no masked forwards.
            args.lambda_response=0
            with patch.object(module,'log'):only=module.training_step(batch[:5],1)
            expected=.5*(embedding_loss(module.model.extract_feature(ph,'photo'),batch[2])+embedding_loss(module.model.extract_feature(sk,'sketch'),batch[3]))
            torch.testing.assert_close(only,expected)
            baseline=deepcopy(args);baseline.retrieval_head='main';baseline.lambda_domain=3
            with patch('src.model._load_clip_model',return_value=deepcopy(tiny)):
                main=ZS_SBIR(baseline,['cat','dog']).eval()
            self.assertFalse(hasattr(main.model,'retrieval_projection'))
            torch.testing.assert_close(main.model.extract_feature(ph,'photo'),main.model.encode_student_image(ph,'photo'),rtol=0,atol=0)
            with patch.object(main,'log'):loss=main.training_step(batch[:5],0)
            torch.testing.assert_close(loss,loss_fn(baseline,main(batch[:5]))[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
