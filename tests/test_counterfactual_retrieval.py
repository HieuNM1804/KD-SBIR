import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw
from src.counterfactual_retrieval_kd import (
    CLIP_MEAN, CLIP_STD, avcrd_loss, erase_ink_box, erase_ink_batch,
    patch_ink_mass, attention_proposals, random_mass_matched_proposals,
)
from src.attention_output_kd import PatchContributionCapture, patch_attention_contributions, patch_attention_output


def normalized(rgb):
    return (rgb-torch.tensor(CLIP_MEAN).view(3,1,1))/torch.tensor(CLIP_STD).view(3,1,1)


def args_for(root,max_size=56):
    return SimpleNamespace(root=str(root),dataset='sketchy_2',seed=42,max_size=max_size,
        teacher_cache_path=str(root/'teacher.pt'),avcrd_cache_path=str(root/'cf.pt'),
        workers=0,avcrd_teacher_batch_size=2,avcrd_window=1,avcrd_proposals=2,
        avcrd_photo_bank=4,avcrd_audit_per_class=0,avcrd_ink_threshold=.08,
        avcrd_ink_softness=.12,avcrd_diagnostic_examples=2)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_patch_contributions_sum_and_attention_match(self):
        for layout in (True,False):
            attn=torch.nn.MultiheadAttention(32,4,batch_first=layout).eval()
            x=torch.randn(2,8,32,requires_grad=True);q=x if layout else x.transpose(0,1)
            contributions,attention=patch_attention_contributions(attn,q,q,q,5)
            torch.testing.assert_close(contributions.sum(1),patch_attention_output(attn,q,q,q,5),atol=1e-6,rtol=1e-5)
            _,explicit=attn(q,q,q,need_weights=True,average_attn_weights=False)
            torch.testing.assert_close(attention,explicit[:,:,0,1:6].mean(1))
            grad1=torch.autograd.grad(contributions.sum(1).square().sum(),x,retain_graph=True)[0]
            grad2=torch.autograd.grad(patch_attention_output(attn,q,q,q,5).square().sum(),x)[0]
            torch.testing.assert_close(grad1,grad2,atol=1e-6,rtol=1e-5)

    def test_white_background_and_batched_erasure_replay(self):
        rgb=torch.ones(3,16,16);rgb[:,2:9,3:12]=0
        image=normalized(rgb);box=torch.tensor([4,4,10,10])
        erased=erase_ink_box(image,box)
        batched=erase_ink_batch(image[None].repeat(2,1,1,1),box[None].repeat(2,1))
        torch.testing.assert_close(erased,batched[0])
        restored=erased*torch.tensor(CLIP_STD).view(3,1,1)+torch.tensor(CLIP_MEAN).view(3,1,1)
        torch.testing.assert_close(restored[:,4:9,4:10],torch.ones(3,5,6),atol=1e-6,rtol=0)
        torch.testing.assert_close(erased[:,:4],image[:,:4])
        white=normalized(torch.ones(3,16,16))
        torch.testing.assert_close(erase_ink_box(white,box),white)
        with self.assertRaises(ValueError):erase_ink_batch(image[None],torch.tensor([[0,0,20,20]]))

    def test_proposals_and_mass_matched_random_reproducibility(self):
        rgb=torch.ones(3,56,56);rgb[:,3:53:3,3:53]=0
        image=normalized(rgb);ink=patch_ink_mass(image,4)
        saliency=torch.arange(16).float().view(4,4)+1
        proposals=attention_proposals(saliency,ink,1,3,56)
        controls=random_mass_matched_proposals(ink,proposals,1,56,torch.Generator().manual_seed(7))
        replay=random_mass_matched_proposals(ink,proposals,1,56,torch.Generator().manual_seed(7))
        self.assertEqual(controls,replay)
        self.assertEqual(len(proposals),3)
        for box,_,_ in controls:self.assertEqual((box[2]-box[0])*(box[3]-box[1]),14**2)

    def test_identical_geometry_zero_and_gradients_detach_teacher(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            sk=torch.randn(4,16,device=device,requires_grad=True)
            masked=torch.randn(4,16,device=device,requires_grad=True)
            ph=torch.randn(5,16,device=device,requires_grad=True)
            teacher=[F.pad(x.detach(),(0,8)).requires_grad_() for x in (sk,masked,ph)]
            loss,stats=avcrd_loss(sk,masked,ph,*teacher)
            self.assertLess(loss.item(),1e-5)
            bad=[torch.randn_like(x,requires_grad=True) for x in teacher]
            loss,_=avcrd_loss(sk,masked,ph,*bad)
            loss.backward()
            for x in (sk,masked,ph):self.assertGreater(x.grad.norm().item(),0)
            for x in bad:self.assertIsNone(x.grad)
            self.assertTrue(torch.isfinite(loss))

    def test_zero_teacher_response_and_zero_student_response_are_finite(self):
        sk=torch.randn(4,16,requires_grad=True);ph=torch.randn(5,16,requires_grad=True)
        loss,_=avcrd_loss(sk,sk,ph,sk.detach(),sk.detach(),ph.detach())
        self.assertLess(loss.item(),1e-5);loss.backward()
        self.assertTrue(torch.isfinite(sk.grad).all());self.assertTrue(torch.isfinite(ph.grad).all())
        with self.assertRaises(ValueError):avcrd_loss(sk,sk,ph,sk,sk,ph,clean_weight=0,effect_weight=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fp16_prompt_update_is_finite(self):
        from clip.model import CLIP,convert_weights
        from src.model import IndependentVisualPromptLearner
        model=CLIP(32,16,3,64,4,16,128,64,1,1).cuda().eval().requires_grad_(False)
        convert_weights(model)
        photo_prompts=IndependentVisualPromptLearner(3,64,43,3).cuda()
        sketch_prompts=IndependentVisualPromptLearner(3,64,44,3).cuda()
        inputs=torch.randn(4,3,16,16,device='cuda')
        masked_inputs=erase_ink_batch(inputs,torch.tensor([[2,2,9,9]]*4,device='cuda'))
        params=list(photo_prompts.parameters())+list(sketch_prompts.parameters())
        optimizer=torch.optim.SGD(params,lr=.01,momentum=.9)
        with torch.autocast('cuda',dtype=torch.float16):
            ph=model.visual(inputs.half(),*photo_prompts())
            sk=model.visual(inputs.half(),*sketch_prompts())
            masked=model.visual(masked_inputs.half(),*sketch_prompts())
            teachers=[torch.randn(4,48,device='cuda',requires_grad=True) for _ in range(3)]
            loss,_=avcrd_loss(sk,masked,ph,*teachers)
        self.assertEqual(loss.dtype,torch.float32)
        loss.backward()
        for parameter in params:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(),0)
        for target in teachers:self.assertIsNone(target.grad)
        optimizer.step()
        for parameter in params:self.assertTrue(torch.isfinite(parameter).all())

    def test_shuffled_effect_is_a_meaningful_control(self):
        sk=torch.randn(4,16);masked=torch.randn(4,16);ph=torch.randn(5,16)
        normal,_=avcrd_loss(sk,masked,ph,sk,masked,ph)
        shuffled,_=avcrd_loss(sk,masked,ph,sk,masked,ph,shuffle_effect=True)
        self.assertGreater(shuffled.item(),normal.item()+.1)


class CacheTests(unittest.TestCase):
    def test_real_teacher_capture_cache_hit_content_invalidation_and_replay(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        from src.dataset import TrainDataset,TeacherFeatureDataset
        from src.model import DFN5B_MODEL,DFN5B_PRETRAINED,TEACHER_CACHE_FORMAT_VERSION
        from src.counterfactual_cache import prepare_cache,validate_payload
        teacher=torch.nn.Module();teacher.visual=VisionTransformer(56,14,64,1,1,2,output_dim=32).eval().requires_grad_(False)
        controller=TeacherPromptController(teacher.visual,2,1,.02,42).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for modality in ('sketch','photo'):
                for category in ('cat','dog'):
                    (root/modality/category).mkdir(parents=True)
                    for i in range(2):
                        image=Image.new('RGB',(56,56),'white');draw=ImageDraw.Draw(image)
                        draw.rectangle((5+i*4,8,44,47),outline='black',width=3);draw.line((6,10,49,44),fill='black',width=3)
                        image.save(root/modality/category/f'{i}.png')
            args=args_for(root);dataset=TrainDataset(args)
            def encode(paths,mod):
                inputs=torch.stack([TeacherFeatureDataset(paths,56)[i] for i in range(len(paths))])
                with torch.no_grad():return controller(inputs,mod).half()
            meta={'format_version':TEACHER_CACHE_FORMAT_VERSION,'dataset':args.dataset,'max_size':56,
                  'teacher_model':DFN5B_MODEL,'teacher_pretrained':DFN5B_PRETRAINED,'teacher_n_ctx_visual':2,'teacher_prompt_depth':1,
                  'teacher_prompt_std':.02,'teacher_prompt_seed':42}
            torch.save({'metadata':meta,'teacher_prompt_state_dict':controller.state_dict(),
                'teacher_sketch_features':encode(dataset.all_sketches_path,'sketch'),
                'teacher_photo_features':encode(dataset.all_photo_paths,'photo')},args.teacher_cache_path)
            with patch('open_clip.create_model',side_effect=lambda *a,**kw:teacher.to(kw['device'])):
                payload=prepare_cache(args,dataset,root/'report')
            self.assertEqual(payload['masked'].shape,(4,2,32));self.assertIsNotNone(dataset.counterfactual_targets)
            current=dataset[(1,0)]
            self.assertIsInstance(current[-1],dict)
            self.assertEqual(current[-1]['sketch_index'],0)
            with torch.no_grad():
                for selected in (0,1,2,3):
                    view=erase_ink_box(current[1],current[-1]['boxes'][selected])
                    encoded=controller(view[None].to(teacher.visual.conv1.weight.device),'sketch').cpu()
                    torch.testing.assert_close(encoded.half()[0],current[-1]['masked'][selected],atol=.01,rtol=.01)
            state=torch.get_rng_state().clone()
            with patch('open_clip.create_model',side_effect=AssertionError('Cache hit must skip teacher')):
                restored=prepare_cache(args,TrainDataset(args),root/'report_replay')
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            torch.testing.assert_close(payload['masked'],restored['masked'])
            self.assertTrue((root/'report/teacher_probe.json').is_file())
            self.assertTrue((root/'report/teacher_interventions.png').is_file())
            malformed=dict(payload);malformed['boxes']=payload['boxes'].clone();malformed['boxes'][0,0,2]=1000
            with self.assertRaises(ValueError):validate_payload(malformed,payload['metadata'],4,32)
            Image.new('RGB',(56,56),'black').save(dataset.all_sketches_path[0])
            with self.assertRaises(ValueError):prepare_cache(args,dataset,root/'invalid')


class LifecycleTests(unittest.TestCase):
    def test_real_prompt_training_diagnostics_do_not_change_sgd_updates(self):
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader,Dataset
        from clip.model import CLIP
        from src.model import ZS_SBIR,IndependentVisualPromptLearner
        from src.counterfactual_diagnostics import CounterfactualDiagnostics
        from src.losses import loss_fn
        torch.manual_seed(1);torch.set_num_threads(1)
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__();self.clip_model=CLIP(32,16,3,64,4,16,128,64,1,1).eval().requires_grad_(False)
                self.photo_visual_prompt=IndependentVisualPromptLearner(3,64,43,3)
                self.sketch_visual_prompt=IndependentVisualPromptLearner(3,64,44,3);self.classnames=('cat','dog')
            def encode_student_image(self,x,mod):
                learner=getattr(self,mod+'_visual_prompt')
                return F.normalize(self.clip_model.visual(x,*learner()),dim=-1)
            extract_feature=encode_student_image
            def forward(self,batch):
                photo,sketch,tp,ts,_=batch
                return self.encode_student_image(photo,'photo'),self.encode_student_image(sketch,'sketch'),tp,ts,True,None,None,None,None
        module=ZS_SBIR.__new__(ZS_SBIR);pl.LightningModule.__init__(module);module.model=Wrapper()
        module.lambda_av=0;module.lambda_global_feature=0;module.lambda_avcrd=1
        module.args=SimpleNamespace(seed=42,lambda_domain=3.,lambda_modality=0.,kd_temperature=.07,
            lr=.01,momentum=.9,weight_decay=.0005,avcrd_selection='verified',avcrd_objective='field',
            avcrd_clean_weight=1.,avcrd_effect_weight=1.,avcrd_magnitude_weight=.25,
            avcrd_ink_threshold=.08,avcrd_ink_softness=.12,avcrd_diagnostic_batch_size=4,avcrd_diagnostics=True)
        module.val_step_outputs_sk=[];module.val_step_outputs_ph=[];module.best_precision=0.;module.args.dataset='sketchy_2'
        batch=(torch.randn(4,3,16,16),normalized(torch.rand(3,16,16))[None].repeat(4,1,1,1),
               torch.randn(4,48),torch.randn(4,48),torch.tensor([0,0,1,1]),
               {'clean':torch.randn(4,48),'masked':torch.randn(4,4,48),
                'boxes':torch.tensor([[[2,2,7,7],[8,8,13,13],[2,2,7,7],[8,8,13,13]]]*4),'sketch_index':torch.arange(4)})
        class Fixed(Dataset):
            def __len__(self):return 4
            def __getitem__(self,key):
                i=key[1] if isinstance(key,tuple) else key
                return tuple({k:v[i] for k,v in x.items()} if isinstance(x,dict) else x[i] for x in batch)
        class Valid(Dataset):
            def __init__(self,mode):self.mode=mode
            def __len__(self):return 4
            def __getitem__(self,i):return batch[self.mode][i],batch[4][i]
        original=copy.deepcopy(module)
        with patch.object(module,'log'):
            module.lambda_avcrd=0
            torch.testing.assert_close(module.training_step(batch,0),loss_fn(module.args,module(batch[:5]))[0],atol=0,rtol=0)
        states=[]
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False,True):
                current=copy.deepcopy(original)
                callback=CounterfactualDiagnostics()
                trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,num_sanity_val_steps=0,
                    logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
                    default_root_dir=str(Path(tmp)/str(enabled)),callbacks=[callback] if enabled else [])
                trainer.validate(current,[DataLoader(Valid(1),batch_size=4),DataLoader(Valid(0),batch_size=4)],verbose=False)
                trainer.fit(current,DataLoader(Fixed(),batch_size=4),[DataLoader(Valid(1),batch_size=4),DataLoader(Valid(0),batch_size=4)])
                states.append(copy.deepcopy(current.state_dict()))
                if enabled:
                    self.assertTrue((callback.out/'fields_epoch_2.png').is_file())
                    self.assertTrue((callback.out/'gradient_interaction.csv').is_file())
                    self.assertEqual(len(callback.epochs),3)
                    epoch_gradients=[r for r in callback.gradients if r['global_step']>0 and r['group']=='all']
                    self.assertEqual(len(epoch_gradients),2)
                    for record in epoch_gradients:
                        self.assertGreater(record['main_norm'],0)
                        self.assertGreater(record['weighted_cf_norm'],0)
                    self.assertTrue((callback.out/'pair_effects_epoch_2.csv').is_file())
                    checkpoint={};current.on_save_checkpoint(checkpoint)
                    self.assertEqual(checkpoint['experiment_config']['method'],'AVCRD')
            for name in states[0]:torch.testing.assert_close(states[0][name],states[1][name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
