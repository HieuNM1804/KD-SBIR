import argparse
import copy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

from src.structural_losses import (
    gw_tensor, sinkhorn, solve_fgw, transport_loss, structure, semantic_cost,
    semantic_loss, multi_positive_loss, contraction_loss,
)
from src.structural_runtime import TargetStore
from src.local_features import FinalPatches, area_pool, project_patches
from src.structural_config import add_arguments, validate


def config(**overrides):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    values = vars(parser.parse_args([]))
    values.update(dict(
        seed=42, n_ctx_visual=3, prompt_depth=12, lambda_domain=3.,
        lambda_modality=1., kd_temperature=.07, photo_text_kd_temperature=.15,
        sketch_text_kd_temperature=.02, teacher_pretrain_epochs=0,
        teacher_cache_path="", rebuild_teacher_cache=False, max_size=224,
        workers=0, batch_size=4,
    ))
    values.update(overrides)
    return SimpleNamespace(**values)


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def test_gw_matches_explicit_four_index_cost(self):
        ct = structure(torch.randn(2, 3, 7))
        cs = structure(torch.randn(2, 4, 5)).requires_grad_()
        plan = torch.rand(2, 3, 4)
        plan /= plan.sum((1, 2), keepdim=True)
        explicit = (((ct[:, :, :, None, None] - cs[:, None, None, :, :])**2)
                    * plan[:, :, None, :, None] * plan[:, None, :, None, :]).sum((1,2,3,4))
        actual = (gw_tensor(ct, cs, plan)*plan).sum((1,2))
        torch.testing.assert_close(actual, explicit, atol=2e-6, rtol=1e-5)
        ga = torch.autograd.grad(actual.sum(), cs, retain_graph=True)[0]
        gb = torch.autograd.grad(explicit.sum(), cs)[0]
        torch.testing.assert_close(ga, gb, atol=2e-6, rtol=1e-5)

    def test_semantic_cost_matches_broadcast(self):
        t, s = torch.randn(2,3,11), torch.randn(2,4,11)
        expected = (t[:,:,None]-s[:,None]).square().mean(-1)
        torch.testing.assert_close(semantic_cost(t,s), expected)

    def test_sinkhorn_and_transport_shape(self):
        p, err = sinkhorn(torch.rand(2,7,5), iterations=300)
        torch.testing.assert_close(p.sum(-1), torch.full((2,7),1/7), atol=1e-4, rtol=0)
        torch.testing.assert_close(p.sum(-2), torch.full((2,5),1/5), atol=1e-4, rtol=0)
        self.assertLessEqual(err, 1e-4)
        with self.assertRaises(RuntimeError):
            sinkhorn(torch.rand(1,7,5), epsilon=.001, iterations=1, tolerance=1e-9)

    def test_transport_stop_gradient_and_all_alpha_endpoints(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for alpha in (0., .5, 1.):
                s = torch.randn(2,9,32,device=device,requires_grad=True)
                t = torch.randn(2,16,64,device=device,requires_grad=True)
                hs = torch.randn(2,9,6,device=device,requires_grad=True)
                ht = torch.randn(2,16,6,device=device,requires_grad=True)
                with torch.autocast(device_type=device, dtype=torch.float16 if device=="cuda" else torch.bfloat16):
                    # Runtime explicitly disables autocast around numerical losses.
                    with torch.autocast(device_type=device, enabled=False):
                        loss, logs = transport_loss(structure(t),s,ht,hs,alpha)
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                self.assertIsNone(t.grad)
                self.assertIsNone(ht.grad)
                self.assertTrue(torch.isfinite(loss))
                if alpha > 0:
                    self.assertGreater(s.grad.abs().sum(), 0)
                if alpha < 1:
                    self.assertGreater(hs.grad.abs().sum(), 0)

    def test_plan_objective_not_worse_than_uniform(self):
        ct, cs = structure(torch.randn(2,9,7)), structure(torch.randn(2,4,5))
        d = torch.rand(2,9,4)
        p, _, _ = solve_fgw(ct,cs,d)
        uniform = torch.full_like(p,1/36)
        def objective(v):
            return (.5*(gw_tensor(ct,cs,v)*v).sum((1,2))+
                    .5*(d*v).sum((1,2))+.05*(v*v.clamp_min(1e-30).log()).sum((1,2)))
        self.assertTrue((objective(p) <= objective(uniform)+1e-6).all())

    def test_global_losses_and_contraction(self):
        s = torch.randn(4,8,requires_grad=True)
        p = torch.randn(4,8,requires_grad=True)
        ts, tp = torch.randn(4,13,requires_grad=True), torch.randn(4,13,requires_grad=True)
        sa, ta = torch.randn(6,8), torch.randn(6,13,requires_grad=True)
        labels=torch.tensor([0,0,1,1])
        c, count = contraction_loss(s,p,ts,tp,labels,sa,ta,rho=0.)
        loss=multi_positive_loss(s,p,labels)+semantic_loss(s,ts,sa,ta)+c
        loss.backward()
        self.assertEqual(count,2)
        self.assertIsNone(ts.grad);self.assertIsNone(tp.grad);self.assertIsNone(ta.grad)
        self.assertGreater(s.grad.abs().sum(),0)
        self.assertAlmostEqual(float(multi_positive_loss(s,p,torch.zeros(4))),0)
        c, count=contraction_loss(s,s,ts,tp,labels,sa,ta)
        self.assertEqual(float(c),0)
        self.assertEqual(count,2)
        _, count=contraction_loss(s,p,ts,tp,torch.arange(4),sa,ta)
        self.assertEqual(count,0)

    def test_area_pool_nondivisible_and_hooks_clean_up(self):
        x=torch.randn(2,256,8,requires_grad=True)
        area_pool(x,7).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        ones=torch.ones(1,256,3)
        torch.testing.assert_close(area_pool(ones,7),torch.ones(1,49,3))

    def test_spatial_baseline_and_report(self):
        from src.local_features import spatial_structure_loss
        from src.structural_report import save_transport_report
        patches=torch.randn(2,49,32,requires_grad=True)
        target=structure(area_pool(patches.detach(),4))
        self.assertAlmostEqual(float(spatial_structure_loss(target,patches)),0,places=7)
        loss=spatial_structure_loss(target+0.01,patches)
        loss.backward()
        self.assertTrue(torch.isfinite(patches.grad).all())
        with tempfile.TemporaryDirectory() as folder:
            save_transport_report(folder,"sketch",torch.zeros(3,16,16),target[0],
                                  structure(patches.detach())[0],torch.full((16,49),1/(16*49)),
                                  {"type":"test"})
            text=(Path(folder)/"sketch_transport.html").read_text()
            self.assertIn("data:image/png;base64,",text)
            with self.assertRaises(FileExistsError):
                save_transport_report(folder,"sketch",torch.zeros(3,16,16),target[0],
                                      target[0],torch.full((16,16),1/256),{})

    def test_cli_validation(self):
        parser=argparse.ArgumentParser()
        for change in ({"fgw_alpha":float("nan")}, {"lambda_semantic":-1},
                       {"student_sampler":"class","batch_size":5,"samples_per_class":2},
                       {"lambda_sfgw":1,"n_ctx_visual":0}):
            with self.assertRaises(SystemExit):validate(config(**change),parser)

    def test_store_roundtrip_partial_and_disk_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            meta={"test":1}
            store=TargetStore(folder,meta,3,4,5)
            self.assertFalse(store.open())
            ct=torch.rand(3,4,4); ht=torch.rand(3,4,5)
            store.build(iter([{"structure":ct[:1],"semantic":ht[:1]},
                              {"structure":ct[1:],"semantic":ht[1:]}]))
            result=store.get(torch.tensor([2,0]),"cpu")
            torch.testing.assert_close(result["structure"],ct[[2,0]].half().float())
            torch.testing.assert_close(result["semantic"],ht[[2,0]].half().float())
            # Close mmap handles for Windows temporary cleanup.
            for v in store.arrays.values():v._mmap.close()
            with self.assertRaises(RuntimeError):
                TargetStore(folder,{"test":2},3,4,5).open()
        with tempfile.TemporaryDirectory() as folder:
            store=TargetStore(folder,{},3,4,0)
            with patch("src.structural_runtime.shutil.disk_usage",
                       return_value=SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):store.build(iter([]))
            self.assertFalse((Path(folder)/"manifest.json").exists())

    def test_class_sampler_unique_balanced_indices(self):
        from src.dataset import TrainDataset
        from src.class_sampler import ClassBatchSampler
        with tempfile.TemporaryDirectory() as folder:
            for c in ("aa","bb","cc"):
                for mod in ("sketch","photo"):
                    path=Path(folder)/mod/c;path.mkdir(parents=True)
                    for i in range(4):Image.new("RGB",(8,8)).save(path/f"{i}.png")
            ds=TrainDataset(config(root=folder,dataset="sketchy_2",lambda_sfgw=1))
            sampler=ClassBatchSampler(ds,4,2,42)
            first=list(sampler)
            self.assertEqual(first,list(ClassBatchSampler(ds,4,2,42)))
            for batch in first:
                self.assertEqual(len({k[1] for k in batch}),4)
                self.assertEqual(len({k[2] for k in batch}),4)
                loaded=[ds[k] for k in batch]
                self.assertEqual(sorted([v[4] for v in loaded]).count(loaded[0][4]),2)
                for key,value in zip(batch,loaded):
                    self.assertEqual(value[5].tolist(),[key[1],key[2]])
            self.assertNotEqual(first,list(sampler))


class TrainingTests(unittest.TestCase):
    def test_real_teacher_pretraining_with_indexed_dataset(self):
        import open_clip
        from clip.model import CLIP
        from src.model import CustomCLIP
        from src.dataset import TrainDataset
        with tempfile.TemporaryDirectory() as folder:
            for c in ("aa","bb"):
                for mod in ("sketch","photo"):
                    path=Path(folder)/mod/c;path.mkdir(parents=True)
                    for i in range(2):
                        Image.new("RGB",(16,16),(30+i*100,60 if c=="aa" else 200,100)).save(path/f"{i}.png")
            args=config(root=folder,dataset="sketchy_2",max_size=16,lambda_sfgw=1,
                teacher_pretrain_epochs=1,teacher_n_ctx_visual=2,teacher_prompt_depth=2,
                teacher_prompt_std=.02,teacher_prompt_seed=42,teacher_prompt_lr=.03,
                teacher_momentum=.9,teacher_weight_decay=.001,teacher_pretrain_batch_size=2,
                teacher_prompt_gradient_checkpointing=False,lambda_teacher_retrieval=1.5,
                teacher_triplet_margin=.2,teacher_scheduler_step_size=5,teacher_scheduler_gamma=.1,
                teacher_cache_path=str(Path(folder)/"teacher.pt"))
            ds=TrainDataset(args)
            teacher=open_clip.CLIP(embed_dim=1024,
                vision_cfg=open_clip.CLIPVisionCfg(image_size=16,patch_size=2,width=64,layers=2),
                text_cfg=open_clip.CLIPTextCfg(width=64,heads=1,layers=1)).eval().requires_grad_(False)
            teacher.text_tokenizer=open_clip.get_tokenizer("ViT-H-14-quickgelu")
            student=CustomCLIP(args,CLIP(32,16,3,64,4,77,49408,64,1,1),ds.all_categories,teacher)
            frozen=copy.deepcopy(teacher.state_dict())
            with patch.object(student,"_validate_teacher_unseen",return_value=.5) as validation:
                student.cache_teacher_features(ds,None,None,batch_size=2,workers=0,show_progress=False)
            self.assertEqual(validation.call_count,1)
            payload=torch.load(args.teacher_cache_path,weights_only=True)
            self.assertEqual(payload["metadata"]["pretrain_epochs"],1)
            self.assertEqual(tuple(payload["teacher_sketch_features"].shape),(4,1024))
            self.assertIsNone(student._teacher)
            self.assertIsNone(student.teacher_prompts)
            self.assertIsNotNone(ds.teacher_sketch_features)
            for key,value in teacher.state_dict().items():
                torch.testing.assert_close(value,frozen[key],rtol=0,atol=0)

    def test_preparation_restores_prompts_and_cache_matches_online(self):
        import open_clip
        from clip.model import CLIP
        from src.model import CustomCLIP
        from src.dataset import TrainDataset
        from src.structural_runtime import StructuralRuntime
        from src.teacher_prompts import build_teacher_prompt_controller
        with tempfile.TemporaryDirectory() as folder:
            for mod in ("sketch", "photo"):
                path=Path(folder)/mod/"cat";path.mkdir(parents=True)
                Image.new("RGB", (16,16), (80,120,60)).save(path/"0.png")
            args=config(root=folder,dataset="sketchy_2", max_size=16,
                        lambda_semantic=1.,lambda_sfgw=1.,teacher_patch_grid=3,
                        teacher_pretrain_epochs=1,teacher_n_ctx_visual=2,
                        teacher_prompt_depth=2,teacher_prompt_std=.02,teacher_prompt_seed=42,
                        teacher_prompt_gradient_checkpointing=True,teacher_prompt_lr=.03,
                        teacher_momentum=.9,teacher_weight_decay=.001,
                        teacher_pretrain_batch_size=2,lambda_teacher_retrieval=1.5,
                        teacher_triplet_margin=.2,teacher_scheduler_step_size=5,
                        teacher_scheduler_gamma=.1,teacher_cache_path=str(Path(folder)/"teacher.pt"),
                        local_cache_dir=str(Path(folder)/"local"))
            ds=TrainDataset(args)
            backbone=CLIP(32,16,3,64,4,77,49408,64,1,1)
            student=CustomCLIP(args,backbone,["cat"])
            def make_teacher(_args):
                teacher=open_clip.CLIP(embed_dim=48,
                    vision_cfg=open_clip.CLIPVisionCfg(image_size=16,patch_size=2,width=64,layers=2),
                    text_cfg=open_clip.CLIPTextCfg(width=64,heads=1,layers=1))
                teacher.load_state_dict(state)
                return teacher.eval().requires_grad_(False)
            teacher=open_clip.CLIP(embed_dim=48,
                vision_cfg=open_clip.CLIPVisionCfg(image_size=16,patch_size=2,width=64,layers=2),
                text_cfg=open_clip.CLIPTextCfg(width=64,heads=1,layers=1))
            state=copy.deepcopy(teacher.state_dict())
            prompts=build_teacher_prompt_controller(teacher,2,2)
            with torch.no_grad():
                for p in prompts.parameters():p.add_(.1)
            torch.save(dict(metadata=student._teacher_cache_metadata(ds),
                            teacher_prompt_state_dict=prompts.state_dict()),args.teacher_cache_path)
            with patch("src.model._load_teacher", side_effect=make_teacher):
                runtime=StructuralRuntime(student,ds,args)
                for a,b in zip(runtime.prompts.parameters(),prompts.parameters()):
                    torch.testing.assert_close(a,b)
                img=ds[0][1].unsqueeze(0)
                expected=runtime.targets(img,"sketch")
                args.local_target_mode="cache"
                cached=StructuralRuntime(student,ds,args)
                self.assertIsNone(cached.teacher)
                actual=cached.store.get(torch.tensor([0]),"cpu")
                for k in expected:
                    torch.testing.assert_close(actual[k],expected[k].half().float())
                for v in cached.store.arrays.values():v._mmap.close()
                with patch.object(StructuralRuntime,"targets",side_effect=AssertionError("Must reuse")):
                    again=StructuralRuntime(student,ds,args)
                for v in again.store.arrays.values():v._mmap.close()

    def test_baseline_and_all_new_losses_real_student(self):
        from clip.model import CLIP, convert_weights
        from src.model import CustomCLIP, ZS_SBIR
        from src.losses import loss_fn
        from src.structural_runtime import StructuralRuntime
        import open_clip
        device="cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        args=config(lambda_domain=0.,lambda_modality=0.,lambda_retrieval=1.,
                    lambda_semantic=1.,lambda_sfgw=1.,lambda_contract=1.,
                    teacher_patch_grid=3, local_teacher_batch_size=1)
        backbone=CLIP(32,16,12,64,4,77,49408,64,1,1).to(device)
        if device=="cuda":convert_weights(backbone)
        student=CustomCLIP(args,backbone,["cat","dog"]).to(device)
        student.teacher_active=True
        teacher=open_clip.CLIP(
            embed_dim=48,vision_cfg=open_clip.CLIPVisionCfg(
                image_size=16,patch_size=2,width=96,head_width=32,layers=2),
            text_cfg=open_clip.CLIPTextCfg(width=64,heads=1,layers=1),
        ).to(device).eval().requires_grad_(False)
        runtime=StructuralRuntime.__new__(StructuralRuntime)
        runtime.report_written=set()
        runtime.args=args;runtime.teacher=teacher;runtime.prompts=None
        runtime.store=None;runtime.grid=3;runtime.local_semantic=True
        runtime.student_anchors=torch.randn(6,32,device=device)
        runtime.teacher_anchors=torch.randn(6,48,device=device)
        wrapper=ZS_SBIR.__new__(ZS_SBIR);torch.nn.Module.__init__(wrapper)
        wrapper.args=args;wrapper.model=student
        object.__setattr__(wrapper,"_structural_runtime",runtime)
        wrapper.log=lambda *a,**kw:None
        batch=(torch.randn(4,3,16,16,device=device),torch.randn(4,3,16,16,device=device),
               torch.randn(4,48,device=device),torch.randn(4,48,device=device),
               torch.tensor([0,0,1,1],device=device))
        keys=tuple(wrapper.state_dict())
        wrapper.train()
        with torch.autocast(device_type=device,dtype=torch.float16 if device=="cuda" else torch.bfloat16):
            loss=wrapper.training_step(batch,0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name,param in student.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
                self.assertGreater(param.grad.abs().sum(),0,name)
            else:self.assertIsNone(param.grad,name)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(tuple(wrapper.state_dict()),keys)
        self.assertEqual(len(backbone.visual.transformer.resblocks[-1]._forward_hooks),0)
        before=student.photo_visual_prompt.ctx.detach().clone()
        torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=.01).step()
        self.assertFalse(torch.equal(before,student.photo_visual_prompt.ctx))
        wrapper.eval()
        with torch.inference_mode():
            enabled=student.extract_feature(batch[0],"photo")
            args.lambda_sfgw=0
            disabled=student.extract_feature(batch[0],"photo")
        torch.testing.assert_close(enabled,disabled,rtol=0,atol=0)

        # With all new weights zero, training_step equals the original loss path.
        args.lambda_retrieval=args.lambda_semantic=args.lambda_contract=0.
        args.lambda_domain=3.
        args.lambda_modality=1.
        student.image_text_kd_active=True
        student.photo_text_active=student.sketch_text_active=True
        student._teacher_photo_text=torch.randn(2,48,device=device)
        student._teacher_sketch_text=torch.randn(2,48,device=device)
        wrapper.train()
        reference,_=loss_fn(args,wrapper(batch))
        expected=torch.autograd.grad(reference,student.photo_visual_prompt.ctx)[0]
        actual=wrapper.training_step(batch,0)
        gradient=torch.autograd.grad(actual,student.photo_visual_prompt.ctx)[0]
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        torch.testing.assert_close(gradient,expected,rtol=0,atol=0)

        # All six objectives can coexist in the real training step.
        args.lambda_retrieval=args.lambda_semantic=args.lambda_contract=args.lambda_sfgw=1.
        with torch.autocast(device_type=device,dtype=torch.float16 if device=="cuda" else torch.bfloat16):
            together=wrapper.training_step(batch,1)
        together.backward()
        self.assertTrue(torch.isfinite(together))


if __name__=="__main__":
    unittest.main()
