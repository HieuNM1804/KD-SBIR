import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from clip.model import CLIP, convert_weights
from src.model import CustomCLIP, ZS_SBIR, _build_teacher_prompts
from src.losses import loss_fn
from src.dataset import TeacherFeatureDataset
from src.evidence_config import add_arguments, validate
from src.evidence_references import balanced_indices, prototypes, scores
from src.evidence_views import intervene, regions
from src.evidence_targets import prepare_targets, select_region, validate_payload
from src.evidence_runtime import EvidenceRuntime
from src.evidence_losses import evidence_loss
from src.evidence_report import write_report


def config(**overrides):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    values = vars(parser.parse_args([]))
    values.update(seed=42, max_size=16, prompt_depth=12, n_ctx_visual=3,
                  lambda_modality=1., lambda_domain=3., kd_temperature=.07,
                  photo_text_kd_temperature=.15, sketch_text_kd_temperature=.02,
                  teacher_pretrain_epochs=1, teacher_cache_path="", rebuild_teacher_cache=False,
                  teacher_n_ctx_visual=3, teacher_prompt_depth=2,
                  teacher_prompt_std=.02, teacher_prompt_seed=42,
                  teacher_prompt_gradient_checkpointing=False, progress=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_views_and_reference_balance(self):
        image = torch.arange(3 * 16 * 16).float().reshape(3, 16, 16)
        boxes = regions(16, .25)
        self.assertTrue(all((b[2]-b[0])*(b[3]-b[1]) == 64 for b in boxes))
        masked, crop = intervene(image, boxes[0])
        self.assertTrue(torch.equal(masked[:, 8:], image[:, 8:]))
        self.assertEqual(crop.shape, image.shape)
        self.assertTrue(torch.equal(intervene(image, boxes[0])[0], masked))
        donor = torch.ones_like(image)
        self.assertTrue(torch.equal(intervene(image, boxes[0], "donor", donor)[0][:, :8, :8], donor[:, :8, :8]))
        labels = torch.tensor([0,0,0,1,1,1])
        ids = balanced_indices(labels, 2, 42)
        self.assertEqual(labels[ids].bincount().tolist(), [2,2])
        self.assertTrue(torch.equal(ids, balanced_indices(labels, 2, 42)))
        with self.assertRaises(ValueError):
            balanced_indices(labels, 4, 42)

    def test_selection_and_matched_random_coverage(self):
        clean = torch.tensor([.8,.1])
        masked = torch.tensor([[.3,.2],[.8,.1],[.7,.15],[.75,.1],[.8,.11]])
        cropped = torch.tensor([[.7,.2]] * 5)
        args = config()
        self.assertEqual(select_region(clean, masked, cropped, 0, args, torch.Generator(), "important"), 0)
        self.assertEqual(select_region(clean, masked, cropped, 0, args, torch.Generator(), "stable"), 1)
        args.evidence_selection = "random"
        self.assertIsNotNone(select_region(clean, masked, cropped, 0, args, torch.Generator(), "important"))
        self.assertIsNone(select_region(clean.flip(0), masked, cropped, 0, args, torch.Generator(), "important"))

    def test_loss_exact_and_stop_gradient(self):
        clean = torch.tensor([[.8,.1]], requires_grad=True)
        changed = torch.tensor([[.4,.2]], requires_grad=True)
        teacher = torch.tensor([[.7,.2]], requires_grad=True)
        teacher_changed = torch.tensor([[.6,.1]], requires_grad=True)
        loss = evidence_loss(clean, changed, teacher, teacher_changed)
        expected = .5 * torch.tensor([.3**2, .2**2]).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_changed.grad)
        self.assertTrue(torch.equal(clean.grad, -changed.grad))
        self.assertEqual(evidence_loss(clean, changed, clean, changed).item(), 0)

    def test_cli_validation(self):
        validate(config(max_size=224))
        for kwargs in ({"lambda_evidence":float("nan")}, {"evidence_batch_fraction":0},
                       {"lambda_evidence":1, "teacher_pretrain_epochs":0}, {"evidence_area":1}):
            with self.assertRaises(ValueError):
                validate(config(**kwargs))

    def test_dataset_indices_leave_sampling_unchanged(self):
        from src.dataset import TrainDataset, WorkerInvariantSampler
        from torch.utils.data import DataLoader
        with TemporaryDirectory() as folder:
            for modality in ("photo", "sketch"):
                for name in ("cat", "dog"):
                    directory = Path(folder) / modality / name
                    directory.mkdir(parents=True)
                    for i in range(2):
                        Image.new("RGB", (16, 16), (i*50, 10, 20)).save(directory / f"{i}.png")
            args = config(root=folder, dataset="sketchy_2")
            with patch("src.dataset.UNSEEN_CLASSES", {"sketchy_2": []}):
                dataset = TrainDataset(args)
            dataset.set_teacher_features(torch.randn(4,48), torch.arange(4).float()[:,None].repeat(1,48))
            for key in ((0,0),(2,1)):
                original = dataset[key]
                dataset.return_evidence_index = True
                extended = dataset[key]
                self.assertEqual(len(extended),6)
                for left,right in zip(original,extended[:5]):
                    self.assertTrue(torch.equal(left,right) if isinstance(left,torch.Tensor) else left==right)
                self.assertEqual(extended[2][0],extended[5])
                dataset.return_evidence_index = False
            dataset.return_evidence_index = True
            loader = DataLoader(dataset,batch_size=2,sampler=WorkerInvariantSampler(dataset,42),num_workers=0)
            self.assertEqual(len(next(iter(loader))),6)

    def test_tuned_teacher_cache_replay_and_content_identity(self):
        from open_clip.transformer import VisionTransformer
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {}
            for modality in ("photo", "sketch"):
                paths[modality] = []
                for c, name in enumerate(("cat", "dog")):
                    directory = root / modality / name
                    directory.mkdir(parents=True)
                    for i in range(2):
                        path = directory / f"{i}.png"
                        pixels = np.random.default_rng(c*10+i).integers(0,256,(16,16,3),dtype=np.uint8)
                        Image.fromarray(pixels).save(path)
                        paths[modality].append(str(path))
            dataset = SimpleNamespace(max_size=16, all_categories=["cat","dog"],
                                      all_photo_paths=paths["photo"], all_sketches_path=paths["sketch"])
            args = config(root=str(root), teacher_cache_path=str(root / "teacher.pt"),
                          evidence_cache_path=str(root / "evidence.pt"), evidence_refs_per_class=2)
            teacher = torch.nn.Module()
            teacher.visual = VisionTransformer(16, 2, 96, 2, 3, 2, output_dim=48)
            teacher.eval().requires_grad_(False)
            controller = _build_teacher_prompts(args, teacher)
            for parameter in controller.parameters():
                parameter.data.add_(.1)
            torch.save({"metadata":{"fixture":1}, "teacher_prompt_state_dict":controller.state_dict()}, args.teacher_cache_path)
            stub = SimpleNamespace(_teacher_cache_metadata=lambda d: {"fixture":1})
            def factory(*a, **kw):
                result = copy.deepcopy(teacher).to(kw["device"])
                if kw["precision"] == "fp16":
                    result.half()
                return result
            with patch("open_clip.create_model", side_effect=factory):
                first = prepare_targets(args, stub, dataset)
            with patch("open_clip.create_model", side_effect=AssertionError("must reuse")):
                second = prepare_targets(args, stub, dataset)
            self.assertTrue(torch.equal(first[0]["masked"], second[0]["masked"]))
            self.assertEqual(first[0]["masked"].shape, (4,5,2))
            # Independently replay one recorded teacher view and exact tuned prompts.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            restored = factory(device=device, precision="fp16" if device.type == "cuda" else "fp32")
            prompts = _build_teacher_prompts(args, restored)
            prompts.load_state_dict(controller.state_dict())
            dtype = restored.visual.conv1.weight.dtype
            refs = TeacherFeatureDataset(first[1], 16)
            with torch.no_grad():
                encoded = prompts(torch.stack([refs[i] for i in range(4)]).to(device,dtype), "sketch")
                proto = prototypes(encoded.cpu(), first[2], 2)
                row = 0
                index = first[0]["metadata"]["photo_indices"][row]
                image = TeacherFeatureDataset(paths["photo"],16)[index]
                altered,_ = intervene(image, first[0]["metadata"]["boxes"][0])
                q = scores(prompts(altered[None].to(device,dtype),"photo").cpu(), proto)
            self.assertTrue(torch.allclose(q[0], first[0]["masked"][row,0], atol=2e-3))
            runtime = EvidenceRuntime(args, dataset, *first)
            write_report(runtime, root / "report")
            self.assertTrue((root / "report/report.html").is_file())
            with self.assertRaises(FileExistsError):
                write_report(runtime, root / "report")
            Image.new("RGB",(16,16),"white").save(paths["photo"][0])
            with self.assertRaisesRegex(ValueError,"metadata mismatch"):
                prepare_targets(args, stub, dataset)

    def test_real_training_amp_all_prompts_baseline_and_inference(self):
        for device in (["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device), TemporaryDirectory() as folder:
                args = config(lambda_evidence=1., evidence_batch_fraction=1., evidence_region_kind="stable",
                              evidence_stable_rms=2., evidence_refs_per_class=2)
                backbone = CLIP(32,16,12,64,4,77,49408,64,1,1).to(device)
                if device == "cuda":
                    convert_weights(backbone)
                student = CustomCLIP(args, backbone, ["cat","dog"], teacher=None).to(device)
                student.teacher_active = True
                student._teacher_sketch_text = F.normalize(torch.randn(2,1024,device=device),dim=-1)
                student._teacher_photo_text = F.normalize(torch.randn(2,1024,device=device),dim=-1)
                dataset = SimpleNamespace(max_size=16, all_categories=["cat","dog"], all_photo_paths=[])
                refpaths = []
                for i in range(4):
                    path = Path(folder) / f"{i}.png"
                    Image.fromarray(np.random.default_rng(i).integers(0,256,(16,16,3),dtype=np.uint8)).save(path)
                    refpaths.append(str(path))
                payload = {"metadata":{"photo_indices":[0,1,2],"boxes":regions(16,.25)},
                           "clean":torch.tensor([[.8,.1],[.1,.8],[.8,.1]]),
                           "masked":torch.randn(3,5,2)*.1,"cropped":torch.randn(3,5,2)}
                runtime = EvidenceRuntime(args,dataset,payload,refpaths,torch.tensor([0,0,1,1]),torch.tensor([0,1,0]))
                runtime.refresh(student,0)
                self.assertFalse(runtime.student_prototypes.requires_grad)
                wrapper = ZS_SBIR.__new__(ZS_SBIR)
                torch.nn.Module.__init__(wrapper)
                wrapper.args, wrapper.model = args,student
                wrapper.log = lambda *a,**kw:None
                object.__setattr__(wrapper,"_evidence_runtime",runtime)
                wrapper.train()
                batch = (torch.randn(3,3,16,16,device=device),torch.randn(3,3,16,16,device=device),
                         torch.randn(3,1024,device=device),torch.randn(3,1024,device=device),
                         torch.tensor([0,1,0],device=device),torch.tensor([0,1,2],device=device))
                keys = tuple(student.state_dict())
                with torch.autocast(device_type=device, dtype=torch.float16 if device=="cuda" else torch.bfloat16):
                    loss = wrapper.training_step(batch,0)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(loss.dtype,torch.float32)
                loss.backward()
                for name,p in student.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                        self.assertGreater(p.grad.abs().sum().item(),0,name)
                    else:
                        self.assertIsNone(p.grad,name)
                before = student.photo_visual_prompt.ctx.detach().clone()
                torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=.01).step()
                self.assertFalse(torch.equal(before,student.photo_visual_prompt.ctx))
                self.assertEqual(tuple(student.state_dict()),keys)
                student.eval()
                with torch.inference_mode():
                    enabled = student.extract_feature(batch[0],"photo")
                    args.lambda_evidence=0
                    disabled = student.extract_feature(batch[0],"photo")
                self.assertTrue(torch.equal(enabled,disabled))
                wrapper.train()
                student.zero_grad(set_to_none=True)
                baseline = loss_fn(args,student(batch[:5]))[0]
                baseline.backward()
                grads={n:p.grad.clone() for n,p in student.named_parameters() if p.requires_grad}
                student.zero_grad(set_to_none=True)
                actual = wrapper.training_step(batch,0)
                actual.backward()
                self.assertTrue(torch.equal(baseline,actual))
                for n,p in student.named_parameters():
                    if p.requires_grad:
                        self.assertTrue(torch.equal(p.grad,grads[n]),n)
                self.assertTrue(torch.equal(student(batch[:5])[0],student(batch)[0]))
                # Auxiliary alone reaches every photo prompt; the detached
                # sketch prototypes deliberately do not train sketch prompts.
                student.zero_grad(set_to_none=True)
                args.lambda_evidence = 1.
                for objective in ("response", "masked"):
                    args.evidence_objective = objective
                    auxiliary, logs = runtime.loss(student,batch,student(batch),0)
                    self.assertEqual(logs["evidence_samples"],3)
                    auxiliary.backward()
                    for name,p in student.photo_visual_prompt.named_parameters():
                        self.assertIsNotNone(p.grad,name)
                        self.assertGreater(p.grad.abs().sum().item(),0,name)
                    student.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
