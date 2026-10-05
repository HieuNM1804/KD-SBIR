"""CPU checks of four-teacher mixing, cached alignment and actual prompt training."""

import ast
import copy
from pathlib import Path
import random
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.teacher_ensemble import TEACHERS, DEFAULT_TEACHERS, normalized_weights, cache_ensemble
from src.losses import loss_fn, ensemble_teacher_log_probs


BASE = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"


def config(**overrides):
    values = dict(
        root="", dataset="sketchy_2", max_size=32, seed=42, backbone="ViT-B/32",
        teacher_models=list(DEFAULT_TEACHERS), teacher_ensemble_weights=[.1, .2, .3, .4],
        n_ctx_visual=3, prompt_depth=2, lambda_domain=3., lambda_modality=1.,
        kd_temperature=.07, photo_text_kd_temperature=.15, sketch_text_kd_temperature=.02,
        lr=.01, momentum=.9, weight_decay=1e-5, teacher_cache_path="", teacher_cache_dir="",
        rebuild_teacher_cache=False, teacher_pretrain_epochs=0, teacher_pretrain_batch_size=2,
        teacher_n_ctx_visual=3, teacher_prompt_depth=2, teacher_prompt_std=.02,
        teacher_prompt_seed=42, teacher_prompt_gradient_checkpointing=False,
        teacher_prompt_lr=.01, teacher_momentum=.9, teacher_weight_decay=.001,
        lambda_teacher_retrieval=1.5, teacher_triplet_margin=2.,
        teacher_scheduler_step_size=5, teacher_scheduler_gamma=.1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def features(args, n=3, classes=3):
    dims = [TEACHERS[key]["output_dim"] for key in args.teacher_models]
    g = torch.Generator().manual_seed(23)
    def rand(*shape, grad=False):
        return torch.randn(*shape, generator=g, requires_grad=grad)
    return (rand(n, 16, grad=True), rand(n, 16, grad=True),
            rand(n, sum(dims), grad=True), rand(n, sum(dims), grad=True), True,
            rand(classes, 16), rand(classes, 16),
            rand(classes, sum(dims)), rand(classes, sum(dims)))


class TeacherEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weights_validation(self):
        self.assertEqual(normalized_weights(None, 4), (.25,) * 4)
        self.assertEqual(normalized_weights([1, 2, 3, 4], 4), (.1, .2, .3, .4))
        for values in ([0] * 4, [1, -1, 1, 1], [float("nan")] * 4,
                       [float("inf")] * 4, [1, 2], [1e308] * 4):
            with self.assertRaises(ValueError):
                normalized_weights(values, 4)

    def test_probability_mixture_all_three_losses_and_gradients(self):
        args = config()
        f = features(args)
        dims = [TEACHERS[key]["output_dim"] for key in args.teacher_models]
        weights = args.teacher_ensemble_weights
        def mix(images, candidates, tau):
            result = 0.
            for weight, image, candidate in zip(weights, images.detach().split(dims, -1), candidates.detach().split(dims, -1)):
                logits = F.normalize(image.float(), dim=-1) @ F.normalize(candidate.float(), dim=-1).t()
                result = result + weight * F.softmax(logits / tau, dim=-1)
            return result
        target_domain = mix(f[3], f[2], args.kd_temperature)
        actual_log = ensemble_teacher_log_probs(f[3], f[2], args, args.kd_temperature)
        torch.testing.assert_close(actual_log.exp(), target_domain)
        torch.testing.assert_close(actual_log.exp().sum(-1), torch.ones(3))
        self.assertFalse(actual_log.requires_grad)
        student_domain = F.log_softmax(F.normalize(f[1], dim=-1) @ F.normalize(f[0], dim=-1).t() / args.kd_temperature, -1)
        domain = F.kl_div(student_domain, target_domain, reduction="batchmean")
        def symmetric(image, text, target):
            log = F.log_softmax(F.normalize(image, dim=-1) @ F.normalize(text, dim=-1).t() / tau, -1)
            return .5 * (F.kl_div(log, target, reduction="batchmean")
                         + (log.exp() * (log - target.log())).sum(-1).mean())
        tau = args.photo_text_kd_temperature
        photo = symmetric(f[0], f[6], mix(f[2], f[8], tau))
        tau = args.sketch_text_kd_temperature
        sketch = symmetric(f[1], f[5], mix(f[3], f[7], tau))
        expected = 3 * domain + photo + sketch
        actual, _ = loss_fn(args, f)
        torch.testing.assert_close(actual, expected)
        for left, right in zip(torch.autograd.grad(actual, f[:2], retain_graph=True),
                               torch.autograd.grad(expected, f[:2], retain_graph=True)):
            torch.testing.assert_close(left, right)
        actual.backward()
        self.assertIsNone(f[2].grad)
        self.assertIsNone(f[3].grad)
        with self.assertRaises(ValueError):
            ensemble_teacher_log_probs(f[2][:, :-1], f[8], args, tau)

    def test_single_dfn5b_exact_main_loss_and_gradients(self):
        old = subprocess.check_output(["git", "show", f"{BASE}:src/losses.py"], text=True,
                                      cwd=Path(__file__).resolve().parents[1])
        baseline = ModuleType("baseline")
        exec(compile(old, "main/losses.py", "exec"), baseline.__dict__)
        for dom, mod in ((3., 1.), (3., 0.), (0., 1.)):
            args = config(teacher_models=["dfn5b"], teacher_ensemble_weights=[1.],
                          lambda_domain=dom, lambda_modality=mod)
            f = features(args)
            expected = baseline.loss_fn(args, f)[0]
            actual = loss_fn(args, f)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for left, right in zip(torch.autograd.grad(actual, f[:2], retain_graph=True),
                                   torch.autograd.grad(expected, f[:2])):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_each_loader_matches_existing_branch_specs_and_cache_config(self):
        from src.model import _load_teacher, _teacher_training_config, default_teacher_cache_path
        # Reference loader values from the existing teacher branches. The Kaggle
        # bundle is a single-branch clone and cannot rely on their Git refs.
        references = {
            "dfn5b": ("ViT-H-14-quickgelu", "dfn5b", 1024, 6),
            "openai_l14": ("ViT-L-14", "openai", 768, 7),
            "dfn2b_l14": ("ViT-L-14", "dfn2b_s39b", 768, 7),
            "laion_h14": ("ViT-H-14", "laion2b_s32b_b79k", 1024, 7),
        }
        source = subprocess.check_output(["git", "show", f"{BASE}:src/model.py"], text=True)
        tree = ast.parse(source)
        config_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_teacher_training_config")
        paths = []
        for key, reference in references.items():
            spec = TEACHERS[key]
            self.assertEqual(tuple(spec[field] for field in ("model", "pretrained", "output_dim", "cache_version")), reference)
            args = config(teacher_models=[key], teacher_ensemble_weights=[1.])
            fake = torch.nn.Linear(2, 2)
            with patch("src.model.open_clip.create_model", return_value=fake) as loader, \
                 patch("src.model.open_clip.get_tokenizer", return_value=object()):
                teacher = _load_teacher(args)
                self.assertEqual(loader.call_args.args[0], spec["model"])
                self.assertEqual(loader.call_args.kwargs["pretrained"], spec["pretrained"])
                self.assertEqual(teacher.output_dim, spec["output_dim"])
                self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            dataset = SimpleNamespace(max_size=32, all_categories=["cat"],
                                      all_sketches_path=[], all_photo_paths=[])
            paths.append(default_teacher_cache_path(args, dataset))
            namespace = dict(DFN5B_MODEL=reference[0], DFN5B_PRETRAINED=reference[1], DFN5B_OUTPUT_DIM=reference[2])
            exec(compile(ast.Module(body=[config_fn], type_ignores=[]), "old_config", "exec"), namespace)
            self.assertEqual(_teacher_training_config(args), namespace["_teacher_training_config"](args))
        self.assertEqual(len(set(paths)), 4)

    def test_sequential_real_cache_roundtrip_and_dataset_alignment(self):
        from clip.model import CLIP
        from src.dataset import TrainDataset, TeacherFeatureDataset
        from src.model import CustomCLIP
        from open_clip.transformer import VisionTransformer
        with tempfile.TemporaryDirectory() as root:
            for modality in ("photo", "sketch"):
                for category in ("first_seen_class", "second_seen_class"):
                    folder = Path(root) / modality / category
                    folder.mkdir(parents=True)
                    rng = np.random.default_rng(len(str(folder)))
                    for i in range(2):
                        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f"{i}.png")
            args = config(root=root, teacher_cache_dir=str(Path(root) / "cache"),
                          teacher_pretrain_epochs=1, teacher_pretrain_batch_size=4)
            dataset = TrainDataset(args)
            student = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                           vision_width=64, vision_patch_size=16, context_length=77,
                           vocab_size=49408, transformer_width=64, transformer_heads=1,
                           transformer_layers=1)
            owner = CustomCLIP(args, student, dataset.all_categories)
            loaded = []
            def load(single):
                from src.model import _persistent_teacher_cache_available
                if _persistent_teacher_cache_available(single):
                    return None
                key = single.teacher_models[0]
                loaded.append(key)
                teacher = torch.nn.Module()
                teacher.visual = VisionTransformer(image_size=32, patch_size=16, width=64,
                                                    layers=2, heads=2, mlp_ratio=4,
                                                    output_dim=TEACHERS[key]["output_dim"])
                teacher.text_embedding = torch.nn.Parameter(torch.randn(4, TEACHERS[key]["output_dim"]), requires_grad=False)
                teacher.text_tokenizer = lambda texts: torch.arange(len(texts))
                teacher.encode_text = lambda tokens: teacher.text_embedding[tokens]
                teacher.encode_image = lambda images: teacher.visual(images)
                return teacher.eval().requires_grad_(False)
            rng_before = torch.get_rng_state().clone()
            with patch("src.model._load_teacher", side_effect=load), \
                 patch.object(CustomCLIP, "_validate_teacher_unseen", return_value=.5):
                cache_ensemble(owner, dataset, None, None, 2, 0, False)
            torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
            self.assertEqual(loaded, list(DEFAULT_TEACHERS))
            self.assertIsNone(owner._teacher)
            self.assertIsNone(owner.teacher_prompts)
            self.assertEqual(dataset.teacher_photo_features.shape, (4, 3584))
            packed = dataset.teacher_photo_features.clone()
            packed_text = owner._teacher_photo_text.clone()
            files = list((Path(root) / "cache").glob("*.pt"))
            self.assertEqual(len(files), 4)
            expected_parts = []
            for key in DEFAULT_TEACHERS:
                spec = TEACHERS[key]
                payload = next(torch.load(p, weights_only=True) for p in files
                               if torch.load(p, weights_only=True)["metadata"]["teacher_pretrained"] == spec["pretrained"])
                expected_parts.append(payload["teacher_photo_features"])
            torch.testing.assert_close(packed, torch.cat(expected_parts, -1), rtol=0, atol=0)
            for epoch in (0, 3):
                sample = dataset[(epoch, 0)]
                possible = [dataset.photo_path_to_index[p] for p in dataset.all_photos_path["first_seen_class"]]
                self.assertTrue(any(torch.equal(sample[2], packed[i]) for i in possible))
            # Reuse all four caches with no pretrained backbone load/pretraining.
            with patch("src.model.open_clip.create_model", side_effect=AssertionError("Unexpected teacher download")), \
                 patch.object(CustomCLIP, "_pretrain_teacher_prompts", side_effect=AssertionError("Unexpected retraining")):
                cache_ensemble(owner, dataset, None, None, 2, 0, False)
            torch.testing.assert_close(dataset.teacher_photo_features, packed, rtol=0, atol=0)
            torch.testing.assert_close(owner._teacher_photo_text, packed_text, rtol=0, atol=0)
            # Explicit cache belonging to another model cannot be used.
            bad = copy.copy(args)
            bad.teacher_models = ["dfn5b"]
            bad.teacher_cache_path = str(next(p for p in files
                if torch.load(p, weights_only=True)["metadata"]["teacher_pretrained"] == "openai"))
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                CustomCLIP(bad, student, dataset.all_categories)._load_persistent_teacher_cache(dataset)
            # Original pretrained teachers (zero pretraining epochs) also have
            # independent reusable caches and do not create prompt controllers.
            args.teacher_pretrain_epochs = 0
            with patch("src.model._load_teacher", side_effect=load), \
                 patch.object(CustomCLIP, "_pretrain_teacher_prompts", side_effect=AssertionError("Unexpected pretraining")):
                cache_ensemble(owner, dataset, None, None, 2, 0, False)
            self.assertEqual(len(list((Path(root) / "cache").glob("*.pt"))), 8)
            self.assertTrue(torch.isfinite(dataset.teacher_sketch_features).all())

    def test_real_student_lightning_uses_ensemble_only_prompts_update(self):
        from clip.model import CLIP
        from src.model import ZS_SBIR
        from pytorch_lightning import Trainer
        from torch.utils.data import DataLoader
        args = config()
        tiny = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                    vision_width=64, vision_patch_size=16, context_length=77,
                    vocab_size=49408, transformer_width=64, transformer_heads=1,
                    transformer_layers=1)
        with patch("src.model._load_clip_model", return_value=tiny):
            model = ZS_SBIR(args, ["cat", "dog", "car"])
        f = features(args)
        model.model.teacher_active = True
        model.model._teacher_sketch_text = f[7].detach()
        model.model._teacher_photo_text = f[8].detach()
        batch = (torch.randn(3, 3, 32, 32), torch.randn(3, 3, 32, 32),
                 f[2].detach(), f[3].detach(), torch.arange(3))
        before = {n: p.clone() for n, p in model.named_parameters()}
        samples = [tuple(v[i] for v in batch) for i in range(3)]
        trainer = Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, DataLoader(samples, batch_size=3))
        self.assertEqual(trainer.global_step, 2)
        self.assertTrue(all(p.grad is None for p in tiny.parameters()))
        changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
        self.assertTrue(changed)
        self.assertTrue(all("visual_prompt" in n for n in changed))
        self.assertEqual(model.model.extract_feature(batch[0], "photo").shape, (3, 16))
        self.assertIsNone(model.model._teacher)


if __name__ == "__main__":
    unittest.main()
