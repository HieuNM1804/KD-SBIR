"""Regression checks for direct prompt supervision and main parity."""

import copy
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from clip.model import CLIP, convert_weights
from src.losses import loss_fn
from src.model import ZS_SBIR, CustomCLIP
from src.stroke_graph import FinalBlockInputCapture, cls_patch_attention, hellinger_loss
from src.stroke_prompt_probe import localization_probe


def config(**overrides):
    values = {
        "prompt_depth": 3,
        "n_ctx_visual": 3,
        "seed": 42,
        "backbone": "ViT-B/32",
        "retrieval_head": "sgcd",
        "sgcd_student_mode": "native_prompt",
        "lambda_domain": 3.0,
        "lambda_modality": 0.0,
        "kd_temperature": 0.07,
        "photo_text_kd_temperature": 0.15,
        "sketch_text_kd_temperature": 0.02,
        "teacher_cache_path": "",
        "rebuild_teacher_cache": False,
        "teacher_pretrain_epochs": 0,
        "lambda_sgcd": 1.0,
        "lambda_sgcd_where": 1.0,
        "lambda_sgcd_what": 0.25,
        "lambda_sgcd_effect": 0.25,
        "lambda_sgcd_anchor": 0.0,
        "lambda_sgcd_rank": 0.5,
        "sgcd_rank_margin": 0.2,
        "sgcd_effect_magnitude_weight": 0.25,
        "sgcd_target": "verified",
        "sgcd_mask_fraction": 0.1,
        "sgcd_beta": 0.0,
        "sgcd_ink_threshold": 0.08,
        "sgcd_ink_softness": 0.12,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    }
    values.update(overrides)
    return Namespace(**values)


def tiny_clip():
    return CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=3,
        vision_width=64,
        vision_patch_size=8,
        context_length=77,
        vocab_size=50000,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    ).eval()


class NativePromptTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_descriptor_and_prompt_initialization_match_main_in_both_dtypes(self):
        for half in (False, True):
            backbone = tiny_clip()
            if half:
                convert_weights(backbone)
            model = CustomCLIP(config(), backbone, ["cat", "dog"]).eval()
            main = CustomCLIP(
                config(retrieval_head="main"), copy.deepcopy(backbone), ["cat", "dog"]
            ).eval()
            images = torch.randn(2, 3, 32, 32)
            self.assertIsNone(model.stroke_graph_head)
            self.assertTrue(
                all(
                    "visual_prompt" in name
                    for name, p in model.named_parameters()
                    if p.requires_grad
                )
            )
            for modality in ("photo", "sketch"):
                expected = main.encode_student_image(images, modality)
                actual = model.encode_student_image_details(images, modality)
                prompt, compound = main.get_visual_prompt(modality)
                raw = main.clip_model.visual(images.to(main.dtype), prompt, compound)
                reference = raw / raw.norm(dim=-1, keepdim=True)
                self.assertEqual(actual["descriptor"].dtype, raw.dtype)
                self.assertTrue(torch.equal(expected, reference))
                self.assertTrue(torch.equal(actual["descriptor"], reference))
                self.assertTrue(torch.equal(actual["native"], reference))
                self.assertEqual(actual["correction"].count_nonzero(), 0)

    def test_attention_readout_matches_frozen_multihead_attention(self):
        model = CustomCLIP(config(), tiny_clip(), ["cat", "dog"]).eval()
        images = torch.randn(2, 3, 32, 32)
        prompt, compound = model.get_visual_prompt("sketch")
        with FinalBlockInputCapture(model.clip_model.visual) as capture:
            model.clip_model.visual(images, prompt, compound)
        block = model.clip_model.visual.transformer.resblocks[-1]
        tokens = block.ln_1(capture.residual())
        _, expected = block.attn(tokens, tokens, tokens, need_weights=True)
        actual = cls_patch_attention(model.clip_model.visual, capture.residual())
        self.assertTrue(torch.allclose(actual, expected[:, 0, 1:17], atol=1e-7))

    def test_each_auxiliary_component_reaches_sketch_prompts_without_head(self):
        with (
            patch("src.model._load_clip_model", return_value=tiny_clip()),
            patch("src.model._load_teacher", return_value=None),
        ):
            module = ZS_SBIR(config(), ["cat", "dog"]).eval()
        images = torch.randn(4, 3, 32, 32)
        target_map = F.softmax(torch.randn(4, 16), dim=-1)
        target = {
            "maps": target_map[:, None].expand(-1, 3, -1),
            "mask_priorities": target_map[:, None].expand(-1, 3, -1),
            "teacher_evidence": torch.randn(4, 3, 48),
            "teacher_masked": torch.randn(4, 3, 48),
            "confidence": torch.ones(4, 3),
            "selected_effect": torch.ones(4, 3),
            "clean_margin": torch.ones(4),
            "masked_margin": torch.zeros(4, 3),
        }
        batch = (
            torch.randn_like(images),
            images,
            torch.randn(4, 48),
            torch.randn(4, 48),
            torch.tensor([0, 0, 1, 1]),
            target,
        )
        features, output = module.model.forward_with_stroke_graph(batch[:5])
        # Use teacher-space targets with a different width from the student.
        features = list(features)
        features[2:4] = batch[2:4]
        _, _, _, components = module.stroke_graph_loss(
            batch, features, output, return_components=True
        )
        sketch = list(module.model.sketch_visual_prompt.parameters())
        photo = list(module.model.photo_visual_prompt.parameters())
        for name in ("where", "what", "effect", "rank"):
            gradients = torch.autograd.grad(
                components[name], sketch + photo, retain_graph=True, allow_unused=True
            )
            current = [g for g in gradients[: len(sketch)] if g is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in current), name)
            self.assertGreater(sum(g.abs().sum().item() for g in current), 1e-8, name)
            self.assertTrue(all(g is None for g in gradients[len(sketch) :]), name)
        self.assertTrue(
            all(p.grad is None for p in module.model.clip_model.parameters())
        )

    def test_prompt_only_optimization_learns_nonuniform_locations(self):
        model = CustomCLIP(config(), tiny_clip(), ["cat", "dog"]).eval()
        images = torch.randn(2, 3, 32, 32)
        target = F.one_hot(torch.tensor([2, 11]), 16).float()
        frozen_before = {k: v.clone() for k, v in model.clip_model.state_dict().items()}
        optimizer = torch.optim.Adam(model.sketch_visual_prompt.parameters(), lr=0.03)
        initial = model.encode_student_image_details(images, "sketch")
        before = hellinger_loss(initial["weights"], target).item()
        for _ in range(60):
            output = model.encode_student_image_details(images, "sketch")
            loss = hellinger_loss(output["weights"], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = model.encode_student_image_details(images, "sketch")
        after = hellinger_loss(final["weights"], target).item()
        self.assertLess(after, before * 0.95)
        self.assertFalse(torch.equal(initial["native"], final["native"]))
        self.assertTrue(
            all(
                torch.equal(v, frozen_before[k])
                for k, v in model.clip_model.state_dict().items()
            )
        )

    def test_preflight_restores_parameters_gradients_modes_and_rng(self):
        model = CustomCLIP(config(), tiny_clip(), ["cat", "dog"]).train()
        images = torch.randn(2, 3, 32, 32)
        target = F.one_hot(torch.tensor([2, 11]), 16).float()
        for p in model.sketch_visual_prompt.parameters():
            p.grad = torch.randn_like(p)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        gradients = [p.grad.clone() for p in model.sketch_visual_prompt.parameters()]
        rng = torch.get_rng_state().clone()
        modes = [child.training for child in model.modules()]
        summary, *_ = localization_probe(model, images, target, torch.ones(2), steps=3)
        self.assertEqual(summary["trainable_head_parameters"], 0)
        self.assertTrue(summary["sketch_prompts_restored"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual([child.training for child in model.modules()], modes)
        self.assertTrue(
            all(torch.equal(v, saved[k]) for k, v in model.state_dict().items())
        )
        self.assertTrue(
            all(
                torch.equal(p.grad, grad)
                for p, grad in zip(model.sketch_visual_prompt.parameters(), gradients)
            )
        )

    def test_native_mode_rejects_disconnected_or_head_only_settings(self):
        import argparse

        from src.stroke_graph_cache import add_arguments, validate_arguments

        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "--retrieval_head",
                "sgcd",
                "--sgcd_student_mode",
                "native_prompt",
                "--lambda_sgcd",
                "1",
                "--sgcd_beta",
                "0",
                "--lambda_sgcd_anchor",
                "0",
            ]
        )
        args.n_ctx_visual, args.prompt_depth, args.teacher_pretrain_epochs = 3, 12, 1
        validate_arguments(parser, args)
        args.prompt_depth = 1
        with self.assertRaises(SystemExit):
            validate_arguments(parser, args)
        args.prompt_depth, args.sgcd_beta = 12, 0.1
        with self.assertRaises(SystemExit):
            validate_arguments(parser, args)

    def test_main_loss_prompt_gradients_are_identical_before_auxiliary_updates(self):
        backbone = tiny_clip()
        convert_weights(backbone)
        direct = CustomCLIP(config(), backbone, ["cat", "dog"]).eval()
        main = CustomCLIP(
            config(retrieval_head="main"), copy.deepcopy(backbone), ["cat", "dog"]
        ).eval()
        direct.teacher_active = main.teacher_active = True
        images = torch.randn(4, 3, 32, 32)
        batch = (
            torch.randn_like(images),
            images,
            torch.randn(4, 48),
            torch.randn(4, 48),
            torch.tensor([0, 0, 1, 1]),
        )
        loss_direct, _ = loss_fn(direct.cfg, direct(batch))
        loss_main, _ = loss_fn(main.cfg, main(batch))
        self.assertTrue(torch.equal(loss_direct, loss_main))
        loss_direct.backward()
        loss_main.backward()
        grads = {name: p.grad for name, p in main.named_parameters() if p.requires_grad}
        for name, parameter in direct.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.equal(parameter.grad, grads[name]), name)

    def test_component_diagnostics_work_under_inference_mode_without_updating_state(
        self,
    ):
        from pathlib import Path

        from src.stroke_graph_diagnostics import StrokeGraphDiagnostics

        with (
            patch("src.model._load_clip_model", return_value=tiny_clip()),
            patch("src.model._load_teacher", return_value=None),
        ):
            module = ZS_SBIR(config(), ["cat", "dog"]).eval()
        module.model.teacher_active = True
        images = torch.randn(4, 3, 32, 32)
        maps = F.softmax(torch.randn(4, 3, 16), dim=-1)
        target = {
            "maps": maps,
            "mask_priorities": maps,
            "teacher_evidence": torch.randn(4, 3, 48),
            "teacher_masked": torch.randn(4, 3, 48),
            "confidence": torch.ones(4, 3),
            "selected_effect": torch.ones(4, 3),
            "clean_margin": torch.ones(4),
            "masked_margin": torch.zeros(4, 3),
        }
        callback = StrokeGraphDiagnostics()
        callback.batch = (
            images,
            images,
            torch.randn(4, 48),
            torch.randn(4, 48),
            torch.tensor([0, 0, 1, 1]),
            target,
        )
        saved = {name: value.clone() for name, value in module.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            trainer = Namespace(
                log_dir=directory, default_root_dir=directory, global_step=0
            )
            with torch.inference_mode():
                callback.measure(trainer, module, "initial")
            out = Path(directory) / "sgcd_diagnostics"
            self.assertTrue((out / "component_gradients.csv").is_file())
            self.assertTrue((out / "prompt_learning.png").is_file())
            for name in ("where", "what", "effect", "rank"):
                row = next(
                    r
                    for r in callback.component_gradients
                    if r["component"] == name and r["group"] == "sketch_prompts"
                )
                self.assertGreater(row["weighted_sgcd_norm"], 0)
            self.assertFalse(
                any(r["group"] == "evidence_head" for r in callback.gradients)
            )
        self.assertTrue(
            all(
                torch.equal(value, saved[name])
                for name, value in module.state_dict().items()
            )
        )
        self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
