import unittest

import torch
from torch.nn import functional as F

from clip.model import VisionTransformer
from src.stroke_evidence import (
    FinalBlockInputCapture,
    StrokeEvidenceHead,
    centered_field_alignment,
    cls_patch_attention,
    counterfactual_field_alignment,
    erase_by_patch_evidence,
    graph_smooth_evidence,
    hellinger_loss,
    patch_ink_mass,
    projected_patch_features,
)


class StrokeEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_final_block_capture_returns_dense_features_and_attention(self):
        visual = VisionTransformer(
            input_resolution=32,
            patch_size=8,
            width=32,
            layers=2,
            heads=4,
            output_dim=16,
        ).eval()
        images = torch.randn(3, 3, 32, 32)
        with FinalBlockInputCapture(visual) as capture:
            native = visual(images)
        dense = projected_patch_features(visual, capture.residual())
        attention = cls_patch_attention(visual, capture.residual())
        self.assertEqual(tuple(native.shape), (3, 16))
        self.assertEqual(tuple(dense.shape), (3, 16, 16))
        self.assertEqual(tuple(attention.shape), (3, 16))
        self.assertTrue(torch.isfinite(dense).all())
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue(torch.all(attention >= 0))

    def test_zero_initialized_fusion_preserves_native_descriptor_but_trains(self):
        head = StrokeEvidenceHead(
            width=16,
            grid=2,
            bottleneck=8,
            beta=0.2,
            temperature=0.1,
            graph_steps=1,
            graph_mix=0.25,
        )
        native = F.normalize(torch.randn(4, 16), dim=-1)
        dense = torch.randn(4, 4, 16)
        ink = torch.ones(4, 4)
        output = head(native, dense, ink)
        self.assertTrue(torch.allclose(output["descriptor"], native, atol=1e-6))
        target = F.normalize(torch.rand(4, 4), p=1, dim=-1)
        loss = hellinger_loss(output["weights"], target)
        loss = loss + (1 - F.cosine_similarity(output["descriptor"], torch.roll(native, 1, 0))).mean()
        loss.backward()
        self.assertIsNotNone(head.key.weight.grad)
        self.assertGreater(head.key.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(head.fusion.up.weight.grad)
        self.assertGreater(head.fusion.up.weight.grad.abs().sum().item(), 0)

    def test_graph_diffusion_never_leaves_ink_support(self):
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ink = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = graph_smooth_evidence(weights, ink, grid=2, steps=3, mix=0.5)
        self.assertTrue(torch.allclose(result.sum(-1), torch.ones(1)))
        self.assertEqual(result[0, 2].item(), 0.0)
        self.assertEqual(result[0, 3].item(), 0.0)
        self.assertGreater(result[0, 1].item(), 0.0)

    def test_erasure_respects_exact_ink_budget(self):
        from src.stroke_evidence import CLIP_MEAN, CLIP_STD
        rgb = torch.ones(1, 3, 8, 8)
        rgb[:, :, :4, :] = 0.0
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        images = (rgb - mean) / std
        weights = torch.tensor([[0.7, 0.2, 0.08, 0.02]])
        erased, removed = erase_by_patch_evidence(images, weights, fraction=0.25)
        self.assertEqual(tuple(erased.shape), tuple(images.shape))
        self.assertAlmostEqual(removed.item(), 0.25, places=5)
        self.assertTrue(torch.isfinite(erased).all())
        self.assertEqual(tuple(patch_ink_mass(images, 2).shape), (1, 4))

    def test_main_head_matches_original_prompted_visual_path(self):
        from argparse import Namespace
        from clip.model import CLIP
        from src.model import CustomCLIP

        clip_model = CLIP(
            embed_dim=32,
            image_resolution=32,
            vision_layers=2,
            vision_width=64,
            vision_patch_size=8,
            context_length=77,
            vocab_size=50000,
            transformer_width=64,
            transformer_heads=1,
            transformer_layers=1,
        ).eval()
        cfg = Namespace(
            prompt_depth=2,
            n_ctx_visual=3,
            seed=42,
            retrieval_head="main",
            lambda_domain=0.0,
            lambda_modality=0.0,
            kd_temperature=0.07,
            photo_text_kd_temperature=0.15,
            sketch_text_kd_temperature=0.02,
            teacher_cache_path="",
            rebuild_teacher_cache=False,
            teacher_pretrain_epochs=0,
        )
        model = CustomCLIP(cfg, clip_model, ["cat", "dog"], teacher=None).eval()
        images = torch.randn(2, 3, 32, 32)
        prompt, compound = model.get_visual_prompt("sketch")
        expected = F.normalize(
            model.clip_model.visual(images.type(model.dtype), prompt, compound).float(),
            dim=-1,
        )
        actual = model.encode_student_image(images, "sketch")
        self.assertIsNone(model.stroke_evidence_head)
        self.assertTrue(torch.equal(actual, expected))

    def test_rsed_custom_clip_causally_uses_student_evidence_head(self):
        from argparse import Namespace
        from clip.model import CLIP
        from src.model import CustomCLIP

        clip_model = CLIP(
            embed_dim=32,
            image_resolution=32,
            vision_layers=2,
            vision_width=64,
            vision_patch_size=8,
            context_length=77,
            vocab_size=50000,
            transformer_width=64,
            transformer_heads=1,
            transformer_layers=1,
        ).eval()
        cfg = Namespace(
            prompt_depth=2,
            n_ctx_visual=3,
            seed=42,
            retrieval_head="rsed",
            lambda_rsed=1.0,
            rsed_bottleneck=16,
            rsed_beta=0.1,
            rsed_temperature=0.07,
            rsed_graph_steps=1,
            rsed_graph_mix=0.25,
            rsed_ink_threshold=0.08,
            rsed_ink_softness=0.12,
            rsed_target="retrieval",
            lambda_domain=0.0,
            lambda_modality=0.0,
            kd_temperature=0.07,
            photo_text_kd_temperature=0.15,
            sketch_text_kd_temperature=0.02,
            teacher_cache_path="",
            rebuild_teacher_cache=False,
            teacher_pretrain_epochs=0,
        )
        model = CustomCLIP(cfg, clip_model, ["cat", "dog"], teacher=None).eval()
        rgb = torch.ones(2, 3, 32, 32)
        rgb[0, :, 8:24, 14:18] = 0
        rgb[1, :, 14:18, 8:24] = 0
        from src.stroke_evidence import CLIP_MEAN, CLIP_STD
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        images = (rgb - mean) / std
        output = model.encode_student_image_details(images, "sketch")
        self.assertEqual(tuple(output["weights"].shape), (2, 16))
        self.assertTrue(torch.allclose(output["weights"].sum(-1), torch.ones(2)))
        self.assertTrue(torch.allclose(output["descriptor"], output["native"], atol=1e-6))
        self.assertIs(model.encode_student_image_details(images, "photo")["weights"], None)

    def test_relational_fields_are_zero_for_matching_student_teacher(self):
        query = F.normalize(torch.randn(5, 12), dim=-1)
        gallery = F.normalize(torch.randn(5, 12), dim=-1)
        confidence = torch.linspace(0.2, 1.0, 5)
        what, cosine = centered_field_alignment(
            query, gallery, query.detach(), gallery.detach(), confidence
        )
        self.assertLess(what.item(), 1e-6)
        self.assertGreater(cosine.item(), 0.999999)
        masked = F.normalize(query + 0.2 * torch.randn_like(query), dim=-1)
        effect, stats = counterfactual_field_alignment(
            query, masked, gallery,
            query.detach(), masked.detach(), gallery.detach(),
            confidence, magnitude_weight=0.5,
        )
        self.assertLess(effect.item(), 1e-6)
        self.assertGreater(stats["effect_cosine"].item(), 0.999999)
        self.assertAlmostEqual(stats["effect_magnitude_ratio"].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
