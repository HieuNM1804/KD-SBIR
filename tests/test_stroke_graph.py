import unittest
from argparse import Namespace

import torch
from torch.nn import functional as F

from src.stroke_graph import (
    CLIP_MEAN,
    CLIP_STD,
    StrokeGraphEvidenceHead,
    erase_by_patch_evidence,
    local_photo_correspondence,
    patch_ink_mass,
    path_priority_maps,
    stroke_path_maps,
    zhang_suen_thinning,
)


class StrokeGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def normalized_sketches():
        rgb = torch.ones(2, 3, 56, 56)
        # A T-shaped line graph with a junction and three branches.
        rgb[0, :, 8:48, 26:30] = 0
        rgb[0, :, 8:12, 10:46] = 0
        # A loop plus a connected tail; path splitter must produce candidates.
        rgb[1, :, 10:14, 10:46] = 0
        rgb[1, :, 42:46, 10:46] = 0
        rgb[1, :, 10:46, 10:14] = 0
        rgb[1, :, 10:46, 42:46] = 0
        rgb[1, :, 42:54, 26:30] = 0
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        return (rgb - mean) / std

    def test_zhang_suen_thins_a_thick_line(self):
        mask = torch.zeros(1, 15, 15, dtype=torch.bool)
        mask[:, 2:13, 6:9] = True
        result = zhang_suen_thinning(mask)
        self.assertGreater(result.sum().item(), 5)
        self.assertLess(result.sum().item(), mask.sum().item())
        self.assertTrue(result[:, 3:12].any())

    def test_stroke_paths_are_structural_and_distinct(self):
        images = self.normalized_sketches()
        maps, valid, skeleton = stroke_path_maps(
            images, output_grid=7, skeleton_grid=28, max_paths=6
        )
        self.assertEqual(tuple(maps.shape), (2, 6, 49))
        self.assertEqual(tuple(skeleton.shape), (2, 28, 28))
        self.assertGreaterEqual(valid[0].sum().item(), 3)
        self.assertGreaterEqual(valid[1].sum().item(), 2)
        for row in range(2):
            current = maps[row, valid[row]]
            self.assertTrue(torch.allclose(current.sum(-1), torch.ones(len(current))))
            if len(current) > 1:
                cosine = F.cosine_similarity(current[0:1], current[1:], dim=-1)
                self.assertTrue((cosine < 0.999).all())

    def test_path_priority_preserves_identity_and_exact_budget(self):
        images = self.normalized_sketches()[:1]
        maps, valid, _ = stroke_path_maps(images, 7, 28, 6)
        ink = patch_ink_mass(images, 7)
        priorities = path_priority_maps(maps, ink, 7, valid)
        first_two = priorities[:, :2]
        self.assertFalse(torch.allclose(first_two[:, 0], first_two[:, 1]))
        expanded = images[:, None].expand(-1, 2, -1, -1, -1).reshape(2, 3, 56, 56)
        erased, removed = erase_by_patch_evidence(
            expanded, first_two.reshape(2, 49), fraction=0.10
        )
        self.assertEqual(tuple(erased.shape), (2, 3, 56, 56))
        self.assertTrue(torch.allclose(removed, torch.full_like(removed, 0.10), atol=1e-5))

    def test_local_photo_correspondence_selects_matching_path(self):
        paths = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        photos = torch.tensor([[[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
                                  [0.0, 0.0, 1.0]]]])
        scores = local_photo_correspondence(paths, photos, top_k=2)
        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())

    def test_head_starts_as_main_descriptor_and_receives_gradients(self):
        head = StrokeGraphEvidenceHead(16, 2, bottleneck=8, beta=0.1,
                                       temperature=0.07, graph_steps=0, graph_mix=0)
        native = F.normalize(torch.randn(4, 16), dim=-1)
        output = head(native, torch.randn(4, 4, 16), torch.ones(4, 4))
        self.assertTrue(torch.allclose(output['descriptor'], native, atol=1e-6))
        target = F.normalize(torch.rand(4, 4), p=1, dim=-1)
        loss = (output['weights'].sqrt() - target.sqrt()).square().sum()
        loss += (1 - F.cosine_similarity(output['descriptor'], torch.roll(native, 1, 0))).mean()
        loss.backward()
        self.assertGreater(head.key.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.fusion.up.weight.grad.abs().sum().item(), 0)

    def test_main_path_is_unchanged(self):
        from clip.model import CLIP
        from src.model import CustomCLIP
        clip_model = CLIP(
            embed_dim=32, image_resolution=32, vision_layers=2, vision_width=64,
            vision_patch_size=8, context_length=77, vocab_size=50000,
            transformer_width=64, transformer_heads=1, transformer_layers=1,
        ).eval()
        cfg = Namespace(
            prompt_depth=2, n_ctx_visual=3, seed=42, retrieval_head='main',
            lambda_domain=0.0, lambda_modality=0.0, kd_temperature=0.07,
            photo_text_kd_temperature=0.15, sketch_text_kd_temperature=0.02,
            teacher_cache_path='', rebuild_teacher_cache=False,
            teacher_pretrain_epochs=0,
        )
        model = CustomCLIP(cfg, clip_model, ['cat', 'dog'], teacher=None).eval()
        images = torch.randn(2, 3, 32, 32)
        prompt, compound = model.get_visual_prompt('sketch')
        expected = F.normalize(model.clip_model.visual(images, prompt, compound).float(), dim=-1)
        self.assertTrue(torch.equal(model.encode_student_image(images, 'sketch'), expected))
        self.assertIsNone(model.stroke_graph_head)

    def test_teacher_audit_gate_detects_verified_advantage(self):
        from src.stroke_graph_cache import _audit_summary
        parts = [{
            'selected_effect': torch.tensor([[0.03, 0.02, 0.01],
                                             [0.02, 0.015, 0.005]]),
            'selected_path_index': torch.tensor([[1, 0, 2], [2, 0, 1]]),
            'path_count': torch.tensor([4, 5]),
            'maps': torch.tensor([
                [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
                [[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]],
            ]),
        }]
        args = Namespace(sgcd_min_effect_ratio=1.5, sgcd_min_win_rate=0.5,
                         sgcd_max_random_map_cosine=0.75,
                         sgcd_force_prepare=False)
        summary = _audit_summary(parts, args)
        self.assertTrue(summary['passed'])
        self.assertGreater(summary['verified_random_effect_ratio'], 2)
        self.assertEqual(summary['verified_beats_random_rate'], 1.0)

    def test_teacher_target_batch_has_matched_structural_controls(self):
        from clip.model import VisionTransformer
        from src.stroke_graph_cache import _target_batch
        from src.teacher_prompts import TeacherPromptController

        visual = VisionTransformer(
            input_resolution=56, patch_size=14, width=64,
            layers=2, heads=1, output_dim=16,
        ).eval()
        # The vendored CLIP transformer is sequence-first; OpenCLIP exposes this flag.
        visual.transformer.batch_first = False
        visual.patch_dropout = torch.nn.Identity()
        visual._pool = lambda sequence: (sequence[:, 0], None)
        controller = TeacherPromptController(
            visual, n_ctx=1, depth=2, std=0.02, seed=42
        ).eval().requires_grad_(False)
        args = Namespace(
            sgcd_student_grid=7, sgcd_skeleton_grid=28, sgcd_max_paths=4,
            sgcd_ink_threshold=0.08, sgcd_ink_softness=0.12,
            sgcd_local_topk_patches=2, sgcd_mask_fraction=0.10,
            sgcd_teacher_batch_size=2, sgcd_proposal_topk=2,
            sgcd_max_random_map_cosine=0.75,
            max_size=56, seed=42,
        )
        images = self.normalized_sketches()
        photo_dense = F.normalize(torch.randn(2, 2, 16, 16), dim=-1)
        photo_global = F.normalize(torch.randn(2, 2, 16), dim=-1)
        result = _target_batch(
            images, torch.tensor([4, 9]), torch.tensor([0, 1]),
            controller, photo_dense, photo_global, args,
        )
        self.assertEqual(tuple(result['maps'].shape), (2, 3, 49))
        self.assertEqual(tuple(result['mask_priorities'].shape), (2, 3, 49))
        self.assertEqual(tuple(result['teacher_evidence'].shape), (2, 3, 16))
        self.assertEqual(tuple(result['teacher_masked'].shape), (2, 3, 16))
        self.assertTrue(torch.allclose(result['maps'].sum(-1), torch.ones(2, 3)))
        self.assertTrue(torch.allclose(
            result['removed_ink_fraction'], torch.full((2, 3), 0.10), atol=1e-5
        ))
        self.assertTrue(((result['confidence'] >= 0) & (result['confidence'] <= 1)).all())
        for row in range(2):
            if result['path_count'][row] > 1:
                self.assertNotEqual(
                    result['selected_path_index'][row, 0].item(),
                    result['selected_path_index'][row, 2].item(),
                )

    def test_cache_payload_contract(self):
        from src.stroke_graph_cache import validate_payload
        n, variants, grid, width, paths = 2, 3, 49, 16, 4
        metadata = {
            'sketch_count': n, 'student_grid': 7, 'max_paths': paths,
            'mask_fraction': 0.10,
            'teacher_metadata': {'teacher_output_dim': width},
        }
        maps = torch.full((n, variants, grid), 1 / grid, dtype=torch.float16)
        payload = {
            'metadata': metadata,
            'maps': maps.clone(), 'mask_priorities': maps.clone(),
            'teacher_evidence': torch.zeros(n, variants, width, dtype=torch.float16),
            'teacher_masked': torch.zeros(n, variants, width, dtype=torch.float16),
            'confidence': torch.ones(n, variants),
            'selected_effect': torch.zeros(n, variants),
            'removed_ink_fraction': torch.full((n, variants), 0.10),
            'candidate_effects': torch.zeros(n, paths, dtype=torch.float16),
            'candidate_local_scores': torch.zeros(n, paths, dtype=torch.float16),
            'path_valid': torch.ones(n, paths, dtype=torch.bool),
            'selected_path_index': torch.zeros(n, variants, dtype=torch.int16),
            'path_count': torch.full((n,), paths, dtype=torch.int16),
            'clean_cache_cosine': torch.ones(n),
        }
        validate_payload(payload, metadata)


if __name__ == '__main__':
    unittest.main()
