import unittest

import torch

from clip.model import VisionTransformer
import open_clip
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg
from open_clip.transformer import VisionTransformer as OpenClipVisionTransformer
from src.patch_prompts import (
    SharedImageConditionedPrompt,
    SharedPatchToPromptProjector,
    shared_prompt_infonce_loss,
)
from src.teacher_prompts import TeacherPromptController


class SharedPatchPromptTests(unittest.TestCase):
    def _projector(self):
        return SharedPatchToPromptProjector(
            visual_width=32,
            text_width=24,
            latent_width=16,
            context_tokens=8,
            heads=4,
            dropout=0.0,
            gate_init=0.1,
            seed=42,
            initial_context=torch.zeros(8, 24),
        )

    def test_one_projector_accepts_photo_and_sketch_patches(self):
        projector = self._projector()
        photo_context, photo_attention = projector(torch.randn(5, 49, 32))
        sketch_context, sketch_attention = projector(torch.randn(3, 49, 32))
        self.assertEqual(photo_context.shape, (5, 8, 24))
        self.assertEqual(sketch_context.shape, (3, 8, 24))
        self.assertEqual(photo_attention.shape, (5, 8, 49))
        self.assertEqual(sketch_attention.shape, (3, 8, 49))

    def test_prompt_loss_does_not_backpropagate_into_patch_features(self):
        projector = self._projector()
        patches = torch.randn(2, 49, 32, requires_grad=True)
        contexts, _ = projector(patches)
        contexts.square().mean().backward()
        self.assertIsNone(patches.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in projector.parameters()
            )
        )

    def test_exact_instance_prompt_infonce_uses_full_gallery(self):
        gallery = torch.eye(100)
        queries = gallery[[2, 41, 87]]
        loss = shared_prompt_infonce_loss(
            queries,
            gallery,
            torch.tensor([2, 41, 87]),
            temperature=0.07,
        )
        self.assertLess(loss.item(), 1e-3)

    def test_visual_transformer_can_return_only_real_patch_tokens(self):
        model = VisionTransformer(
            input_resolution=32,
            patch_size=16,
            width=64,
            layers=1,
            heads=1,
            output_dim=32,
        )
        images = torch.randn(2, 3, 32, 32)
        visual_prompt = torch.randn(3, 64)
        pooled, patches = model(
            images,
            prompt=visual_prompt,
            return_patch_tokens=True,
        )
        self.assertEqual(pooled.shape, (2, 32))
        self.assertEqual(patches.shape, (2, 4, 64))

    def test_teacher_visual_controller_returns_only_real_patch_tokens(self):
        visual = OpenClipVisionTransformer(
            image_size=32,
            patch_size=16,
            width=64,
            layers=1,
            heads=1,
            mlp_ratio=4.0,
            output_dim=32,
        )
        controller = TeacherPromptController(
            visual=visual,
            n_ctx=3,
            depth=1,
            std=0.02,
            seed=42,
        )
        pooled, patches = controller(
            torch.randn(2, 3, 32, 32),
            "photo",
            return_patch_tokens=True,
        )
        self.assertEqual(pooled.shape, (2, 32))
        self.assertEqual(patches.shape, (2, 4, 64))

    def test_openclip_teacher_text_tower_accepts_generated_soft_prompts(self):
        teacher = CLIP(
            embed_dim=32,
            vision_cfg=CLIPVisionCfg(
                layers=1,
                width=64,
                head_width=64,
                patch_size=16,
                image_size=32,
            ),
            text_cfg=CLIPTextCfg(width=64, heads=1, layers=1),
        ).requires_grad_(False)
        prompt = SharedImageConditionedPrompt(
            text_model=teacher,
            tokenizer=open_clip.tokenize,
            seen_classnames=["airplane"],
            unseen_classnames=["cat"],
            visual_width=64,
            latent_width=16,
            context_tokens=4,
            heads=4,
            dropout=0.0,
            seed=42,
            text_backend="open_clip",
        )
        patches = torch.randn(2, 4, 64, requires_grad=True)
        features, attention = prompt(
            teacher,
            patches,
            torch.zeros(2, dtype=torch.long),
            split="seen",
        )
        self.assertEqual(features.shape, (2, 32))
        self.assertEqual(attention.shape, (2, 4, 4))
        features.sum().backward()
        self.assertIsNone(patches.grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in prompt.parameters())
        )


if __name__ == "__main__":
    unittest.main()
