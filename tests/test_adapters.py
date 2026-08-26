import unittest

import torch

from clip.model import VisionTransformer
from src.adapters import BottleneckAdapter, ModalityBottleneckAdapters


class BottleneckAdapterTests(unittest.TestCase):
    def test_zero_expansion_starts_as_exact_identity(self):
        adapter = BottleneckAdapter(
            width=8,
            bottleneck=3,
            std=0.02,
            seed=42,
        )
        x = torch.randn(5, 2, 8)
        self.assertTrue(torch.equal(adapter(x), x))
        self.assertGreater(adapter.down_weight.std().item(), 0.0)
        self.assertEqual(adapter.up_weight.count_nonzero().item(), 0)
        self.assertEqual(adapter.up_bias.count_nonzero().item(), 0)

    def test_zero_expansion_receives_gradient_without_changing_backbone(self):
        adapter = BottleneckAdapter(
            width=8,
            bottleneck=3,
            std=0.02,
            seed=42,
        )
        x = torch.randn(5, 2, 8, requires_grad=True)
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.up_weight.grad.norm().item(), 0.0)
        self.assertEqual(adapter.down_weight.grad.count_nonzero().item(), 0)
        self.assertIsNotNone(x.grad)

    def test_photo_and_sketch_adapters_are_independent(self):
        adapters = ModalityBottleneckAdapters(
            width=8,
            bottleneck=3,
            depth=2,
            std=0.02,
            seed=42,
        )
        photo = adapters.for_modality("photo")[0]
        sketch = adapters.for_modality("sketch")[0]
        self.assertIsNot(photo.down_weight, sketch.down_weight)
        self.assertFalse(torch.equal(photo.down_weight, sketch.down_weight))

    def test_student_vit_applies_adapters_after_transformer_blocks(self):
        visual = VisionTransformer(
            input_resolution=4,
            patch_size=2,
            width=8,
            layers=2,
            heads=1,
            output_dim=4,
        ).eval()
        adapters = ModalityBottleneckAdapters(
            width=8,
            bottleneck=3,
            depth=2,
            std=0.02,
            seed=42,
        ).eval()
        images = torch.randn(2, 3, 4, 4)
        baseline = visual(images)
        initial = visual(
            images,
            adapters=adapters.for_modality("photo"),
        )
        self.assertTrue(torch.equal(initial, baseline))
        with torch.no_grad():
            adapters.for_modality("photo")[0].up_weight.fill_(0.1)
        adapted = visual(
            images,
            adapters=adapters.for_modality("photo"),
        )
        self.assertFalse(torch.equal(adapted, baseline))


if __name__ == "__main__":
    unittest.main()
