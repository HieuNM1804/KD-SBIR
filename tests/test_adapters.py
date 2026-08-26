import unittest

import torch

from src.adapters import BottleneckAdapter, ModalityOutputAdapters


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
        adapters = ModalityOutputAdapters(
            width=8,
            bottleneck=3,
            std=0.02,
            seed=42,
        )
        photo = adapters.for_modality("photo")
        sketch = adapters.for_modality("sketch")
        self.assertIsNot(photo.down_weight, sketch.down_weight)
        self.assertFalse(torch.equal(photo.down_weight, sketch.down_weight))

    def test_modality_adapter_changes_only_the_final_embedding(self):
        adapters = ModalityOutputAdapters(
            width=4,
            bottleneck=3,
            std=0.02,
            seed=42,
        ).eval()
        baseline = torch.randn(2, 4)
        initial = adapters(baseline, "photo")
        self.assertTrue(torch.equal(initial, baseline))
        with torch.no_grad():
            adapters.for_modality("photo").up_weight.fill_(0.1)
        adapted = adapters(baseline, "photo")
        self.assertFalse(torch.equal(adapted, baseline))


if __name__ == "__main__":
    unittest.main()
