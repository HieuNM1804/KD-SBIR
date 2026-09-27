import unittest

import torch
from torch import nn

from src.tinyclip_student import PromptedTinyCLIPVision


class RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states):
        self.seen = hidden_states.detach().clone()
        return hidden_states


class FakeTransformer(nn.Module):
    def __init__(self, layer_count):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [RecordingLayer() for _ in range(layer_count)]
        )


class FakeVisual(nn.Module):
    def __init__(self, width=8, layer_count=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(5, width))
        self.ln_pre = nn.Identity()
        self.transformer = FakeTransformer(layer_count)
        self.ln_post = nn.Identity()
        self.proj = nn.Parameter(torch.eye(width))


class PromptedTinyCLIPVisionTest(unittest.TestCase):
    def test_deep_prompts_replace_instead_of_accumulate(self):
        visual = FakeVisual()
        prompted = PromptedTinyCLIPVision(visual)
        shallow = torch.ones(2, 8)
        deep_one = torch.full((2, 8), 2.0)
        deep_two = torch.full((2, 8), 3.0)

        output = prompted(
            torch.randn(4, 3, 8, 8),
            shallow,
            [deep_one, deep_two],
        )

        layers = visual.transformer.resblocks
        self.assertEqual(output.shape, (4, 8))
        self.assertEqual(layers[0].seen.shape, (7, 4, 8))
        self.assertTrue(
            torch.equal(layers[0].seen[-2:], shallow[:, None].expand(-1, 4, -1))
        )
        self.assertTrue(
            torch.equal(layers[1].seen[-2:], deep_one[:, None].expand(-1, 4, -1))
        )
        self.assertTrue(
            torch.equal(layers[2].seen[-2:], deep_two[:, None].expand(-1, 4, -1))
        )

    def test_no_prompt_matches_native_visual_path(self):
        visual = FakeVisual()
        prompted = PromptedTinyCLIPVision(visual)
        images = torch.randn(4, 3, 8, 8)

        output = prompted(images)

        hidden = visual.conv1(images).flatten(2).permute(0, 2, 1)
        class_token = visual.class_embedding.expand(images.shape[0], 1, -1)
        hidden = torch.cat((class_token, hidden), dim=1)
        hidden = hidden + visual.positional_embedding
        hidden = hidden.permute(1, 0, 2)
        for layer in visual.transformer.resblocks:
            hidden = layer(hidden)
        expected = visual.ln_post(hidden.permute(1, 0, 2)[:, 0]) @ visual.proj
        self.assertTrue(torch.equal(output, expected))

    def test_rejects_wrong_prompt_width(self):
        prompted = PromptedTinyCLIPVision(FakeVisual())
        with self.assertRaisesRegex(ValueError, "Visual prompt"):
            prompted(torch.randn(1, 3, 8, 8), torch.randn(2, 9))


if __name__ == "__main__":
    unittest.main()
