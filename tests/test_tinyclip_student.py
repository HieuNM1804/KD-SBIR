import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.tinyclip_student import PromptedTinyCLIPVision


class RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        self.seen = hidden_states.detach().clone()
        return (hidden_states,)


class FakeEmbeddings(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, images):
        return torch.zeros(images.shape[0], 5, self.width)


class PromptedTinyCLIPVisionTest(unittest.TestCase):
    def test_deep_prompts_replace_instead_of_accumulate(self):
        width = 8
        layers = nn.ModuleList([RecordingLayer() for _ in range(3)])
        vision = nn.Module()
        vision.config = SimpleNamespace(hidden_size=width)
        vision.embeddings = FakeEmbeddings(width)
        vision.pre_layrnorm = nn.Identity()
        vision.post_layernorm = nn.Identity()
        vision.encoder = nn.Module()
        vision.encoder.layers = layers
        prompted = PromptedTinyCLIPVision(vision, nn.Identity())

        shallow = torch.ones(2, width)
        deep_one = torch.full((2, width), 2.0)
        deep_two = torch.full((2, width), 3.0)
        output = prompted(
            torch.randn(4, 3, 16, 16),
            shallow,
            [deep_one, deep_two],
        )

        self.assertEqual(output.shape, (4, width))
        self.assertEqual(layers[0].seen.shape, (4, 7, width))
        self.assertTrue(torch.equal(layers[0].seen[:, -2:], shallow.expand(4, -1, -1)))
        self.assertTrue(torch.equal(layers[1].seen[:, -2:], deep_one.expand(4, -1, -1)))
        self.assertTrue(torch.equal(layers[2].seen[:, -2:], deep_two.expand(4, -1, -1)))

    def test_rejects_wrong_prompt_width(self):
        width = 8
        vision = nn.Module()
        vision.config = SimpleNamespace(hidden_size=width)
        vision.embeddings = FakeEmbeddings(width)
        vision.pre_layrnorm = nn.Identity()
        vision.post_layernorm = nn.Identity()
        vision.encoder = nn.Module()
        vision.encoder.layers = nn.ModuleList([RecordingLayer()])
        prompted = PromptedTinyCLIPVision(vision, nn.Identity())

        with self.assertRaisesRegex(ValueError, "Visual prompt"):
            prompted(torch.randn(1, 3, 16, 16), torch.randn(2, width + 1))


if __name__ == "__main__":
    unittest.main()
