import unittest

import torch

from src.text_prompts import (
    MultiAspectPromptGenerator,
    attention_diversity_loss,
    multi_aspect_infonce_loss,
    multi_aspect_similarity,
    relational_logits_kd_loss,
)


class MultiAspectPromptTests(unittest.TestCase):
    def test_generator_shapes_and_text_only_gradients(self):
        patches = torch.randn(3, 49, 32, requires_grad=True)
        generator = MultiAspectPromptGenerator(
            visual_width=32,
            text_width=24,
            latent_width=16,
            aspects=4,
            context_tokens=4,
            heads=4,
            dropout=0.0,
            gate_init=0.1,
            seed=42,
            initial_context=torch.randn(4, 24),
        )
        contexts, attention = generator(patches)
        self.assertEqual(contexts.shape, (3, 4, 4, 24))
        self.assertEqual(attention.shape, (3, 4, 49))
        contexts.sum().backward()
        self.assertIsNone(patches.grad)
        self.assertTrue(any(p.grad is not None for p in generator.parameters()))

    def test_similarity_does_not_reward_duplicate_aspects(self):
        query = torch.tensor([[1.0, 0.0]])
        one = torch.tensor([[[1.0, 0.0]]])
        four = one.expand(-1, 4, -1).clone()
        score_one = multi_aspect_similarity(query, one, 0.1)
        score_four = multi_aspect_similarity(query, four, 0.1)
        self.assertTrue(torch.allclose(score_one, score_four, atol=1e-6))

    def test_exact_instance_loss_uses_the_100_photo_gallery(self):
        query = torch.eye(4, 8)
        gallery = torch.randn(100, 4, 8)
        for index in range(4):
            gallery[index] = query[index]
        loss, logits = multi_aspect_infonce_loss(
            query,
            gallery,
            torch.arange(4),
            instance_temperature=0.07,
            aspect_temperature=0.1,
        )
        self.assertEqual(logits.shape, (4, 100))
        self.assertTrue(torch.isfinite(loss))

    def test_diversity_and_relational_kd_are_zero_at_the_optimum(self):
        attention = torch.eye(4).unsqueeze(0)
        self.assertAlmostEqual(attention_diversity_loss(attention).item(), 0.0)
        logits = torch.randn(5, 100)
        self.assertAlmostEqual(
            relational_logits_kd_loss(logits, logits, 1.0).item(),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
