"""Check KL values, student gradients, and frozen teacher targets on CPU."""

from types import SimpleNamespace
import unittest

import torch

from src.losses import image_text_kd_loss, loss_fn, relational_kd_loss


def reference_terms(sq, sc, tq, tc, temperature):
    # Independent probability construction: do not use the production losses.
    def log_distribution(query, candidates):
        query = query / query.norm(dim=-1, keepdim=True)
        candidates = candidates / candidates.norm(dim=-1, keepdim=True)
        logits = query @ candidates.t() / temperature
        return logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    sl = log_distribution(sq, sc)
    tl = log_distribution(tq.detach(), tc.detach())
    forward = (tl.exp() * (tl - sl)).sum(-1).mean()
    reverse = (sl.exp() * (sl - tl)).sum(-1).mean()
    return forward, reverse


class SymmetricKDTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(19)
        # Different query/candidate counts and feature widths prevent confusing
        # KL direction with a transposed retrieval matrix or feature matching.
        self.inputs = tuple(torch.randn(*shape, generator=generator)
                            for shape in ((4, 5), (6, 5), (4, 7), (6, 7)))

    def test_domain_averages_both_kl_directions_on_photo_candidates(self):
        forward, reverse = reference_terms(*self.inputs, 0.07)
        self.assertGreater(abs(float(forward - reverse)), 1e-3)
        actual = relational_kd_loss(*self.inputs, 0.07)
        torch.testing.assert_close(actual, (forward + reverse) / 2)
        self.assertGreater(abs(float(actual - forward)), 1e-3)
        self.assertGreater(abs(float(actual - (forward + reverse))), 1e-3)

    def test_both_kd_functions_keep_reverse_probability_gradients(self):
        for function, temperature in ((relational_kd_loss, 0.07),
                                      (image_text_kd_loss, 0.15),
                                      (image_text_kd_loss, 0.02)):
            with self.subTest(loss=function.__name__, temperature=temperature):
                values = tuple(value.clone().requires_grad_() for value in self.inputs)
                actual = function(*values, temperature)
                actual.backward()
                forward, reverse = reference_terms(*values, temperature)
                expected = torch.autograd.grad((forward + reverse) / 2, values[:2])
                for value, gradient in zip(values[:2], expected):
                    torch.testing.assert_close(value.grad, gradient, rtol=2e-5, atol=2e-5)
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertGreater(float(value.grad.norm()), 0)
                self.assertIsNone(values[2].grad)
                self.assertIsNone(values[3].grad)

    def test_text_losses_average_both_kl_terms_at_each_temperature(self):
        for temperature in (0.15, 0.02):
            with self.subTest(temperature=temperature):
                forward, reverse = reference_terms(*self.inputs, temperature)
                actual = image_text_kd_loss(*self.inputs, temperature)
                torch.testing.assert_close(actual, (forward + reverse) / 2)

    def test_equal_distributions_have_zero_loss(self):
        query, candidates = self.inputs[:2]
        for function in (relational_kd_loss, image_text_kd_loss):
            actual = function(query, candidates, query, candidates, 0.15)
            torch.testing.assert_close(actual, torch.zeros(()), atol=1e-6, rtol=0)

    def test_combined_objective_keeps_weights_and_separate_text_temperatures(self):
        sq, sp, tq, tp = self.inputs
        generator = torch.Generator().manual_seed(31)
        sst, spt, tst, tpt = (torch.randn(*shape, generator=generator)
                              for shape in ((3, 5), (3, 5), (3, 7), (3, 7)))
        args = SimpleNamespace(lambda_domain=3.0, lambda_modality=1.0,
                               kd_temperature=0.07, photo_text_kd_temperature=0.15,
                               sketch_text_kd_temperature=0.02)
        actual, parts = loss_fn(args, (sp, sq, tp, tq, True, sst, spt, tst, tpt))
        domain = sum(reference_terms(sq, sp, tq, tp, 0.07)) / 2
        photo = sum(reference_terms(sp, spt, tp, tpt, 0.15)) / 2
        sketch = sum(reference_terms(sq, sst, tq, tst, 0.02)) / 2
        torch.testing.assert_close(parts['domain_kd'], domain)
        torch.testing.assert_close(parts['modality_kd'], photo + sketch)
        torch.testing.assert_close(actual, 3 * domain + photo + sketch)


if __name__ == '__main__':
    unittest.main()
