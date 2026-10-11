"""Check the image-text KL direction, gradients, and weighted student loss."""

from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from src.losses import image_text_kd_loss, loss_fn, relational_kd_loss


def reference_forward_kl(student_image, student_text, teacher_image, teacher_text, temperature):
    student_logits = F.normalize(student_image.float(), dim=-1) @ F.normalize(student_text.float(), dim=-1).t() / temperature
    teacher_logits = F.normalize(teacher_image.detach().float(), dim=-1) @ F.normalize(teacher_text.detach().float(), dim=-1).t() / temperature
    student_log = student_logits - torch.logsumexp(student_logits, dim=-1, keepdim=True)
    teacher_log = teacher_logits - torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    return (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()


class ImageTextForwardKLTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(27)
        self.student_image = torch.randn(4, 5, generator=generator)
        self.student_text = torch.randn(6, 5, generator=generator)
        self.teacher_image = torch.randn(4, 7, generator=generator)
        self.teacher_text = torch.randn(6, 7, generator=generator)

    def test_forward_kl_without_reverse_term_or_half_factor(self):
        for temperature in (0.15, 0.02):
            with self.subTest(temperature=temperature):
                expected = reference_forward_kl(self.student_image, self.student_text,
                                                self.teacher_image, self.teacher_text, temperature)
                reverse = reference_forward_kl(self.teacher_image, self.teacher_text,
                                               self.student_image, self.student_text, temperature)
                actual = image_text_kd_loss(self.student_image, self.student_text,
                                           self.teacher_image, self.teacher_text, temperature)
                torch.testing.assert_close(actual, expected)
                self.assertGreater(abs(float(actual - 0.5 * (expected + reverse))), 1e-3)
                self.assertGreater(abs(float(actual - 0.5 * expected)), 1e-3)

    def test_student_gradients_match_forward_kl_and_teacher_is_detached(self):
        student = self.student_image.clone().requires_grad_()
        teacher = self.teacher_image.clone().requires_grad_()
        teacher_text = self.teacher_text.clone().requires_grad_()
        actual = image_text_kd_loss(student, self.student_text, teacher, teacher_text, 0.15)
        student_grad, teacher_grad, text_grad = torch.autograd.grad(
            actual, (student, teacher, teacher_text), allow_unused=True)
        expected_grad, = torch.autograd.grad(
            reference_forward_kl(student, self.student_text, teacher, teacher_text, 0.15), (student,))
        torch.testing.assert_close(student_grad, expected_grad)
        self.assertTrue(torch.isfinite(student_grad).all())
        self.assertGreater(float(student_grad.norm()), 0)
        self.assertIsNone(teacher_grad)
        self.assertIsNone(text_grad)

    def test_same_direction_and_reduction_as_domain_kd(self):
        actual = image_text_kd_loss(self.student_image, self.student_text,
                                   self.teacher_image, self.teacher_text, 0.07)
        domain = relational_kd_loss(self.student_image, self.student_text,
                                    self.teacher_image, self.teacher_text, 0.07)
        torch.testing.assert_close(actual, domain)

    def test_matching_distributions_have_zero_loss(self):
        actual = image_text_kd_loss(self.student_image, self.student_text,
                                   self.student_image, self.student_text, 0.15)
        self.assertLess(abs(float(actual)), 1e-6)

    def test_loss_fn_uses_forward_kl_for_both_modalities(self):
        photo = self.student_image
        sketch = self.student_image.roll(1, dims=0)
        teacher_photo = self.teacher_image
        teacher_sketch = self.teacher_image.roll(1, dims=0)
        photo_text = self.student_text
        sketch_text = self.student_text.roll(1, dims=1)
        teacher_photo_text = self.teacher_text
        teacher_sketch_text = self.teacher_text.roll(1, dims=1)
        args = SimpleNamespace(lambda_domain=3.0, lambda_modality=1.0,
                               kd_temperature=0.07, photo_text_kd_temperature=0.15,
                               sketch_text_kd_temperature=0.02)
        features = (photo, sketch, teacher_photo, teacher_sketch, True,
                    sketch_text, photo_text, teacher_sketch_text, teacher_photo_text)
        total, terms = loss_fn(args, features)
        expected_modality = reference_forward_kl(photo, photo_text, teacher_photo, teacher_photo_text, 0.15)
        expected_modality += reference_forward_kl(sketch, sketch_text, teacher_sketch, teacher_sketch_text, 0.02)
        expected_domain = relational_kd_loss(sketch, photo, teacher_sketch, teacher_photo, 0.07)
        torch.testing.assert_close(terms["modality_kd"], expected_modality)
        torch.testing.assert_close(terms["domain_kd"], expected_domain)
        torch.testing.assert_close(total, 3.0 * expected_domain + expected_modality)


if __name__ == "__main__":
    unittest.main()
