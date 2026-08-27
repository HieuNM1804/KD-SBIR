import unittest

import torch

from src.losses import multipositive_teacher_infonce_loss


class MultiPositiveTeacherInfoNCETests(unittest.TestCase):
    def test_within_class_photo_swap_remains_positive(self):
        sketches = torch.eye(3)
        aligned_photos = torch.eye(3)
        within_class_swap = aligned_photos[[1, 0, 2]]
        labels = torch.tensor([0, 0, 1])

        aligned = multipositive_teacher_infonce_loss(
            sketches, aligned_photos, labels
        )
        swapped = multipositive_teacher_infonce_loss(
            sketches, within_class_swap, labels
        )

        self.assertTrue(torch.allclose(aligned, swapped))

    def test_cross_class_swap_has_higher_loss(self):
        sketches = torch.eye(4)
        photos = torch.eye(4)
        labels = torch.tensor([0, 0, 1, 1])
        cross_class_swap = photos[[2, 3, 0, 1]]

        aligned = multipositive_teacher_infonce_loss(
            sketches, photos, labels
        )
        swapped = multipositive_teacher_infonce_loss(
            sketches, cross_class_swap, labels
        )

        self.assertLess(aligned.item(), swapped.item())

    def test_loss_is_symmetric_between_modalities(self):
        generator = torch.Generator().manual_seed(42)
        sketches = torch.randn(8, 16, generator=generator)
        photos = torch.randn(8, 16, generator=generator)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        forward = multipositive_teacher_infonce_loss(
            sketches, photos, labels, 0.1
        )
        reverse = multipositive_teacher_infonce_loss(
            photos, sketches, labels, 0.1
        )

        self.assertTrue(torch.allclose(forward, reverse))

    def test_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            multipositive_teacher_infonce_loss(
                torch.eye(2), torch.eye(2), torch.tensor([0, 1]), 0.0
            )


if __name__ == "__main__":
    unittest.main()
