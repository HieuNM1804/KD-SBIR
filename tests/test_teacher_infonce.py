import unittest

import torch

from src.losses import diagonal_teacher_infonce_loss


class DiagonalTeacherInfoNCETests(unittest.TestCase):
    def test_aligned_diagonal_has_lower_loss_than_swapped_photos(self):
        sketches = torch.eye(4)
        aligned_photos = torch.eye(4)
        swapped_photos = aligned_photos[[1, 0, 3, 2]]

        aligned = diagonal_teacher_infonce_loss(sketches, aligned_photos)
        swapped = diagonal_teacher_infonce_loss(sketches, swapped_photos)

        self.assertLess(aligned.item(), swapped.item())

    def test_loss_is_symmetric_between_modalities(self):
        generator = torch.Generator().manual_seed(42)
        sketches = torch.randn(8, 16, generator=generator)
        photos = torch.randn(8, 16, generator=generator)

        forward = diagonal_teacher_infonce_loss(sketches, photos, 0.1)
        reverse = diagonal_teacher_infonce_loss(photos, sketches, 0.1)

        self.assertTrue(torch.allclose(forward, reverse))

    def test_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            diagonal_teacher_infonce_loss(torch.eye(2), torch.eye(2), 0.0)


if __name__ == "__main__":
    unittest.main()
