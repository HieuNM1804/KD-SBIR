"""Regression checks for the dedicated OpenAI CLIP L/14 teacher branch."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src import model


class DummyTeacher:
    def eval(self):
        return self

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self


class OpenAiL14TeacherTest(unittest.TestCase):
    def test_teacher_identity(self):
        self.assertEqual(model.TEACHER_MODEL, "ViT-L-14")
        self.assertEqual(model.TEACHER_PRETRAINED, "openai")
        self.assertEqual(model.TEACHER_OUTPUT_DIM, 768)

    @mock.patch.object(model.open_clip, "get_tokenizer")
    @mock.patch.object(model.open_clip, "create_model")
    def test_loader_uses_openai_l14(self, create_model, get_tokenizer):
        teacher = DummyTeacher()
        tokenizer = object()
        create_model.return_value = teacher
        get_tokenizer.return_value = tokenizer
        args = SimpleNamespace(
            teacher_cache_path="",
            rebuild_teacher_cache=False,
            lambda_domain=1.0,
            lambda_modality=1.0,
            teacher_pretrain_epochs=1,
        )

        loaded = model._load_teacher(args)

        create_model.assert_called_once_with(
            "ViT-L-14",
            pretrained="openai",
            precision="fp16",
            device=model.device,
        )
        get_tokenizer.assert_called_once_with("ViT-L-14")
        self.assertIs(loaded, teacher)
        self.assertEqual(loaded.output_dim, 768)
        self.assertFalse(loaded.requires_grad_value)


if __name__ == "__main__":
    unittest.main()
