"""Regression checks for the dedicated DFN2B L/14 S39B teacher branch."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src import model


class DummyTeacher:
    def __init__(self):
        self.requires_grad_value = None

    def eval(self):
        return self

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self


class Dfn2bTeacherTest(unittest.TestCase):
    def test_dedicated_teacher_identity(self):
        self.assertEqual(model.TEACHER_MODEL, "ViT-L-14")
        self.assertEqual(model.TEACHER_PRETRAINED, "dfn2b_s39b")
        self.assertEqual(model.TEACHER_OUTPUT_DIM, 768)

    @mock.patch.object(model.open_clip, "get_tokenizer")
    @mock.patch.object(model.open_clip, "create_model")
    def test_loader_uses_dfn2b_s39b_checkpoint(
        self,
        create_model,
        get_tokenizer,
    ):
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

        self.assertIs(loaded, teacher)
        create_model.assert_called_once_with(
            "ViT-L-14",
            pretrained="dfn2b_s39b",
            precision="fp16",
            device=model.device,
        )
        get_tokenizer.assert_called_once_with("ViT-L-14")
        self.assertIs(loaded.text_tokenizer, tokenizer)
        self.assertEqual(loaded.output_dim, 768)
        self.assertFalse(loaded.requires_grad_value)


if __name__ == "__main__":
    unittest.main()
