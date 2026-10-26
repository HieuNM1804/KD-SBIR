"""Accounting tests and a real tiny CPU run through main; no pretrained downloads."""
import csv
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.benchmark_training_cost import build_training_argv
from src.training_cost import TrainingCostRecorder, instrument_training, local_training_weights, scalar_metrics


class TrainingCostTests(unittest.TestCase):
    def test_nested_validation_is_not_double_counted_as_cache(self):
        clock = SimpleNamespace(now=0.0)
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", new=io.StringIO()):
            recorder = TrainingCostRecorder(folder, clock=lambda: clock.now, sync=lambda: None)
            with recorder.stage("pipeline"):
                with recorder.stage("teacher_preparation"):
                    with recorder.stage("teacher_adaptation"):
                        clock.now += 8
                        with recorder.stage("teacher_validation"):
                            clock.now += 2
                    clock.now += 3
                with recorder.stage("student_distillation"):
                    clock.now += 4
                    with recorder.stage("student_validation"):
                        clock.now += 1
            recorder.finish("completed")
            summary = recorder.report["summary"]
            self.assertEqual(summary["pipeline"]["seconds"], 18)
            self.assertEqual(summary["teacher_adaptation"]["seconds"], 10)
            self.assertEqual(summary["teacher_feature_cache_build"]["seconds"], 3)
            self.assertEqual(summary["student_distillation"]["seconds"], 5)
            self.assertEqual(recorder.report["training_and_cache_gpu_hours"], 18 / 3600)
            self.assertTrue(recorder.report["full_fresh_training_measurement"])

    def test_cache_hit_is_not_reported_as_teacher_training(self):
        clock = SimpleNamespace(now=0.0)
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", new=io.StringIO()):
            recorder = TrainingCostRecorder(folder, clock=lambda: clock.now, sync=lambda: None)
            recorder.report["cache_hit"] = True
            with recorder.stage("teacher_preparation"):
                with recorder.stage("teacher_cache_load"):
                    clock.now += 3
            with recorder.stage("student_distillation"):
                clock.now += 2
            recorder.finish("completed")
            self.assertIsNone(recorder.report["summary"]["teacher_adaptation"]["seconds"])
            self.assertIsNone(recorder.report["summary"]["teacher_feature_cache_build"]["seconds"])
            self.assertEqual(recorder.report["summary"]["teacher_cache_load"]["seconds"], 3)
            self.assertFalse(recorder.report["full_fresh_training_measurement"])
            self.assertIsNone(recorder.report["training_and_cache_gpu_hours"])

    def test_interrupted_run_saves_partial_timings(self):
        clock = SimpleNamespace(now=0.0)
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", new=io.StringIO()):
            recorder = TrainingCostRecorder(folder, clock=lambda: clock.now, sync=lambda: None)
            try:
                with recorder.stage("teacher_adaptation"):
                    clock.now += 7
                    raise KeyboardInterrupt()
            except KeyboardInterrupt:
                recorder.finish("interrupted")
            report = json.loads((Path(folder) / "training_cost.json").read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["summary"]["teacher_adaptation"]["seconds"], 7)
            self.assertEqual(report["summary"]["teacher_adaptation"]["status"], "interrupted")
            self.assertFalse(report["full_fresh_training_measurement"])
            with (Path(folder) / "training_cost.csv").open() as stream:
                self.assertTrue(list(csv.DictReader(stream)))

    def test_reusing_existing_cache_requires_explicit_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.pt"
            path.touch()
            args = SimpleNamespace(teacher_cache_dir=Path(folder), teacher_cache_path=path,
                                   allow_cached_teacher=False, rebuild_teacher_cache=False,
                                   teacher_pretrain_epochs=1, exp_name="cost")
            with self.assertRaises(ValueError):
                build_training_argv(args, ["--dataset", "sketchy_2"], "run1")
            args.allow_cached_teacher = True
            argv, cache = build_training_argv(args, ["--dataset", "sketchy_2"], "run1")
            self.assertEqual(cache, path.resolve())
            self.assertIn("cost_run1", argv)
            self.assertEqual(argv[argv.index("--teacher_pretrain_epochs") + 1], "1")
            args.teacher_cache_path = None
            _, first = build_training_argv(args, [], "run1")
            _, second = build_training_argv(args, [], "run2")
            self.assertNotEqual(first, second)

    def test_local_loaders_redirect_only_main_models_and_restore_afterward(self):
        def create(model_name, pretrained=None, precision="fp32", device="cpu"):
            return model_name, pretrained, precision, device
        def download(backbone):
            raise AssertionError("Network loader must not run")
        module = SimpleNamespace(open_clip=SimpleNamespace(create_model=create),
                                 clip=SimpleNamespace(download_model=download))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("dfn5b_h14/open_clip_pytorch_model.bin", "clip_b32/ViT-B-32.pt"):
                path = root / name
                path.parent.mkdir(parents=True)
                path.touch()
            with local_training_weights(module, root):
                actual = module.open_clip.create_model("ViT-H-14-quickgelu", pretrained="dfn5b",
                                                       precision="fp16", device="cuda")
                self.assertEqual(actual[1], str(root.resolve() / "dfn5b_h14/open_clip_pytorch_model.bin"))
                self.assertEqual(actual[2:], ("fp16", "cuda"))
                self.assertEqual(module.clip.download_model("ViT-B/32"), str(root.resolve() / "clip_b32/ViT-B-32.pt"))
                with self.assertRaises(ValueError):
                    module.clip.download_model("ViT-L/14")
            self.assertIs(module.open_clip.create_model, create)
            self.assertIs(module.clip.download_model, download)

    def test_nonfinite_metrics_do_not_break_json(self):
        self.assertEqual(scalar_metrics({"mAP": torch.tensor(float("nan")), "precision": .5}),
                         {"mAP": None, "precision": .5})

    def test_real_main_teacher_cache_student_smoke_on_tiny_cpu_models(self):
        """Execute the original loss, teacher optimizer, feature cache and Lightning fit."""
        from PIL import Image
        from clip.model import CLIP
        from clip import clip
        import open_clip
        from pytorch_lightning import Callback, Trainer
        from pytorch_lightning.loggers import CSVLogger
        from src import model as model_module
        from src.data_config import UNSEEN_CLASSES

        torch.set_num_threads(2)
        student = CLIP(embed_dim=512, image_resolution=32, vision_layers=1, vision_width=64,
                       vision_patch_size=8, context_length=77, vocab_size=49408,
                       transformer_width=64, transformer_heads=1, transformer_layers=1)
        teacher = open_clip.model.CLIP(
            embed_dim=1024,
            vision_cfg=dict(image_size=32, patch_size=8, width=32, layers=1, head_width=8),
            text_cfg=dict(context_length=77, vocab_size=49408, width=32, heads=4, layers=1))
        teacher.eval().requires_grad_(False)
        teacher.text_tokenizer = clip.tokenize
        teacher.output_dim = 1024
        original_init = Trainer.__init__
        def cpu_trainer(self, *args, **kwargs):
            kwargs.update(accelerator="cpu", devices=1, enable_model_summary=False)
            original_init(self, *args, **kwargs)
        original_teacher = model_module.CustomCLIP._pretrain_teacher_prompts
        original_fit = Trainer.fit
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            names = ["seen_a", "seen_b", *UNSEEN_CLASSES["sketchy_2"]]
            for modality in ("photo", "sketch"):
                for index, name in enumerate(names):
                    directory = data / modality / name
                    directory.mkdir(parents=True)
                    for number in range(2 if name.startswith("seen_") else 1):
                        Image.new("RGB", (32, 32), ((index * 19 + number * 37) % 255, 70, 130)).save(directory / f"{number}.png")
            cache = root / "teacher.pt"
            recorder = TrainingCostRecorder(root / "report", metadata={"tiny_cpu_test": True}, sync=lambda: None)
            argv = ["src.train", "--root", str(data), "--dataset", "sketchy_2", "--max_size", "32",
                    "--epochs", "1", "--workers", "0", "--batch_size", "4", "--test_batch_size", "32",
                    "--n_ctx_visual", "3", "--prompt_depth", "1", "--teacher_n_ctx_visual", "3",
                    "--teacher_prompt_depth", "1", "--teacher_pretrain_epochs", "1",
                    "--teacher_pretrain_batch_size", "4", "--teacher_prompt_lr", ".001",
                    "--lr", ".001", "--teacher_cache_path", str(cache),
                    "--lambda_domain", "2", "--lambda_modality", "2",
                    "--no_teacher_prompt_gradient_checkpointing"]
            try:
                os.chdir(root)
                with patch.object(model_module, "_load_clip_model", return_value=student), \
                        patch.object(model_module, "_load_teacher", return_value=teacher), \
                        patch.object(Trainer, "__init__", cpu_trainer), \
                        patch("pytorch_lightning.loggers.TensorBoardLogger", CSVLogger), \
                        patch.object(sys, "argv", argv), patch("sys.stdout", new=io.StringIO()):
                    with recorder.stage("pipeline"), instrument_training(recorder, model_module, Trainer, Callback):
                        runpy.run_module("src.train", run_name="__main__")
                    recorder.finish("completed")
                self.assertTrue(cache.is_file())
                self.assertTrue(recorder.report["full_fresh_training_measurement"])
                self.assertEqual(recorder.report["teacher_best_epoch"], 1)
                for name in ("teacher_adaptation", "teacher_validation", "teacher_feature_cache_build",
                             "student_distillation", "student_validation", "student_sanity_validation"):
                    self.assertEqual(recorder.report["summary"][name]["status"], "completed", name)
                self.assertEqual(recorder.report["dataset"]["seen_sketches"], 4)
                self.assertTrue(Path(recorder.report["student_checkpoint"]["best_model_path"]).is_file())
            finally:
                os.chdir(previous_dir)
                torch.use_deterministic_algorithms(False)
        self.assertIs(model_module.CustomCLIP._pretrain_teacher_prompts, original_teacher)
        self.assertIs(Trainer.fit, original_fit)


if __name__ == "__main__":
    unittest.main()
