import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.experiment_results import best_validation_epoch
from src.model import CustomCLIP, ZS_SBIR, default_teacher_cache_path
from src.text_prompts import TEXT_PROMPT_PAIRS, class_texts, prompt_pair_config

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prompt_sweep", PROJECT / "test" / "kaggle_text_prompt_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def configuration(pair="baseline"):
    return SimpleNamespace(
        text_prompt_pair=pair, n_ctx_visual=3, prompt_depth=2, seed=42,
        lambda_modality=1.0, lambda_domain=3.0, kd_temperature=0.07,
        photo_text_kd_temperature=0.15, sketch_text_kd_temperature=0.02,
        teacher_cache_path="", rebuild_teacher_cache=False,
        teacher_pretrain_epochs=0, teacher_n_ctx_visual=10,
        teacher_prompt_depth=12, teacher_prompt_std=0.02,
        teacher_prompt_seed=42, teacher_prompt_gradient_checkpointing=True,
        teacher_prompt_lr=0.03, teacher_momentum=0.9,
        teacher_weight_decay=0.001, teacher_pretrain_batch_size=64,
        lambda_teacher_retrieval=1.5, teacher_triplet_margin=0.2,
        teacher_scheduler_step_size=5, teacher_scheduler_gamma=0.1,
        root="/dataset", dataset="sketchy_2", teacher_cache_dir="/cache",
        results_path="", backbone="ViT-B/32",
    )


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.dtype = torch.float32
        self.visual = nn.Module()
        self.visual.ln_pre = nn.LayerNorm(8)
        self.visual.transformer = SimpleNamespace(layers=2)
        self.teacher_texts = []

    def text_tokenizer(self, texts):
        self.teacher_texts.extend(texts)
        return torch.ones(len(texts), 4, dtype=torch.int64)

    def encode_text(self, tokens):
        return torch.ones(len(tokens), 4)


class PromptIntegrationTests(unittest.TestCase):
    def test_each_pair_is_used_by_both_teacher_and_student(self):
        categories = ("cat", "red_fox")
        for pair in TEXT_PROMPT_PAIRS:
            with self.subTest(pair=pair):
                teacher = FakeCLIP()
                student_inputs = []

                def tokenize(texts):
                    student_inputs.append(list(texts))
                    return torch.ones(len(texts), 4, dtype=torch.int64)

                with patch("src.model.clip.tokenize", side_effect=tokenize):
                    model = CustomCLIP(configuration(pair), FakeCLIP(), categories, teacher)
                sketch, photo = model.get_teacher_text_features()
                self.assertEqual(teacher.teacher_texts, student_inputs[1] + student_inputs[0])
                self.assertEqual(student_inputs[0], class_texts(categories, "photo", pair))
                self.assertEqual(student_inputs[1], class_texts(categories, "sketch", pair))
                self.assertEqual(sketch.shape, (2, 4))
                self.assertEqual(photo.shape, (2, 4))
                self.assertIsNot(model.photo_visual_prompt, model.sketch_visual_prompt)
                self.assertTrue(all(not p.requires_grad for p in model.clip_model.parameters()))

    def test_prompt_pair_changes_cache_identity_and_old_cache_is_rejected(self):
        dataset = SimpleNamespace(
            max_size=224, all_categories=("cat",),
            all_sketches_path=["/dataset/sketch/cat/1.png"],
            all_photo_paths=["/dataset/photo/cat/1.jpg"],
        )
        caches = {default_teacher_cache_path(configuration(pair), dataset) for pair in TEXT_PROMPT_PAIRS}
        self.assertEqual(len(caches), 5)
        cfg = configuration()
        with patch("src.model.clip.tokenize", return_value=torch.ones(1, 4, dtype=torch.int64)):
            model = CustomCLIP(cfg, FakeCLIP(), dataset.all_categories)
        old_metadata = model._teacher_cache_metadata(dataset)
        old_metadata.pop("text_prompt_pair")
        with tempfile.TemporaryDirectory() as directory:
            cfg.teacher_cache_path = str(Path(directory) / "old.pt")
            torch.save({
                "metadata": old_metadata,
                "teacher_sketch_features": torch.ones(1, 4),
                "teacher_photo_features": torch.ones(1, 4),
                "teacher_sketch_text": torch.ones(1, 4),
                "teacher_photo_text": torch.ones(1, 4),
            }, cfg.teacher_cache_path)
            with self.assertRaisesRegex(RuntimeError, "text_prompt_pair"):
                model._load_persistent_teacher_cache(dataset)

    def test_sanity_validation_does_not_enter_epoch_history(self):
        cfg = configuration()
        with patch("src.model._load_clip_model", return_value=FakeCLIP()), \
             patch("src.model._load_teacher", return_value=None), \
             patch("src.model.clip.tokenize", return_value=torch.ones(1, 4, dtype=torch.int64)):
            model = ZS_SBIR(cfg, ["cat"])
        model.trainer = SimpleNamespace(
            sanity_checking=True, global_step=1, current_epoch=0,
            callback_metrics={},
        )
        with patch.object(model, "log"), patch("src.model._retrieval_metrics", return_value=(
            torch.tensor(0.8), torch.tensor(0.7), 200, 200,
        )):
            for sanity_checking in (True, False):
                model.trainer.sanity_checking = sanity_checking
                model.val_step_outputs_sk.append((torch.ones(1, 4), torch.tensor([0])))
                model.val_step_outputs_ph.append((torch.ones(1, 4), torch.tensor([0])))
                model.on_validation_epoch_end()
                self.assertEqual(len(model.validation_history), 0 if sanity_checking else 1)
        self.assertEqual(model.validation_history[0]["epoch"], 1)


class ResultTests(unittest.TestCase):
    def test_lightning_checkpoint_matches_the_best_trained_epoch(self):
        cfg = configuration()
        cfg.lr, cfg.momentum, cfg.weight_decay = 0.01, 0.9, 0.00001
        with patch("src.model._load_clip_model", return_value=FakeCLIP()), \
             patch("src.model._load_teacher", return_value=None), \
             patch("src.model.clip.tokenize", return_value=torch.ones(1, 4, dtype=torch.int64)):
            model = ZS_SBIR(cfg, ["cat"])

        def training_step(current_model, batch, batch_index):
            loss = current_model.model.photo_visual_prompt.ctx.square().mean()
            current_model.log("train_loss", loss, on_epoch=True, on_step=False)
            return loss

        def validation_step(current_model, batch, batch_index, dataloader_idx):
            outputs = current_model.val_step_outputs_sk if dataloader_idx == 0 else current_model.val_step_outputs_ph
            outputs.append(batch)

        loader = DataLoader(TensorDataset(torch.ones(2, 4), torch.tensor([0, 0])), batch_size=2)
        metrics = [(0.95, 0.99), (0.9, 0.4), (0.3, 0.8)]
        with tempfile.TemporaryDirectory() as directory:
            callback = ModelCheckpoint(dirpath=directory, monitor="precision", mode="max", save_top_k=1)
            trainer = Trainer(
                accelerator="cpu", devices=1, max_epochs=2, logger=False,
                enable_progress_bar=False, enable_model_summary=False,
                callbacks=[callback], num_sanity_val_steps=1,
            )
            with patch.object(ZS_SBIR, "training_step", training_step), \
                 patch.object(ZS_SBIR, "validation_step", validation_step), \
                 patch("src.model._retrieval_metrics", side_effect=[
                     (torch.tensor(m), torch.tensor(p), 200, 200) for m, p in metrics
                 ]):
                trainer.fit(model, loader, [loader, loader])
            self.assertEqual(len(model.validation_history), 2)
            selected = best_validation_epoch(model.validation_history)
            self.assertEqual(selected["epoch"], 2)
            self.assertAlmostEqual(selected["precision"], 0.8)
            self.assertAlmostEqual(selected["mAP"], 0.3)
            self.assertAlmostEqual(float(callback.best_model_score), selected["precision"])
            saved = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], selected["epoch"] - 1)

    def test_selects_map_from_the_precision_winning_epoch_and_keeps_first_tie(self):
        history = [
            {"epoch": 1, "precision": 0.6, "mAP": 0.9},
            {"epoch": 2, "precision": 0.8, "mAP": 0.5},
            {"epoch": 3, "precision": 0.8, "mAP": 0.7},
        ]
        self.assertEqual(best_validation_epoch(history), history[1])
        with self.assertRaises(ValueError):
            best_validation_epoch([])
        with self.assertRaises(ValueError):
            best_validation_epoch([{"precision": float("nan"), "mAP": 0.5}])

    def test_user_settings_and_templates_are_forwarded_to_all_five_commands(self):
        args = SimpleNamespace(root="/dataset", dataset="sketchy_2", epochs=5, workers=8)
        for pair in TEXT_PROMPT_PAIRS:
            command = sweep.command_for_pair(args, pair, Path("/output"))
            for flag, value in {
                "--text_prompt_pair": pair, "--dataset": "sketchy_2",
                "--epochs": "5", "--workers": "8", "--teacher_pretrain_epochs": "1",
                "--teacher_prompt_lr": "3e-2", "--teacher_n_ctx_visual": "10",
                "--lambda_domain": "3.0", "--lambda_modality": "1.0",
                "--photo_text_kd_temperature": "0.15", "--sketch_text_kd_temperature": "0.02",
                "--lr": "1e-2", "--weight_decay": "1e-5", "--seed": "42",
            }.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--rebuild_teacher_cache", command)

    def test_mocked_sweep_exports_winner_and_resume_skips_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "photo").mkdir(parents=True)
            (root / "sketch").mkdir()
            output = Path(directory) / "results"
            scores = dict(zip(TEXT_PROMPT_PAIRS, (0.4, 0.5, 0.8, 0.6, 0.7)))

            def fake_training(command, log_path):
                pair = command[command.index("--text_prompt_pair") + 1]
                path = Path(command[command.index("--results_path") + 1])
                checkpoint = output / "runs" / f"{pair}.ckpt"
                checkpoint.write_bytes(pair.encode("utf-8"))
                history = [
                    {"epoch": epoch, "precision": scores[pair] if epoch == 2 else 0.1,
                     "mAP": 0.3 if epoch == 2 else 0.9, "p_k": 200, "map_k": 200}
                    for epoch in range(1, 6)
                ]
                path.write_text(json.dumps({
                    "status": "completed", "dataset": "sketchy_2",
                    "text_prompt_pair": prompt_pair_config(pair),
                    "history": history, "best_epoch": history[1],
                    "selection_metric": "P@200", "best_checkpoint": str(checkpoint),
                }), encoding="utf-8")

            arguments = ["--root", str(root), "--output_dir", str(output)]
            with patch.object(sweep, "run_training", side_effect=fake_training) as run:
                sweep.main(arguments)
                self.assertEqual(run.call_count, 5)
            best = json.loads((output / "best_run.json").read_text(encoding="utf-8"))
            self.assertEqual(best["status"], "completed")
            self.assertEqual(best["winner"]["text_prompt_pair"]["name"], "shows")
            self.assertEqual(best["winner"]["best_epoch"]["mAP"], 0.3)
            self.assertEqual((output / "best.ckpt").read_bytes(), b"shows")
            with patch.object(sweep, "run_training") as run:
                sweep.main([*arguments, "--resume"])
                run.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                sweep.main([*arguments, "--resume", "--workers", "0"])


if __name__ == "__main__":
    unittest.main()
