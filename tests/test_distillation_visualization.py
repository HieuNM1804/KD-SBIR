"""Run the standalone Kaggle cell on a tiny real CLIP and aligned fake caches."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from clip.model import CLIP
from src.dataset import TrainDataset, WorkerInvariantSampler
from src.losses import loss_fn
from src.model import CustomCLIP, default_teacher_cache_path
from src.teacher_ensemble import TEACHERS
from test_teacher_ensemble import config


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_cell_all_relations_matches_training_loss_and_batch(self):
        project = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("matrix_cell", project / "test/kaggle_distillation_matrices.py")
        cell = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cell)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for modality in ("sketch", "photo"):
                for category in ("first_seen_class", "second_seen_class"):
                    directory = root / modality / category
                    directory.mkdir(parents=True)
                    for index in range(3):
                        Image.new("RGB", (29, 35), (index * 80, 30, 90)).save(directory / f"{index}.png")
            args = config(root=str(root), teacher_cache_dir=str(root / "cache"))
            dataset = TrainDataset(args)
            backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                            vision_width=64, vision_patch_size=16, context_length=77,
                            vocab_size=49408, transformer_width=64, transformer_heads=1,
                            transformer_layers=1)
            owner = CustomCLIP(args, backbone, dataset.all_categories)
            checkpoint = root / "student.ckpt"
            torch.save({"state_dict": {"model." + k: v for k, v in owner.state_dict().items()}}, checkpoint)
            generator = torch.Generator().manual_seed(13)
            payloads = []
            for key in args.teacher_models:
                single = copy.copy(args)
                single.teacher_models = [key]
                owner.cfg = single
                dim = TEACHERS[key]["output_dim"]
                payload = dict(metadata=owner._teacher_cache_metadata(dataset),
                               teacher_sketch_features=torch.randn(len(dataset), dim, generator=generator),
                               teacher_photo_features=torch.randn(len(dataset.all_photo_paths), dim, generator=generator),
                               teacher_sketch_text=torch.randn(2, dim, generator=generator),
                               teacher_photo_text=torch.randn(2, dim, generator=generator))
                path = Path(default_teacher_cache_path(single, dataset))
                path.parent.mkdir(exist_ok=True)
                torch.save(payload, path)
                payloads.append(payload)
            cell.PROJECT, cell.CHECKPOINT_PATH, cell.ROOT = project, checkpoint, root
            cell.CACHE_DIR, cell.OUTPUT_DIR = root / "cache", root / "output"
            cell.MAX_SIZE, cell.BATCH_SIZE, cell.EPOCH, cell.BATCH_INDEX = 32, 4, 3, 0
            cell.TEACHER_WEIGHTS = args.teacher_ensemble_weights
            cell.TEACHER_TRAINING = {k: getattr(args, k) for k in cell.TEACHER_TRAINING}
            si, pi, sketches, photos, labels = cell.select_batch(dataset, 4, 3, 0)
            sampler = WorkerInvariantSampler(dataset, args.seed)
            sampler.epoch = 3
            self.assertEqual(si, [i for _, i in list(sampler)[:4]])
            packed_sk = torch.cat([p["teacher_sketch_features"] for p in payloads], -1)
            packed_ph = torch.cat([p["teacher_photo_features"] for p in payloads], -1)
            dataset.set_teacher_features(packed_sk, packed_ph)
            for i, sk_index in enumerate(si):
                sample = dataset[(3, sk_index)]
                torch.testing.assert_close(sample[2], packed_ph[pi[i]], rtol=0, atol=0)
                self.assertEqual(labels[i], sample[4])
            with patch("src.model.open_clip.create_model", side_effect=AssertionError("Teacher loading forbidden")), \
                 patch("src.model.clip.download_model", side_effect=AssertionError("Downloads forbidden")), \
                 patch.object(torch.cuda, "is_available", return_value=False), patch.object(plt, "show"):
                restored = cell.restore_student(checkpoint, args, dataset.all_categories, torch.device("cpu"))
                with torch.inference_mode():
                    photo_features = cell.encode_images(restored, photos, "photo", 32, torch.device("cpu"))
                    sketch_features = cell.encode_images(restored, sketches, "sketch", 32, torch.device("cpu"))
                    sk_text = restored.get_student_text_features("sketch")
                    ph_text = restored.get_student_text_features("photo")
                    _, training_losses = loss_fn(args, (photo_features, sketch_features, packed_ph[pi], packed_sk[si],
                                                       True, sk_text, ph_text,
                                                       torch.cat([p["teacher_sketch_text"] for p in payloads], -1),
                                                       torch.cat([p["teacher_photo_text"] for p in payloads], -1)))
                for relation in ("sketch_photo", "sketch_text", "photo_text"):
                    cell.RELATION = relation
                    teacher, student, metadata = cell.main()
                    np.testing.assert_allclose(teacher.sum(-1), 1, atol=1e-6)
                    np.testing.assert_allclose(student.sum(-1), 1, atol=1e-6)
                    self.assertEqual(teacher.shape, (4, 4 if relation == "sketch_photo" else 2))
                    expected = training_losses["domain_kd" if relation == "sketch_photo" else relation + "_kd"]
                    self.assertAlmostEqual(metadata["unweighted_kd_loss"], expected.item(), places=6)
                    left = "photo" if relation == "photo_text" else "sketch"
                    reference = sum(w * F.softmax(F.normalize(p[f"teacher_{left}_features"][pi if left == "photo" else si], dim=-1)
                                                 @ F.normalize(p["teacher_photo_features"][pi] if relation == "sketch_photo"
                                                               else p[f"teacher_{left}_text"], dim=-1).t()
                                                 / cell.TEMPERATURES[relation], -1)
                                    for w, p in zip(args.teacher_ensemble_weights, payloads))
                    np.testing.assert_allclose(teacher, reference.numpy(), atol=1e-6)
                    stem = cell.OUTPUT_DIR / f"{relation}_epoch3_batch0"
                    self.assertTrue(stem.with_suffix(".png").is_file())
                    with np.load(stem.with_suffix(".npz")) as saved:
                        np.testing.assert_array_equal(saved["teacher"], teacher)
                    self.assertEqual(json.loads(stem.with_suffix(".json").read_text())["photo_paths"], photos)
                    axes = plt.gcf().axes
                    self.assertEqual(axes[0].images[0].get_clim(), axes[1].images[0].get_clim())
                    plt.close("all")
                wrong = copy.copy(args)
                wrong.teacher_prompt_lr = 7.0
                cell.CACHE_PATHS = {key: default_teacher_cache_path(copy_with_key(args, key), dataset)
                                    for key in args.teacher_models}
                with self.assertRaisesRegex(ValueError, "Cache .* differs"):
                    cell.load_targets(wrong, dataset, si, pi, "sketch_photo")
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    cell.select_batch(dataset, 4, 0, 1)


def copy_with_key(args, key):
    single = copy.copy(args)
    single.teacher_models = [key]
    return single


if __name__ == "__main__":
    unittest.main()
