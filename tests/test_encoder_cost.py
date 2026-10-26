"""Offline tests on small real vision implementations; no weights or downloads."""
import ast
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from clip.model import VisionTransformer
from src.cost_prompts import PromptedHFVision, PromptedStudent
from src.encoder_cost_models import (
    PromptedTeacher, parameter_counts, restore_prompts, set_precision, vit_flops,
)
from src.benchmark_encoder_cost import main, measure, profile_flops


torch.set_num_threads(2)


def openai_model(image_size=32, n_ctx=3, depth=2):
    return PromptedStudent(VisionTransformer(image_size, 8, 32, 2, 4, 16),
                           32, 2, n_ctx, depth, 42).eval().requires_grad_(False)


class CostTests(unittest.TestCase):
    def check_cpu_fp16(self, model, key, spec):
        set_precision(model, key, torch.device("cpu"), "fp16")
        images = torch.randn(1, 3, 32, 32).half()
        with torch.inference_mode():
            output = model(images, "sketch")
        self.assertTrue(torch.isfinite(output).all())
        if key == "dfn5b_h14":
            self.assertEqual(model.visual.conv1.weight.dtype, torch.float16)
            self.assertEqual(next(model.controller.prompt_learner.parameters()).dtype, torch.float32)
        elif key == "clip_b32":
            self.assertEqual(model.visual.conv1.weight.dtype, torch.float16)
            self.assertEqual(model.sketch_visual_prompt.ctx.dtype, torch.float32)
        else:
            self.assertTrue(all(p.dtype == torch.float16 for p in model.visual.parameters()))
            self.assertEqual(model.sketch_visual_prompt.ctx.dtype, torch.float32)
        self.assertEqual(profile_flops(model, images, "sketch")["total"],
                         vit_flops(spec, 3, 32)["total"])
        set_precision(model, key, torch.device("cpu"), "fp32")
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_real_prompted_openai_profile_includes_attention(self):
        model = openai_model()
        spec = dict(patch=8, width=32, layers=2, mlp=128, dim=16)
        images = torch.randn(1, 3, 32, 32)
        profile = profile_flops(model, images, "sketch")
        self.assertEqual(profile["total"], vit_flops(spec, 3, 32)["total"])
        self.assertGreater(vit_flops(spec, 3, 32)["breakdown"]["attention_qk_av"], 0)
        self.assertEqual(model(images, "photo").shape, (1, 16))
        self.assertTrue(torch.backends.mha.get_fastpath_enabled())
        self.check_cpu_fp16(model, "clip_b32", spec)

    def test_real_prompted_openclip_teacher_profile(self):
        from open_clip.transformer import VisionTransformer as OpenCLIPVision
        visual = OpenCLIPVision(image_size=32, patch_size=8, width=32, layers=2,
                                heads=4, mlp_ratio=4, output_dim=16)
        model = PromptedTeacher(visual, 3, 2, 42).eval().requires_grad_(False)
        spec = dict(patch=8, width=32, layers=2, mlp=128, dim=16)
        images = torch.randn(1, 3, 32, 32)
        self.assertEqual(profile_flops(model, images, "sketch")["total"],
                         vit_flops(spec, 3, 32)["total"])
        counts = parameter_counts(model, teacher=True)
        self.assertEqual(counts["sketch_prompt_parameters"], 3 * 32 * 2)
        self.assertEqual(counts["deployed_parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(model(images, "photo").shape, (1, 16))
        self.check_cpu_fp16(model, "dfn5b_h14", spec)

    def test_real_prompted_huggingface_profile(self):
        from transformers import CLIPConfig, CLIPModel
        config = CLIPConfig(
            projection_dim=16,
            vision_config=dict(image_size=32, patch_size=8, hidden_size=32,
                               intermediate_size=64, num_hidden_layers=2, num_attention_heads=4),
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2),
        )
        config._attn_implementation = "sdpa"
        raw = CLIPModel(config)
        tower = PromptedHFVision(raw.vision_model, raw.visual_projection)
        tower.add_module("vision_owner", raw.vision_model)
        tower.add_module("projection_owner", raw.visual_projection)
        model = PromptedStudent(tower, 32, 2, 3, 2, 42).eval().requires_grad_(False)
        spec = dict(patch=8, width=32, layers=2, mlp=64, dim=16)
        images = torch.randn(1, 3, 32, 32)
        self.assertEqual(profile_flops(model, images, "sketch")["total"],
                         vit_flops(spec, 3, 32)["total"])
        counts = parameter_counts(model)
        self.assertEqual(counts["deployed_parameters"], sum(p.numel() for p in model.parameters()))
        self.assertLess(counts["deployed_parameters"], sum(p.numel() for p in raw.parameters()))
        self.assertEqual(model(images, "photo").shape, (1, 16))
        self.check_cpu_fp16(model, "tinyclip_61m", spec)

    def test_prompt_tokens_persist_beyond_replacement_depth(self):
        spec = dict(patch=8, width=32, layers=2, mlp=128, dim=16)
        images = torch.randn(1, 3, 32, 32)
        shallow = profile_flops(openai_model(depth=1), images, "sketch")["total"]
        deep = profile_flops(openai_model(depth=2), images, "sketch")["total"]
        self.assertEqual(shallow, deep)
        self.assertEqual(deep, vit_flops(spec, 3, 32)["total"])
        self.assertGreater(deep, vit_flops(spec, 0, 32)["total"])

    def test_student_checkpoint_prompts_match_original_forward(self):
        source = openai_model()
        target = openai_model()
        with torch.no_grad():
            source.sketch_visual_prompt.ctx.fill_(.4)
            source.photo_visual_prompt.ctx.fill_(-.2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "student.ckpt"
            torch.save({"state_dict": {"model." + k: v for k, v in source.state_dict().items()}}, path)
            restore_prompts(target, path)
            for modality in ("sketch", "photo"):
                torch.testing.assert_close(getattr(source, modality + "_visual_prompt").ctx,
                                           getattr(target, modality + "_visual_prompt").ctx)
            with self.assertRaises(RuntimeError):
                restore_prompts(openai_model(n_ctx=4), path)

    def test_teacher_cache_restore_does_not_load_backbone(self):
        from open_clip.transformer import VisionTransformer as OpenCLIPVision
        visual = OpenCLIPVision(image_size=32, patch_size=8, width=32, layers=2,
                                heads=4, mlp_ratio=4, output_dim=16)
        model = PromptedTeacher(visual, 3, 2, 42)
        state = {k: torch.full_like(v, .123) for k, v in model.controller.state_dict().items()}
        original = model.visual.conv1.weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.pt"
            torch.save({"teacher_prompt_state_dict": state, "metadata": {"lr": .03}}, path)
            restore_prompts(model, path, teacher=True)
            shallow = PromptedTeacher(visual, 3, 1, 42)
            with self.assertRaises(RuntimeError):
                restore_prompts(shallow, path, teacher=True)
        torch.testing.assert_close(model.visual.conv1.weight, original)
        for value in model.controller.state_dict().values():
            torch.testing.assert_close(value, torch.full_like(value, .123))

    def test_latency_samples_exclude_warmup_and_keep_inference(self):
        calls = []
        def forward():
            calls.append(torch.is_inference_mode_enabled())
            return torch.ones(1, 8)
        values = measure(forward, torch.device("cpu"), warmup=2, iterations=3)
        self.assertEqual(len(calls), 6)  # one validity pass + warmup + timed calls
        self.assertTrue(all(calls))
        self.assertEqual(len(values["samples_ms"]), 3)
        self.assertEqual(values["output_dim"], 8)

    def test_cpu_cli_exports_complete_reports_without_dataset(self):
        from src import benchmark_encoder_cost as benchmark
        spec = dict(name="test", patch=32, width=8, layers=1, mlp=32, dim=8)
        def load(*args):
            tower = VisionTransformer(224, 32, 8, 1, 2, 8)
            return PromptedStudent(tower, 8, 1, 3, 1, 42).eval().requires_grad_(False), "test"
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark, "load_model", side_effect=load), \
                patch.dict(benchmark.SPECS, {"clip_b32": spec}), \
                patch.object(benchmark, "environment", return_value={}), patch("sys.stdout", new=io.StringIO()):
            main(["--models-root", folder, "--models", "clip_b32", "--devices", "cpu",
                  "--modalities", "sketch", "photo", "--student-depth", "1", "--cpu-threads", "2",
                  "--warmup", "0", "--iterations", "2", "--output-dir", folder])
            output = next(Path(folder).glob("*/summary.json"))
            data = json.loads(output.read_text())
            self.assertEqual(len(data["measurements"]), 2)
            self.assertEqual(data["config"]["cpu_precision"], "fp16")
            self.assertTrue(all(r["precision"] == "fp16" for r in data["measurements"]))
            self.assertEqual(data["models"]["clip_b32"]["profile"]["relative_difference"], 0)
            with output.with_name("table4.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertGreater(float(row["cpu_sketch_median_ms"]), 0)
            self.assertEqual(row["dim"], "8")
            self.assertTrue(output.with_name("measurements.csv").is_file())

    def test_offline_setup_accepts_python_mismatch_and_never_downloads(self):
        path = Path(__file__).resolve().parents[1] / "test/kaggle_encoder_cost_offline.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in {"TASK", "BRANCH", "SOURCE_COMMIT"}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / "offline_bundle"
            (bundle / "KD-SBIR/src").mkdir(parents=True)
            (bundle / "KD-SBIR/src/example.py").write_text("# fixture\n")
            (bundle / "models").mkdir()
            metadata = dict(task=constants["TASK"], branch=constants["BRANCH"],
                            commit=constants["SOURCE_COMMIT"], python=[0, 0], files=[],
                            runtime_requirements=["numpy"])
            (bundle / "bundle_manifest.json").write_text(json.dumps(metadata))
            replacements = {"BUNDLE_PATH": repr(str(bundle)), "WORKING": f"Path({str(root / 'working')!r})"}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    value = replacements.get(node.targets[0].id)
                    if value:
                        node.value = ast.parse(value, mode="eval").body
            ast.fix_missing_locations(tree)
            with patch("subprocess.run") as run, patch("pathlib.Path.symlink_to") as link, \
                    patch("os.chdir"), patch.dict("os.environ"), patch("sys.stdout", new=io.StringIO()):
                exec(compile(tree, str(path), "exec"), {})
            command = run.call_args_list[0].args[0]
            self.assertIn("--no-index", command)
            self.assertNotIn("--python-version", command)
            self.assertEqual(run.call_count, 3)  # local pip + two help calls, no git/network
            link.assert_called_once()
            self.assertTrue((root / "working/KD-SBIR-encoder-cost/src/example.py").is_file())

    def test_wheels_resolve_each_python_independently_without_cuda(self):
        from src.kaggle_cost_wheels import download_runtime_wheels
        calls = []
        def run(command):
            calls.append(command)
            if "--report" in command:
                target = Path(command[command.index("--report") + 1])
                py = command[command.index("--python-version") + 1]
                target.write_text(json.dumps({"install": [
                    {"metadata": {"name": "torch", "version": "2.6.0"}},
                    {"metadata": {"name": "numpy", "version": "2.0.2" if py == "3.12" else "2.2.6"}},
                ]}))
        with tempfile.TemporaryDirectory() as folder, \
                patch("importlib.metadata.distribution", return_value=SimpleNamespace(version="2.6.0", requires=[])):
            root = Path(folder)
            inventories = download_runtime_wheels(["numpy"], ["3.12", "3.13"], root, root / "wheels", run)
            self.assertEqual(inventories["3.12"]["numpy"], "2.0.2")
            self.assertEqual(inventories["3.13"]["numpy"], "2.2.6")
            self.assertNotIn("torch", (root / "requirements-cp313.txt").read_text())
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
