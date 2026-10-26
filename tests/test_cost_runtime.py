"""Environment and online/offline workflow tests; no real model downloads."""
import ast
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import cost_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_colab_prefers_content_even_if_old_cell_created_kaggle_folder(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(Path, "is_dir", lambda p: p.as_posix() in {"/content", "/kaggle/working"}):
            self.assertEqual(runtime.detect_environment(), "colab")
            self.assertEqual(runtime.working_directory(), Path("/content"))
            self.assertEqual(runtime.download_mode(), "online")

    def test_kaggle_defaults_offline_and_can_be_explicitly_overridden(self):
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            self.assertEqual(runtime.detect_environment(), "kaggle")
            self.assertEqual(runtime.working_directory(), Path("/kaggle/working"))
            self.assertEqual(runtime.download_mode(), "offline")
            self.assertEqual(runtime.download_mode("online"), "online")

    def test_offline_missing_files_never_trigger_network(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("huggingface_hub.hf_hub_download") as hf, \
                patch.object(runtime, "download_verified") as direct, patch("sys.stdout", new=io.StringIO()):
            with self.assertRaisesRegex(FileNotFoundError, "offline model files"):
                runtime.ensure_models(["dfn5b_h14", "clip_b32", "tinyclip_61m"], folder, "offline")
            hf.assert_not_called()
            direct.assert_not_called()

    def test_offline_complete_bundle_needs_no_hf_metadata_or_network(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("huggingface_hub.hf_hub_download") as hf, \
                patch.object(runtime, "download_verified") as direct, patch("sys.stdout", new=io.StringIO()):
            root = Path(folder)
            for key in runtime.SPECS:
                for name, _, _ in runtime.model_files(key):
                    path = root / runtime.SPECS[key]["directory"] / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            runtime.ensure_models(list(runtime.SPECS), root, "offline")
            hf.assert_not_called()
            direct.assert_not_called()

    def test_online_downloads_pinned_missing_models_then_reuses_without_network(self):
        data = {"dfn5b_h14": b"dfn", "clip_b32": b"clip", "tinyclip_61m": b"tiny"}
        config = b'{"test":true}'
        specs = {key: dict(value, sha256=hashlib.sha256(data[key]).hexdigest())
                 for key, value in runtime.SPECS.items()}
        specs["tinyclip_61m"]["config_blob_id"] = hashlib.sha1(
            f"blob {len(config)}\0".encode() + config).hexdigest()
        def hf(repo_id, revision, filename, local_dir, **kwargs):
            key = next(k for k, v in specs.items() if v.get("repository") == repo_id)
            self.assertEqual(revision, specs[key]["revision"])
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(config if filename == "config.json" else data[key])
        def direct(urls, target, expected):
            self.assertTrue(urls[0].startswith("https://openaipublic.blob.core.windows.net/"))
            self.assertEqual(expected, specs["clip_b32"]["sha256"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data["clip_b32"])
        with tempfile.TemporaryDirectory() as folder, patch.dict(runtime.SPECS, specs), \
                patch("huggingface_hub.hf_hub_download", side_effect=hf) as hf_download, \
                patch.object(runtime, "download_verified", side_effect=direct) as url_download, \
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}), \
                patch("sys.stdout", new=io.StringIO()):
            runtime.ensure_models(list(specs), folder, "online")
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
            self.assertEqual(hf_download.call_count, 3)
            url_download.assert_called_once()
            hf_download.reset_mock()
            url_download.reset_mock()
            runtime.ensure_models(list(specs), folder, "online")
            hf_download.assert_not_called()
            url_download.assert_not_called()

    def test_colab_default_model_root_is_platform_working_cache(self):
        with patch.object(runtime, "working_directory", return_value=Path("/content")), \
                patch.object(Path, "is_dir", return_value=False):
            self.assertEqual(runtime.models_directory(), Path("/content/encoder_cost_models"))

    def test_training_dataset_can_be_explicit_or_environment_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sketch").mkdir()
            (root / "photo").mkdir()
            self.assertEqual(runtime.dataset_directory(root), root.resolve())
            with patch.dict(os.environ, {"KD_SBIR_DATA_ROOT": str(root)}):
                self.assertEqual(runtime.dataset_directory(), root.resolve())
            with self.assertRaisesRegex(FileNotFoundError, "DATA_ROOT"):
                runtime.dataset_directory(root / "missing")

    def test_shared_run_cells_find_current_project_and_do_not_use_kaggle_cd(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("kaggle_encoder_cost_run.ipy", "kaggle_training_cost_run.ipy"):
            source = (root / "test" / name).read_text(encoding="utf-8")
            ast.parse(source)
            with patch("subprocess.run") as run, patch("sys.stdout", new=io.StringIO()):
                exec(compile(source, name, "exec"), {})
            call = run.call_args
            self.assertEqual(call.kwargs["cwd"], root)
            args = call.args[0]
            self.assertEqual(args[args.index("--download-mode") + 1], "auto")
            self.assertNotIn("/kaggle/working/encoder_cost", args)
            self.assertNotIn("models", args)


if __name__ == "__main__":
    unittest.main()
