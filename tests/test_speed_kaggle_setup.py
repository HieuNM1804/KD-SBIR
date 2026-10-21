"""Offline setup behavior under differing builder/runtime Python versions."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]/"test/kaggle_speed_offline.py"


class SetupTests(unittest.TestCase):
    def execute(self, succeeds):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            bundle=root/"offline_bundle"
            bundle.mkdir()
            (bundle/"KD-SBIR").mkdir()
            (bundle/"wheels").mkdir()
            metadata=dict(task="clip_tinyclip_speed", python=[9,9], commit="test-commit",
                          files=[], runtime_requirements=["example-package==1.0"])
            manifest=bundle/"bundle_manifest.json"
            manifest.write_text(json.dumps(metadata))
            text=SOURCE.read_text(encoding="utf-8").replace("MANIFEST = None",f"MANIFEST = Path({str(manifest)!r})")
            text=text.replace('Path("/kaggle/working/KD-SBIR-speed")',f'Path({str(root/"restored")!r})')
            calls=[]
            def run(command, **kwargs):
                calls.append(command)
                if not succeeds:
                    raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0,"pip stdout","")
            with patch("subprocess.run",side_effect=run), patch("subprocess.check_output",return_value="test-commit"), patch.dict(os.environ), redirect_stdout(io.StringIO()) as output:
                if succeeds:
                    exec(compile(text,str(SOURCE),"exec"),{})
                    self.assertTrue((root/"restored").exists())
                    self.assertEqual(os.environ["SPEED_MODELS_ROOT"],str(bundle/"models"))
                else:
                    with self.assertRaises(subprocess.CalledProcessError):
                        exec(compile(text,str(SOURCE),"exec"),{})
                    self.assertFalse((root/"restored").exists())
            self.assertIn("Bundle Python: 9.9",output.getvalue())
            self.assertEqual(len(calls),1)
            self.assertNotIn("--dry-run",calls[0])
            self.assertIn("--no-index",calls[0])
            self.assertNotIn("--ignore-installed",calls[0])

    def test_different_python_installs_directly_without_precheck(self):
        self.execute(True)

    def test_pip_failure_propagates_without_custom_block(self):
        self.execute(False)

    def test_online_packages_both_python_targets_and_reuses_weights(self):
        text=(SOURCE.parent/"kaggle_speed_online.py").read_text(encoding="utf-8")
        tree=ast.parse(text)
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("TARGET_PYTHONS","TARGET_PLATFORMS")}
        self.assertEqual(values["TARGET_PYTHONS"],("3.12","3.13"))
        self.assertIn("manylinux2014_x86_64",values["TARGET_PLATFORMS"])
        self.assertIn('"--no-deps"',text)
        self.assertIn('"--only-binary=:all:"',text)
        self.assertIn('"--python-version", python_version',text)
        self.assertIn('if not (MODELS/"ViT-B-32.pt").is_file():',text)
        self.assertNotIn('rglob("bundle_manifest.json")',SOURCE.read_text(encoding="utf-8"))


if __name__=="__main__":
    unittest.main()
