"""Regression checks for building wheels across Kaggle Python ABIs."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import zipfile


def load_helper(filename):
    source = Path(__file__).resolve().parents[1] / "test" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "download_runtime_wheels")
    namespace = dict(Path=Path, sys=sys, json=json)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


class KaggleDependencyWheelsTests(unittest.TestCase):
    def test_resolves_numpy_separately_and_never_bundles_provided_stack(self):
        online = next((Path(__file__).resolve().parents[1] / "test").glob("kaggle_*_online.py"))
        helper = load_helper(online.name)
        commands = []
        temporary_stack = []

        def installed(name):
            # The builder's native numpy pin must never be queried or propagated.
            self.assertIn(name, {"torch", "torchvision"})
            return SimpleNamespace(version="2.8.0" if name == "torch" else "0.23.0",
                                   requires=["numpy>=1.20", "nvidia-cublas-cu12==12.0"])

        def run(command):
            commands.append(command)
            python = command[command.index("--python-version") + 1]
            if "--report" in command:
                self.assertIn("--dry-run", command)
                self.assertIn("--ignore-installed", command)
                stubs = [Path(value) for value in command if value.endswith(".whl")]
                self.assertEqual(len(stubs), 2)
                for stub in stubs:
                    temporary_stack.append(stub)
                    with zipfile.ZipFile(stub) as archive:
                        metadata = archive.read(next(n for n in archive.namelist() if n.endswith("/METADATA"))).decode()
                        self.assertIn("Requires-Dist: numpy>=1.20", metadata)
                        self.assertNotIn("nvidia-cublas", metadata)
                report = Path(command[command.index("--report") + 1])
                report.write_text(json.dumps({"install": [
                    {"metadata": {"name": "numpy", "version": "2.0.2" if python == "3.12" else "2.1.3"}},
                    {"metadata": {"name": "torch", "version": "2.8.0"}},
                    {"metadata": {"name": "nvidia-cublas-cu12", "version": "12.0"}},
                ]}), encoding="utf-8")
            else:
                self.assertIn("download", command)
                self.assertIn("--no-deps", command)
                requirements = Path(command[command.index("-r") + 1]).read_text(encoding="utf-8")
                self.assertEqual(requirements, "numpy==" + ("2.0.2" if python == "3.12" else "2.1.3") + "\n")
                self.assertNotIn("torch", requirements)
                self.assertNotIn("nvidia", requirements)

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            wheels = bundle / "wheels"
            wheels.mkdir()
            with patch("importlib.metadata.distribution", side_effect=installed):
                inventories = helper(["numpy"], ("3.12", "3.13"), bundle, wheels, run)
            self.assertEqual(inventories["3.12"]["numpy"], "2.0.2")
            self.assertEqual(inventories["3.13"]["numpy"], "2.1.3")
            self.assertEqual((bundle / "requirements.txt").read_text(encoding="utf-8"), "numpy\n")
            self.assertEqual(list(wheels.iterdir()), [])
        self.assertTrue(all(not path.exists() for path in temporary_stack))
        self.assertEqual(len(commands), 4)


if __name__ == "__main__":
    unittest.main()
