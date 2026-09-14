"""Inspect Kaggle cells without executing their filesystem/network actions."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


def constants(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


class SetupTests(unittest.TestCase):
    def test_setup_source_and_weights_pins_match(self):
        online, offline = constants('test/kaggle_online.py'), constants('test/kaggle_offline.py')
        self.assertEqual(online['BRANCH'], offline['EXPECTED_BRANCH'])
        self.assertEqual(online['COMMIT'], offline['EXPECTED_COMMIT'])
        self.assertEqual(online['TASK'], offline['EXPECTED_TASK'])
        self.assertEqual(online['ENTRYPOINT'], offline['EXPECTED_ENTRYPOINT'])
        self.assertIsNotNone(re.fullmatch(r'[0-9a-f]{40}', online['COMMIT']))
        for key in ('DFN_REPO', 'DFN_REVISION', 'DFN_FILENAME', 'DFN_SHA256', 'STUDENT_FILENAME', 'STUDENT_SHA256'):
            self.assertEqual(online[key], offline[key])
        self.assertEqual(len(online['DFN_SHA256']), 64)
        self.assertEqual(len(online['STUDENT_SHA256']), 64)

    def test_dependency_extras_and_stack_exclusion(self):
        tree = ast.parse(Path('test/kaggle_online.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'installed_dependency_closure')
        distributions = {
            'root': SimpleNamespace(version='1.0', requires=['torch>=2', 'torchvision', 'triton', 'nvidia-cuda-runtime-cu12',
                    'http-dep; extra == "http"', 'dev-dep; extra == "dev"', 'torchmetrics>=1']),
            'http-dep': SimpleNamespace(version='2.0', requires=['leaf>=1']),
            'leaf': SimpleNamespace(version='1.0', requires=[]),
            'torchmetrics': SimpleNamespace(version='1.8.2', requires=[])}
        scope = {'metadata': SimpleNamespace(distribution=lambda name: distributions[name])}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'closure_test', 'exec'), scope)
        actual = scope['installed_dependency_closure'](['root[http]>=1'])
        self.assertEqual(actual, {'root': '1.0', 'http-dep': '2.0', 'leaf': '1.0', 'torchmetrics': '1.8.2'})
        with self.assertRaises(RuntimeError):
            scope['installed_dependency_closure'](['root>=2'])


if __name__ == '__main__':
    unittest.main()
