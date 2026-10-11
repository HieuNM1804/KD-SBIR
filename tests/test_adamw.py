"""Exercise real AdamW updates and both training optimizer call sites on CPU."""

import ast
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from io import StringIO

import torch

from src.optimizers import build_prompt_optimizer
from src.data_config import UNSEEN_CLASSES


MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "model.py"


class AdamWTests(unittest.TestCase):
    def test_cli_lr_and_betas_can_be_overridden_independently(self):
        path = MODEL_PATH.with_name('train.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        block = next(node for node in tree.body if isinstance(node, ast.If))
        declarations = []
        for node in block.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'args'
                    for target in node.targets):
                break
            declarations.append(node)
        namespace = dict(argparse=argparse, UNSEEN_CLASSES=UNSEEN_CLASSES)
        exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), 'exec'), namespace)
        parser = namespace['parser']
        args = parser.parse_args(['--root', 'dataset', '--teacher_prompt_lr', '0.0001',
                                  '--lr', '0.0005', '--teacher_betas', '0.8', '0.99',
                                  '--student_betas', '0.85', '0.98'])
        self.assertEqual(args.teacher_prompt_lr, 1e-4)
        self.assertEqual(args.lr, 5e-4)
        self.assertEqual(args.teacher_betas, [0.8, 0.99])
        self.assertEqual(args.student_betas, [0.85, 0.98])
        validation = next(node for node in block.body if isinstance(node, ast.For)
                          and isinstance(node.target, ast.Name) and node.target.id == 'prefix')
        code = compile(ast.Module(body=[validation], type_ignores=[]), str(path), 'exec')
        exec(code, dict(args=args, parser=parser, math=math))
        for name, value in (('student_betas', (1.0, 0.99)), ('teacher_betas', (-0.1, 0.99)),
                            ('teacher_eps', 0.0)):
            with self.subTest(argument=name):
                invalid = SimpleNamespace(**vars(args))
                setattr(invalid, name, value)
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    exec(code, dict(args=invalid, parser=parser, math=math))

    def test_adaptive_update_matches_adamw_and_excludes_frozen_parameters(self):
        prompt = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
        target = torch.nn.Parameter(prompt.detach().clone())
        frozen = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        options = dict(lr=3e-4, weight_decay=1e-3, betas=(0.85, 0.98), eps=1e-7)
        optimizer = build_prompt_optimizer([prompt, frozen], **options)
        reference = torch.optim.AdamW([target], **options)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups[0]['params']), 1)
        for gradient in (torch.tensor([1.0, -2.0]), torch.tensor([-0.5, 0.2])):
            prompt.grad = gradient.clone()
            target.grad = gradient.clone()
            optimizer.step()
            reference.step()
        torch.testing.assert_close(prompt, target)
        self.assertEqual(float(frozen), 5.0)
        self.assertIn('exp_avg', optimizer.state[prompt])
        self.assertIn('exp_avg_sq', optimizer.state[prompt])

    def test_zero_gradient_uses_decoupled_weight_decay(self):
        prompt = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = build_prompt_optimizer([prompt], lr=0.1, weight_decay=0.2,
                                           betas=(0.9, 0.999), eps=1e-8)
        prompt.grad = torch.zeros_like(prompt)
        optimizer.step()
        torch.testing.assert_close(prompt.detach(), torch.tensor([1.96]))

    def test_student_configuration_returns_adamw_and_preserves_scheduler(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding='utf-8'))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == 'configure_optimizers')
        namespace = dict(torch=torch, build_prompt_optimizer=build_prompt_optimizer)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODEL_PATH), 'exec'), namespace)
        model = torch.nn.Linear(3, 2)
        model.weight.requires_grad_(False)
        args = SimpleNamespace(lr=1e-4, weight_decay=0.01,
                               student_betas=(0.8, 0.95), student_eps=1e-6)
        optimizers, schedulers = namespace['configure_optimizers'](SimpleNamespace(model=model, args=args))
        optimizer = optimizers[0]
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        group = optimizer.param_groups[0]
        self.assertEqual((group['lr'], group['weight_decay'], group['betas'], group['eps']),
                         (1e-4, 0.01, (0.8, 0.95), 1e-6))
        self.assertEqual(len(group['params']), 1)
        self.assertIs(group['params'][0], model.bias)
        self.assertIs(schedulers[0].optimizer, optimizer)
        self.assertEqual(schedulers[0].step_size, 5)

    def test_teacher_optimizer_call_uses_teacher_specific_hyperparameters(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'build_prompt_optimizer']
        self.assertEqual(len(calls), 2)
        call = next(node for node in calls if 'cfg.teacher_prompt_lr' in ast.unparse(node))
        prompts = torch.nn.Linear(3, 2)
        cfg = SimpleNamespace(teacher_prompt_lr=3e-4, teacher_weight_decay=0.001,
                              teacher_betas=(0.9, 0.99), teacher_eps=1e-7)
        optimizer = eval(compile(ast.Expression(call), str(MODEL_PATH), 'eval'),
                         dict(build_prompt_optimizer=build_prompt_optimizer, cfg=cfg,
                              self=SimpleNamespace(teacher_prompts=prompts)))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.99))
        self.assertEqual(optimizer.defaults['eps'], 1e-7)
        self.assertEqual(optimizer.defaults['lr'], 3e-4)
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == 'SGD'
                             for node in ast.walk(tree)))

    def test_teacher_cache_records_adamw_settings_but_ignores_student_settings(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'_teacher_training_config', 'default_teacher_cache_path'}]
        namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path,
                         DFN5B_MODEL='ViT-H-14-quickgelu', DFN5B_PRETRAINED='dfn5b',
                         DFN5B_OUTPUT_DIM=1024, TEACHER_CACHE_FORMAT_VERSION=6)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MODEL_PATH), 'exec'), namespace)
        config_function = namespace['_teacher_training_config']
        fields = {node.attr: 1 for node in ast.walk(functions[0])
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id == 'args'}
        fields['teacher_betas'] = (0.9, 0.999)
        args = SimpleNamespace(**fields, dataset='sketchy_1', root='dataset', teacher_cache_dir='cache')
        dataset = SimpleNamespace(max_size=224, all_categories=['cat'],
                                  all_sketches_path=['dataset/sketch/cat/a.png'],
                                  all_photo_paths=['dataset/photo/cat/a.jpg'])
        config = config_function(args)
        self.assertEqual(config['teacher_optimizer'], 'AdamW')
        self.assertEqual(config['teacher_betas'], [0.9, 0.999])
        self.assertNotIn('teacher_momentum', config)
        cache_path = namespace['default_teacher_cache_path']
        baseline = cache_path(args, dataset)
        args.student_betas, args.student_eps, args.lr = (0.7, 0.9), 1e-6, 0.1
        self.assertEqual(cache_path(args, dataset), baseline)
        for name, value in (('teacher_betas', (0.8, 0.99)), ('teacher_eps', 1e-7),
                            ('teacher_prompt_lr', 3e-4), ('teacher_weight_decay', 0.01)):
            old = getattr(args, name)
            setattr(args, name, value)
            self.assertNotEqual(cache_path(args, dataset), baseline)
            setattr(args, name, old)
        legacy = {key: value for key, value in config.items()
                  if key not in {'teacher_optimizer', 'teacher_betas', 'teacher_eps'}}
        legacy['teacher_momentum'] = 0.9
        namespace['_teacher_training_config'] = lambda unused: legacy
        self.assertNotEqual(cache_path(args, dataset), baseline)


if __name__ == '__main__':
    unittest.main()
