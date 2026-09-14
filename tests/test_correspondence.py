"""No-download tests; real small CLIP backbones, raw 1024 vs 512 targets.

These are numerical/protocol tests, not DFN5B/Sketchy performance evidence.
CUDA tests deliberately exercise strict deterministic FP16 autocast backward.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from clip.model import CLIP, convert_weights
from src.train import seed_everything
from src.train_correspondence import parse_args, SOURCE_FILES
from src.correspondence_data import CorrespondenceDataset, BalancedCorrespondenceSampler, paired_photo_id
from src.correspondence_model import CorrespondenceModule, load_correspondence_checkpoint
from src.correspondence_cache import prepare_correspondence_cache
from src.semantic_region_cache import file_hash
from src.region_correspondence import conditional_plan, observed_visibility, positive_retrieval_loss, retrieval_metrics

torch.set_num_threads(4)


def arguments(root, protocol='fg', mode='teacher'):
    args = parse_args(['--root', str(root), '--retrieval_protocol', protocol, '--correspondence_mode', mode,
                       '--workers', '0', '--batch_size', '4', '--classes_per_batch', '2', '--steps_per_epoch', '2',
                       '--teacher_pretrain_batch_size', '4', '--region_teacher_batch_size', '2',
                       '--region_shard_size', '2', '--output_dir', str(root), '--precision', '32-true'])
    args.max_size = 32
    args.fg_photos_per_category = 0
    args.prompt_depth = 2
    args.teacher_prompt_depth = 4
    args.training_source_sha256 = {n: file_hash(n) for n in SOURCE_FILES}
    return args


def images(root):
    rng = np.random.default_rng(8)
    for category in ('seen_a', 'seen_b', 'unseen_a'):
        for modality in ('photo', 'sketch'):
            directory = root / modality / category
            directory.mkdir(parents=True)
            for i in range(4):
                for j in range(2 if modality == 'sketch' else 1):
                    name = f'photo-{i}-{j}.png' if modality == 'sketch' else f'photo-{i}.png'
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(directory / name)


def fixture(root, protocol='fg', mode='teacher'):
    images(root)
    args = arguments(root, protocol, mode)
    with patch('src.correspondence_data.UNSEEN_CLASSES', {'sketchy_1': ['unseen_a']}):
        dataset = CorrespondenceDataset(args)
    seed_everything(42)
    backbone = CLIP(512, 32, 2, 64, 8, 77, 49408, 64, 1, 1)
    module = CorrespondenceModule(args, dataset.all_categories, backbone).eval()
    teacher = CLIP(1024, 32, 4, 128, 4, 77, 49408, 64, 1, 1).eval().requires_grad_(False)
    return module, dataset, teacher


def prepare(module, dataset, teacher):
    with patch('src.correspondence_cache.load_correspondence_teacher', return_value=teacher.to(module.device)):
        prepare_correspondence_cache(module, dataset)


def known_targets(dataset):
    # A known separable teacher graph guarantees active KD in backward tests.
    dataset.region_targets = {}
    for modality, paths in [('photo', dataset.all_photo_paths), ('sketch', dataset.all_sketches_path)]:
        ids = list(range(len(paths))) if modality == 'photo' else dataset.paired_indices
        vectors = F.one_hot(torch.tensor(ids), num_classes=1024).float()
        dataset.region_targets[modality] = {'global': vectors, 'crops': vectors[:, None, :].expand(-1, 4, -1).clone(),
                                              'visibility': torch.full((len(paths), 4), .25)}


class ProtocolTests(unittest.TestCase):
    def test_hyphenated_photo_ids_and_invalid_names(self):
        self.assertEqual(paired_photo_id('photo-with-hyphens-12.png'), 'photo-with-hyphens')
        with self.assertRaises(ValueError):
            paired_photo_id('invalid.png')

    def test_exact_pairs_holdout_and_epoch_sampler(self):
        with TemporaryDirectory() as td:
            module, data, _ = fixture(Path(td))
            held = set(data.val_photo_indices)
            self.assertTrue(all(data.paired_indices[s] not in held for s in data.train_indices))
            self.assertTrue(all(data.paired_indices[s] in held for s in data.val_sketch_indices))
            sampler = BalancedCorrespondenceSampler(data, 4, 2, 42, 2)
            sampler.epoch_source = type('Epoch', (), {'current_epoch': 7})()
            first = list(sampler)
            self.assertEqual(first, list(sampler))
            for batch in first:
                self.assertTrue(all(epoch == 7 for epoch, _ in batch))
                samples = [data[key] for key in batch]
                for item in samples:
                    self.assertEqual(item['photo_index'], item['positive_id'])
                    self.assertNotIn(item['photo_index'], held)
                self.assertEqual(len(set(item['positive_id'] for item in samples)), 4)
            sampler.epoch_source.current_epoch = 8
            self.assertNotEqual(first, list(sampler))

    def test_category_is_not_instance(self):
        with TemporaryDirectory() as td:
            _, data, _ = fixture(Path(td), 'category')
            for s in data.train_indices:
                item = data[(0, s)]
                self.assertEqual(item['positive_id'], item['category'])
                self.assertNotIn(item['photo_index'], data.val_photo_indices)

    def test_fg_missing_pair_fails(self):
        with TemporaryDirectory() as td:
            root = Path(td); images(root)
            (root / 'sketch' / 'seen_a' / 'missing-1.png').write_bytes((root / 'sketch' / 'seen_a' / 'photo-0-0.png').read_bytes())
            with patch('src.correspondence_data.UNSEEN_CLASSES', {'sketchy_1': ['unseen_a']}):
                with self.assertRaisesRegex(ValueError, 'no exact paired'):
                    CorrespondenceDataset(arguments(root))

    def test_multi_positive_duplicate_photos(self):
        vectors = torch.eye(3)[torch.tensor([0, 1, 0])]
        ids = torch.tensor([10, 11, 10])
        loss = positive_retrieval_loss(vectors, vectors, ids, ids, .01)
        self.assertLess(loss.item(), 1e-5)
        bad = torch.tensor([20, 21, 22])
        with self.assertRaises(ValueError):
            positive_retrieval_loss(vectors, vectors, ids, bad, .01)

    def test_metrics_have_exact_instance_semantics(self):
        vectors = torch.eye(3)
        values = retrieval_metrics(vectors, vectors, [1, 2, 3], [1, 2, 3], 'fg')
        self.assertEqual(values['Acc1'], 1.)
        self.assertEqual(values['primary'], 1.)
        with self.assertRaisesRegex(ValueError, 'no ground-truth'):
            retrieval_metrics(vectors, vectors, [1, 2, 9], [1, 2, 3], 'fg')

    def test_known_category_gallery_is_explicit(self):
        s, p = torch.tensor([[0., 1.]]), torch.tensor([[.7, .7], [1., 0.], [0., 1.]])
        sid, pid = torch.tensor([[10, 0]]), torch.tensor([[10, 0], [11, 0], [20, 1]])
        all_gallery = retrieval_metrics(s, p, sid, pid, 'fg', fg_gallery='all')
        category = retrieval_metrics(s, p, sid, pid, 'fg', fg_gallery='category')
        self.assertEqual(all_gallery['Acc1'], 0.)
        self.assertEqual(category['Acc1'], 1.)
        self.assertEqual(category['gallery_candidates_max'], 2)
        self.assertEqual(category['gallery_scope'], 'known_category')

    def test_no_match_bin_and_blank_sketch(self):
        plan = conditional_plan(torch.full((2, 4, 4), -.9), torch.full((2, 4), .25), .1, .2)
        self.assertTrue((plan[..., -1] > .99).all())
        self.assertTrue(torch.allclose(plan.sum(-1), torch.ones(2, 4)))
        rgb = torch.ones(2, 3, 32, 32)
        mean = torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        blank = (rgb - mean) / std
        self.assertEqual(observed_visibility(blank, 'sketch', 2).sum().item(), 0.)


class CacheTests(unittest.TestCase):
    def test_corrupt_shard_is_regenerated(self):
        with TemporaryDirectory() as td:
            module, data, teacher = fixture(Path(td)); prepare(module, data, teacher)
            shard = Path(module.args.correspondence_cache_dir) / 'photo_0000000.pt'
            original = file_hash(shard)
            with shard.open('ab') as stream:
                stream.write(b'corrupted')
            self.assertNotEqual(original, file_hash(shard))
            prepare(module, data, teacher)
            self.assertEqual(original, file_hash(shard))

    def test_raw_dimensions_resume_and_no_alignment(self):
        with TemporaryDirectory() as td:
            module, data, teacher = fixture(Path(td)); prepare(module, data, teacher)
            self.assertEqual(data.region_targets['photo']['crops'].shape[-1], 1024)
            self.assertEqual(module.model.clip_model.visual.proj.shape[-1], 512)
            directory = Path(module.args.correspondence_cache_dir)
            self.assertFalse((directory / 'alignment.pt').exists())
            hashes = {p.name: file_hash(p) for p in directory.glob('*.pt')}
            with patch('src.correspondence_cache.load_correspondence_teacher', side_effect=AssertionError('Teacher reloaded')):
                prepare_correspondence_cache(module, data)
            self.assertEqual(hashes, {p.name: file_hash(p) for p in directory.glob('*.pt')})
            module.args.region_grid = 3
            with self.assertRaisesRegex(ValueError, 'differ'):
                prepare_correspondence_cache(module, data)
            self.assertEqual(hashes, {p.name: file_hash(p) for p in directory.glob('*.pt')})

    def test_teacher_tuning_uses_seen_only(self):
        with TemporaryDirectory() as td:
            module, data, teacher = fixture(Path(td))
            from open_clip.transformer import VisionTransformer
            teacher = torch.nn.Module()
            teacher.visual = VisionTransformer(32, 4, 128, 4, 2, 2, output_dim=1024)
            teacher.encode_image = lambda x: teacher.visual(x)
            teacher.eval().requires_grad_(False)
            module.args.teacher_pretrain_epochs = 1
            module.args.teacher_steps_per_epoch = 1
            with patch.object(data, 'unseen_validation', side_effect=AssertionError('Unseen used in teacher selection')):
                prepare(module, data, teacher)
            saved = torch.load(Path(module.args.correspondence_cache_dir) / 'teacher.pt', weights_only=True)
            self.assertEqual(saved['selection']['selection'], 'held_out_seen_only')
            self.assertEqual(len(saved['selection']['history']), 2)

    def test_tuned_teacher_cache_is_independent_of_student_seed(self):
        with TemporaryDirectory() as td:
            module, data, _ = fixture(Path(td))
            from open_clip.transformer import VisionTransformer
            teacher = torch.nn.Module()
            teacher.visual = VisionTransformer(32, 4, 128, 4, 2, 2, output_dim=1024)
            teacher.encode_image = lambda x: teacher.visual(x)
            teacher.eval().requires_grad_(False)
            module.args.teacher_pretrain_epochs = 1
            module.args.teacher_steps_per_epoch = 1
            prepare(module, data, teacher)
            module.args.seed = 987
            with patch('src.correspondence_cache.load_correspondence_teacher',
                       side_effect=AssertionError('Student seed invalidated teacher cache')):
                prepare_correspondence_cache(module, data)

    def test_completed_tuned_cache_rejects_corrupt_prompt_state(self):
        with TemporaryDirectory() as td:
            module, data, _ = fixture(Path(td))
            from open_clip.transformer import VisionTransformer
            teacher = torch.nn.Module()
            teacher.visual = VisionTransformer(32, 4, 128, 4, 2, 2, output_dim=1024)
            teacher.encode_image = lambda x: teacher.visual(x)
            teacher.eval().requires_grad_(False)
            module.args.teacher_pretrain_epochs = 1
            module.args.teacher_steps_per_epoch = 1
            prepare(module, data, teacher)
            prompt_state = Path(module.args.correspondence_cache_dir) / 'teacher.pt'
            with prompt_state.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'teacher prompt state'):
                prepare_correspondence_cache(module, data)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_teacher_prompt_checkpointing(self):
        with TemporaryDirectory() as td:
            module, data, _ = fixture(Path(td)); module.cuda()
            from open_clip.transformer import VisionTransformer
            teacher = torch.nn.Module()
            teacher.visual = VisionTransformer(32, 4, 128, 4, 2, 2, output_dim=1024)
            teacher.encode_image = lambda x: teacher.visual(x)
            teacher.cuda().half().eval().requires_grad_(False)
            module.args.teacher_pretrain_epochs = 1
            module.args.teacher_steps_per_epoch = 1
            prepare(module, data, teacher)
            saved = torch.load(Path(module.args.correspondence_cache_dir) / 'teacher.pt', weights_only=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['prompts'].values()))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in module.parameters()))


def backward_test(device):
    with TemporaryDirectory() as td:
        module, data, _ = fixture(Path(td))
        module.args.region_train_prompts = True
        module.model.photo_visual_prompt.requires_grad_(True)
        module.model.sketch_visual_prompt.requires_grad_(True)
        known_targets(data)
        if device == 'cuda':
            convert_weights(module.model.clip_model)
            module.model.dtype = module.model.clip_model.dtype
        module.to(device).train()
        module.log = lambda *a, **kw: None
        sampler = BalancedCorrespondenceSampler(data, 4, 2, 42, 1)
        batch = next(iter(DataLoader(data, batch_sampler=sampler, num_workers=0)))
        batch = {k: v.to(device) for k, v in batch.items()}
        teacher_before = batch['photo_crops'].clone()
        keys = tuple(module.state_dict())
        from src.region_correspondence import correspondence_losses
        for mode in ('gt', 'global', 'uniform', 'teacher', 'shuffled'):
            module.args.correspondence_mode = mode
            module.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
                outputs = {m: module.model.encode_region_image(batch[m], m) for m in ('sketch', 'photo')}
                loss, values = correspondence_losses(outputs['sketch'], outputs['photo'], batch, module.args)
            assert loss.dtype == torch.float32 and torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in module.model.clip_model.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            assert module.model.region_head.fusion.up.weight.grad is not None
            assert module.model.region_head.fusion.up.weight.grad.abs().sum() > 0
            if mode == 'teacher':
                assert values['teacher_acceptance'] == 1
                assert values['correspondence'] > 0
                assert module.model.region_head.region_adapter.up.weight.grad.abs().sum() > 0
        assert tuple(module.state_dict()) == keys
        assert not any('teacher' in k or 'alignment' in k for k in keys)
        assert torch.equal(teacher_before, batch['photo_crops'])
        before = module.model.region_head.fusion.up.weight.detach().clone()
        torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.01).step()
        assert not torch.equal(before, module.model.region_head.fusion.up.weight)
        module.eval()
        with torch.inference_mode():
            first = module.model.extract_feature(batch['sketch'], 'sketch')
            module.args.lambda_correspondence = module.args.lambda_rank = 0
            second = module.model.extract_feature(batch['sketch'], 'sketch')
        assert torch.equal(first, second)


class BackwardTests(unittest.TestCase):
    def test_cpu_deterministic_bfloat16(self):
        backward_test('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_deterministic_fp16(self):
        backward_test('cuda')


class LightningTests(unittest.TestCase):
    def test_train_save_teacher_free_inference_and_resume(self):
        with TemporaryDirectory() as td:
            root = Path(td); module, data, teacher = fixture(root)
            prepare(module, data, teacher)
            sampler = BalancedCorrespondenceSampler(data, 4, 2, 42, 2)
            sampler.epoch_source = module
            train = DataLoader(data, batch_sampler=sampler, num_workers=0)
            valid = [DataLoader(data.seen_validation(m), batch_size=4) for m in ('sketch', 'photo')]
            frozen = {k: p.detach().clone() for k, p in module.model.clip_model.named_parameters()}
            trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=CSVLogger(str(root / 'logs')),
                              enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                              num_sanity_val_steps=0, log_every_n_steps=1, deterministic=True)
            trainer.validate(module, dataloaders=valid, verbose=False)
            trainer.fit(module, train, valid)
            for k, p in module.model.clip_model.named_parameters():
                self.assertTrue(torch.equal(p, frozen[k]))
            checkpoint = root / 'final.ckpt'; trainer.save_checkpoint(checkpoint)
            with (patch('src.correspondence_cache.load_correspondence_teacher', side_effect=AssertionError('Inference teacher')),
                  patch('src.correspondence_model._load_clip_model', side_effect=AssertionError('Inference download'))):
                loaded = load_correspondence_checkpoint(checkpoint)
            x = next(iter(valid[0]))[0]
            with torch.inference_mode():
                self.assertTrue(torch.allclose(module.model.extract_feature(x, 'sketch'),
                                               loaded.model.extract_feature(x, 'sketch'), atol=1e-6))
            for p in loaded.model.region_head.parameters():
                p.requires_grad_(True)
            resumed = Trainer(accelerator='cpu', devices=1, max_epochs=2, logger=CSVLogger(str(root / 'resume')),
                              enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                              num_sanity_val_steps=0, deterministic=True)
            sampler.epoch_source = loaded
            resumed.fit(loaded, train, valid, ckpt_path=str(checkpoint))
            self.assertEqual(resumed.global_step, 4)
            self.assertEqual(resumed.optimizers[0].state_dict()['state'][0]['step'].item(), 4)


if __name__ == '__main__':
    unittest.main()
