"""Train unequal-backbone region correspondence KD without test-set selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data_config import UNSEEN_CLASSES
from src.train import seed_everything, seed_worker
from src.correspondence_data import CorrespondenceDataset, BalancedCorrespondenceSampler
from src.correspondence_model import CorrespondenceModule, load_correspondence_checkpoint
from src.correspondence_cache import prepare_correspondence_cache, evaluate_encoder
from src.semantic_region_cache import atomic_json, file_hash

SOURCE_FILES = ('clip/model.py', 'src/model.py', 'src/dataset.py', 'src/data_config.py',
                'src/teacher_prompts.py', 'src/semantic_region.py', 'src/semantic_region_cache.py',
                'src/correspondence_data.py', 'src/region_correspondence.py', 'src/correspondence_model.py',
                'src/correspondence_cache.py', 'src/train_correspondence.py')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--dataset', choices=sorted(UNSEEN_CLASSES), default='sketchy_1')
    p.add_argument('--retrieval_protocol', choices=('category', 'fg'), default='category')
    p.add_argument('--fg_gallery', choices=('category', 'all'), default='category',
                   help='FG category = known-category gallery like existing FG pipeline; all = entire gallery')
    p.add_argument('--fg_photos_per_category', type=int, default=100, help='FG basic-set validation; 0 for an explicit custom dataset')
    p.add_argument('--correspondence_mode', choices=('gt', 'global', 'uniform', 'teacher', 'shuffled'), default='teacher')
    p.add_argument('--backbone', default='ViT-B/32')
    p.add_argument('--max_size', type=int, default=224)
    p.add_argument('--n_ctx_visual', type=int, default=3)
    p.add_argument('--prompt_depth', type=int, default=12)
    p.add_argument('--region_grid', type=int, default=2)
    p.add_argument('--region_bottleneck', type=int, default=64)
    p.add_argument('--region_beta', type=float, default=.5)
    p.add_argument('--region_train_prompts', action='store_true')
    p.add_argument('--region_head_lr', type=float, default=1e-3)
    p.add_argument('--lr', type=float, default=1e-4, help='Optional trainable visual prompt LR')
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--classes_per_batch', type=int, default=4)
    p.add_argument('--steps_per_epoch', type=int, default=0, help='0 = ceil(number of fit sketches / batch_size)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--test_batch_size', type=int, default=256)
    p.add_argument('--region_teacher_batch_size', type=int, default=8)
    p.add_argument('--region_shard_size', type=int, default=2048)
    p.add_argument('--correspondence_cache_dir', default='')
    p.add_argument('--prepare_only', action='store_true')
    p.add_argument('--audit_teacher', action='store_true')
    p.add_argument('--teacher_pretrain_epochs', type=int, default=0, help='0 = original frozen DFN5B; optional prompts select on seen only')
    p.add_argument('--teacher_pretrain_batch_size', type=int, default=64)
    p.add_argument('--teacher_steps_per_epoch', type=int, default=0)
    p.add_argument('--teacher_n_ctx_visual', type=int, default=3)
    p.add_argument('--teacher_prompt_depth', type=int, default=12)
    p.add_argument('--teacher_prompt_std', type=float, default=.02)
    p.add_argument('--teacher_prompt_lr', type=float, default=2e-5)
    p.add_argument('--teacher_prompt_seed', type=int, default=42, help='Independent teacher seed; permits reuse across student seeds')
    p.add_argument('--no_teacher_gradient_checkpointing', dest='teacher_gradient_checkpointing', action='store_false')
    p.set_defaults(teacher_gradient_checkpointing=True)
    p.add_argument('--teacher_momentum', type=float, default=.9)
    p.add_argument('--teacher_weight_decay', type=float, default=1e-3)
    p.add_argument('--teacher_scheduler_step_size', type=int, default=5)
    p.add_argument('--teacher_scheduler_gamma', type=float, default=.1)
    p.add_argument('--seen_val_fraction', type=float, default=.1)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lambda_retrieval', type=float, default=1.)
    p.add_argument('--lambda_rank', type=float, default=.5)
    p.add_argument('--lambda_correspondence', type=float, default=.25)
    p.add_argument('--retrieval_temperature', type=float, default=.07)
    p.add_argument('--rank_temperature', type=float, default=.1)
    p.add_argument('--match_temperature', type=float, default=.1)
    p.add_argument('--no_match_score', type=float, default=.2, help='Heuristic no-match cosine score, not calibrated confidence')
    p.add_argument('--teacher_local_weight', type=float, default=.5)
    p.add_argument('--teacher_min_margin', type=float, default=0.)
    p.add_argument('--hard_negatives', type=int, default=2)
    p.add_argument('--gradient_clip_val', type=float, default=1.)
    p.add_argument('--scheduler_step_size', type=int, default=5)
    p.add_argument('--scheduler_gamma', type=float, default=.1)
    p.add_argument('--precision_k', type=int, default=100)
    p.add_argument('--metric_chunk_size', type=int, default=64)
    p.add_argument('--precision', choices=('32-true', '16-mixed', 'bf16-mixed'), default='16-mixed')
    p.add_argument('--accelerator', choices=('auto', 'gpu', 'cpu'), default='auto')
    p.add_argument('--output_dir', default='.')
    p.add_argument('--exp_name', default='sketch_region_correspondence')
    p.add_argument('--ckpt_path', default='')
    return p


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backbone != 'ViT-B/32' or args.max_size != 224:
        parser.error('First implementation supports ViT-B/32 at 224; teacher remains DFN5B ViT-H/14')
    for key in ('batch_size', 'classes_per_batch', 'epochs', 'test_batch_size', 'region_teacher_batch_size',
                'region_shard_size', 'region_bottleneck', 'hard_negatives', 'teacher_pretrain_batch_size',
                'scheduler_step_size', 'teacher_scheduler_step_size', 'precision_k', 'metric_chunk_size'):
        if getattr(args, key) < 1:
            parser.error(key + ' must be positive')
    if args.batch_size < 2 or args.batch_size % args.classes_per_batch or args.teacher_pretrain_batch_size % args.classes_per_batch:
        parser.error('Student/teacher batch sizes must be divisible by classes_per_batch; batch_size >= 2')
    if args.retrieval_protocol == 'category' and args.classes_per_batch < 2:
        parser.error('Category mode requires >= 2 classes per batch for actual negatives')
    if not 1 <= args.region_grid <= 3 or args.prompt_depth < 1 or args.n_ctx_visual < 0:
        parser.error('region_grid in 1..3, prompt_depth >= 1, n_ctx_visual >= 0')
    if not 0 < args.seen_val_fraction < .5 or not 0 < args.region_beta <= 1:
        parser.error('seen_val_fraction in (0,.5), region_beta in (0,1]')
    if not 0 <= args.teacher_local_weight <= 1 or not -1 <= args.no_match_score <= 1:
        parser.error('teacher_local_weight in [0,1], no_match_score in [-1,1]')
    for key in ('retrieval_temperature', 'rank_temperature', 'match_temperature', 'region_head_lr', 'lr',
                'teacher_prompt_lr', 'teacher_prompt_std', 'gradient_clip_val', 'scheduler_gamma', 'teacher_scheduler_gamma'):
        if getattr(args, key) <= 0:
            parser.error(key + ' must be positive')
    for key in ('workers', 'steps_per_epoch', 'teacher_pretrain_epochs', 'teacher_steps_per_epoch', 'fg_photos_per_category',
                'lambda_rank', 'lambda_correspondence', 'weight_decay', 'teacher_weight_decay', 'teacher_momentum'):
        if getattr(args, key) < 0:
            parser.error(key + ' must be nonnegative')
    if args.lambda_retrieval <= 0:
        parser.error('GT retrieval loss must remain active')
    if not 0 <= args.teacher_min_margin <= 2:
        parser.error('teacher_min_margin must be in [0,2]')
    if args.teacher_pretrain_epochs and (args.teacher_n_ctx_visual < 1 or args.teacher_prompt_depth == 0 or args.teacher_prompt_depth < -1):
        parser.error('Tuned teacher requires n_ctx >= 1 and depth -1 or positive')
    if args.ckpt_path and not Path(args.ckpt_path).is_file():
        parser.error('Resume checkpoint does not exist')
    if Path(args.exp_name).name != args.exp_name or args.exp_name in ('.', '..'):
        parser.error('exp_name must be one folder name')
    if not args.correspondence_cache_dir:
        args.correspondence_cache_dir = str(Path(args.output_dir) / 'correspondence_cache' /
                                          f'{args.dataset}_{args.retrieval_protocol}_g{args.region_grid}')
    # Compatibility values for the frozen visual-only CustomCLIP implementation.
    args.region_mode = 'semantic'
    args.lambda_domain = args.lambda_modality = 0.
    args.kd_temperature = .07
    args.photo_text_kd_temperature = args.sketch_text_kd_temperature = .1
    args.teacher_cache_path = ''
    args.rebuild_teacher_cache = False
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    source_root = Path(__file__).resolve().parent.parent
    args.training_source_sha256 = {n: file_hash(source_root / n) for n in SOURCE_FILES}
    dataset = CorrespondenceDataset(args)
    if len(dataset.all_categories) < args.classes_per_batch:
        raise ValueError('classes_per_batch exceeds available seen categories')
    print('[Correspondence Dataset]', json.dumps({'protocol': args.retrieval_protocol,
          'fit_sketches': len(dataset.train_indices), 'seen_val_sketches': len(dataset.val_sketch_indices),
          'seen_val_photos': len(dataset.val_photo_indices), 'split_digest': dataset.split_digest,
          'sampling': 'class-balanced; FG distinct photo IDs when available; not every sketch exactly once'}), flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() and args.accelerator != 'cpu' else 'cpu')
    module = CorrespondenceModule(args, dataset.all_categories).to(device).eval()
    if args.correspondence_mode != 'gt' or args.prepare_only or args.audit_teacher:
        prepare_correspondence_cache(module, dataset)
    if args.prepare_only:
        initial = evaluate_encoder(lambda x, m: module.model.extract_feature(x, m), dataset, args, device)
        print('[Preparation Complete] no student optimizer steps; initial seen=', json.dumps(initial), flush=True)
        return
    run_dir = Path(args.output_dir) / 'saved_models' / args.exp_name
    if not args.ckpt_path and run_dir.is_dir() and any(run_dir.glob('*.ckpt')):
        raise ValueError('Existing checkpoints retained; choose another exp_name or explicitly resume with ckpt_path')
    if args.ckpt_path:
        saved = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        config = saved.get('experiment_config', {})
        if config.get('retrieval_head') != 'region_correspondence':
            raise ValueError('Cannot resume another method checkpoint')
        recorded = config['args']
        ignore = {'epochs', 'exp_name', 'ckpt_path', 'output_dir', 'accelerator', 'workers', 'audit_teacher', 'prepare_only'}
        for key, value in vars(args).items():
            if key not in ignore and recorded.get(key) != value:
                raise ValueError('Resume configuration/targets/source differs: ' + key)
        del saved
    sampler = BalancedCorrespondenceSampler(dataset, args.batch_size, args.classes_per_batch, args.seed, args.steps_per_epoch)
    sampler.epoch_source = module
    kwargs = {'num_workers': args.workers, 'pin_memory': device.type == 'cuda', 'worker_init_fn': seed_worker,
              'generator': torch.Generator().manual_seed(args.seed)}
    train = DataLoader(dataset, batch_sampler=sampler, **kwargs)
    valid = [DataLoader(dataset.seen_validation(m), batch_size=args.test_batch_size, shuffle=False, **kwargs)
             for m in ('sketch', 'photo')]
    checkpoint = ModelCheckpoint(dirpath=run_dir, filename='best-{epoch:02d}-{seen_val_primary:.4f}',
                                 monitor='seen_val_primary', mode='max', save_top_k=1, save_last=True)
    logs = Path(args.output_dir) / 'tb_logs'
    loggers = [TensorBoardLogger(str(logs), name=args.exp_name), CSVLogger(str(logs), name=args.exp_name + '_csv')]
    precision = args.precision if device.type == 'cuda' else '32-true'
    trainer = Trainer(accelerator='gpu' if device.type == 'cuda' else 'cpu', devices=1, max_epochs=args.epochs,
                      precision=precision, deterministic=True, benchmark=False, logger=loggers,
                      callbacks=[checkpoint], gradient_clip_val=args.gradient_clip_val,
                      num_sanity_val_steps=0, log_every_n_steps=20)
    if not args.ckpt_path:
        trainer.validate(module, dataloaders=valid, verbose=False)
    trainer.fit(module, train, valid, ckpt_path=args.ckpt_path or None)
    trainer.save_checkpoint(run_dir / 'final.ckpt')
    selected = checkpoint.best_model_path or str(run_dir / 'final.ckpt')
    deployed = load_correspondence_checkpoint(selected, device)
    unseen = evaluate_encoder(lambda x, m: deployed.model.extract_feature(x, m), dataset, args, device, 'unseen')
    native = evaluate_encoder(lambda x, m: deployed.model.encode_student_image(x, m), dataset, args, device, 'unseen')
    report = {'completed': True, 'args': vars(args), 'best_checkpoint': selected, 'final_checkpoint': str(run_dir / 'final.ckpt'),
              'best_seen_primary': float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
              'unseen_deployed': unseen, 'unseen_native': native, 'selection': 'held_out_seen_only',
              'student_dim': deployed.model.clip_model.visual.proj.shape[1], 'teacher_dim': 1024,
              'teacher_required_at_inference': False}
    atomic_json(run_dir / 'run.json', report)
    print('[Correspondence Finished]', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
