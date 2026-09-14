"""Raw 1024-D teacher globals/crops, atomic shards and seen-only selection."""
from pathlib import Path
import json
import shutil

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import TeacherFeatureDataset
from src.semantic_region import crop_regions
from src.semantic_region_cache import atomic_json, atomic_save, file_hash, image_fingerprint, record_file, valid_file
from src.correspondence_data import BalancedCorrespondenceSampler
from src.region_correspondence import observed_visibility, positive_retrieval_loss, retrieval_metrics


def load_correspondence_teacher(device):
    import open_clip
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED
    return open_clip.create_model(DFN5B_MODEL, pretrained=DFN5B_PRETRAINED,
                                 precision='fp16' if device.type == 'cuda' else 'fp32', device=device).eval().requires_grad_(False)


def view_loader(view, args):
    return DataLoader(view, batch_size=args.region_teacher_batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate_encoder(encode, dataset, args, device, split='seen'):
    values, labels = {}, {}
    for modality in ('sketch', 'photo'):
        view = dataset.seen_validation(modality) if split == 'seen' else dataset.unseen_validation(modality)
        if not len(view):
            return None
        features, ys = [], []
        for images, target in view_loader(view, args):
            features.append(F.normalize(encode(images.to(device), modality).float(), dim=-1).cpu())
            ys.append(target)
        values[modality], labels[modality] = torch.cat(features), torch.cat(ys)
    return retrieval_metrics(values['sketch'].to(device), values['photo'].to(device),
                             labels['sketch'], labels['photo'], args.retrieval_protocol,
                             args.precision_k, args.metric_chunk_size, args.fg_gallery)


def tune_teacher(teacher, controller, dataset, args, device):
    """Exact-instance/category GT; no unseen data in training or selection."""
    dtype = teacher.visual.conv1.weight.dtype
    def encode(images, modality):
        return controller(images.to(dtype=dtype), modality)
    initial = evaluate_encoder(encode, dataset, args, device)
    best = initial['primary']
    best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
    best_epoch, history = -1, [{'epoch': -1, 'seen': initial}]
    optimizer = torch.optim.SGD(controller.parameters(), lr=args.teacher_prompt_lr,
                                momentum=args.teacher_momentum, weight_decay=args.teacher_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.teacher_scheduler_step_size,
                                               gamma=args.teacher_scheduler_gamma)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type == 'cuda')
    sampler = BalancedCorrespondenceSampler(dataset, args.teacher_pretrain_batch_size,
                                           args.classes_per_batch, args.teacher_prompt_seed + 999, args.teacher_steps_per_epoch)
    train = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers)
    controller.requires_grad_(True)
    for epoch in range(args.teacher_pretrain_epochs):
        for batch in tqdm(train, desc=f'Teacher correspondence GT {epoch+1}/{args.teacher_pretrain_epochs}', mininterval=5):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                s = encode(batch['sketch'].to(device), 'sketch')
                p = encode(batch['photo'].to(device), 'photo')
            with torch.autocast(device_type=device.type, enabled=False):
                labels = batch['positive_id'].to(device)
                loss = positive_retrieval_loss(s, p, labels, labels, args.retrieval_temperature)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite teacher GT loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # GradScaler legitimately skips overflowing FP16 steps and lowers its scale.
            torch.nn.utils.clip_grad_norm_(controller.parameters(), args.gradient_clip_val, error_if_nonfinite=False)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        result = evaluate_encoder(encode, dataset, args, device)
        history.append({'epoch': epoch, 'seen': result})
        print('[Teacher Seen Selection]', json.dumps(history[-1]), flush=True)
        if result['primary'] > best:
            best, best_epoch = result['primary'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
    controller.load_state_dict(best_state, strict=True)
    controller.requires_grad_(False).eval()
    return {'best_epoch': best_epoch, 'best_seen_primary': best, 'history': history,
            'selection': 'held_out_seen_only', 'protocol': args.retrieval_protocol}


def validate_targets(shard, count, regions, modality):
    shapes = {'global': (count, 1024), 'crops': (count, regions, 1024), 'visibility': (count, regions)}
    for key, shape in shapes.items():
        if key not in shard or tuple(shard[key].shape) != shape or not torch.isfinite(shard[key]).all():
            raise ValueError('Invalid correspondence cache tensor: ' + key)
    for key in ('global', 'crops'):
        if (shard[key].float().norm(dim=-1) < .99).any():
            raise ValueError('Degenerate teacher embeddings: ' + key)
    vis = shard['visibility']
    sums = vis.sum(-1)
    valid_sum = torch.isclose(sums, torch.ones_like(sums), atol=1e-5)
    if modality == 'sketch':
        valid_sum = valid_sum | (sums == 0)
    if (vis < 0).any() or not valid_sum.all():
        raise ValueError('Invalid visibility distribution')


@torch.no_grad()
def prepare_correspondence_cache(module, dataset):
    args = module.args
    directory = Path(args.correspondence_cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    device = module.device
    paths = {'sketch': dataset.all_sketches_path, 'photo': dataset.all_photo_paths}
    teacher_config = {k: getattr(args, k) for k in
                      ('teacher_pretrain_epochs', 'teacher_n_ctx_visual', 'teacher_prompt_depth', 'teacher_prompt_std',
                       'teacher_prompt_seed', 'teacher_prompt_lr', 'teacher_momentum', 'teacher_weight_decay',
                       'teacher_pretrain_batch_size', 'teacher_steps_per_epoch', 'teacher_scheduler_step_size',
                       'teacher_scheduler_gamma', 'teacher_gradient_checkpointing', 'classes_per_batch',
                       'retrieval_temperature', 'gradient_clip_val', 'fg_gallery')}
    metadata = {'format': 1, 'definition': 'raw_dfn5b_global_and_resized_grid_crop_embeddings',
                'teacher': 'ViT-H-14-quickgelu/dfn5b', 'teacher_config': teacher_config,
                'target_dim': 1024, 'alignment': None, 'region_grid': args.region_grid,
                'teacher_precision': 'fp16' if device.type == 'cuda' else 'fp32',
                'image_size': args.max_size, 'shard_size': args.region_shard_size,
                'protocol': args.retrieval_protocol, 'split_digest': dataset.split_digest,
                'selection': 'held_out_seen_only', 'visibility': 'ink_mass_not_semantic_visibility; blank_sketch_zero',
                'image_hashes': {m: image_fingerprint(p, args.root) for m, p in paths.items()},
                'source_hashes': {n: file_hash(n) for n in ('src/dataset.py', 'src/teacher_prompts.py',
                                  'src/semantic_region.py', 'src/correspondence_data.py', 'src/correspondence_cache.py',
                                  'src/region_correspondence.py')}}
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['metadata'] != metadata:
            raise ValueError('Correspondence cache source/config/images/split differ; use another cache directory. Existing data retained.')
    else:
        manifest = {'metadata': metadata, 'files': {}, 'completed': False}
        atomic_json(manifest_path, manifest)
    regions = args.region_grid ** 2
    if args.teacher_pretrain_epochs and manifest['completed'] and not valid_file(directory, manifest, 'teacher.pt'):
        raise ValueError('Completed cache has a missing/corrupted teacher prompt state; use another cache directory. Existing data retained.')
    names = {m: [f'{m}_{i:07d}.pt' for i in range(0, len(p), args.region_shard_size)] for m, p in paths.items()}
    missing = any(not valid_file(directory, manifest, n) for ns in names.values() for n in ns)
    need_audit = args.audit_teacher and not valid_file(directory, manifest, 'teacher_audit.json')
    needed_bytes = sum(min(args.region_shard_size, len(paths[m]) - start) * ((regions + 1) * 1024 * 2 + regions * 4)
                       for m in paths for start, name in zip(range(0, len(paths[m]), args.region_shard_size), names[m])
                       if not valid_file(directory, manifest, name))
    if shutil.disk_usage(directory).free < needed_bytes + 64 * 1024**2:
        raise OSError(f'Insufficient free space for correspondence cache: {needed_bytes / 1024**3:.2f} GiB required')
    teacher = controller = None
    if missing or need_audit:
        teacher = load_correspondence_teacher(device)
        dtype = teacher.visual.conv1.weight.dtype
        if args.teacher_pretrain_epochs:
            from src.teacher_prompts import build_teacher_prompt_controller
            controller = build_teacher_prompt_controller(teacher, args.teacher_n_ctx_visual,
                          args.teacher_prompt_depth, args.teacher_prompt_std, args.teacher_prompt_seed)
            controller.gradient_checkpointing = args.teacher_gradient_checkpointing
            if valid_file(directory, manifest, 'teacher.pt'):
                state = torch.load(directory / 'teacher.pt', map_location='cpu', weights_only=True)
                controller.load_state_dict(state['prompts'], strict=True)
                controller.requires_grad_(False).eval()
            else:
                with torch.enable_grad():
                    selection = tune_teacher(teacher, controller, dataset, args, device)
                atomic_save(directory / 'teacher.pt', {'prompts': {k: v.cpu() for k, v in controller.state_dict().items()},
                                                     'selection': selection})
                record_file(directory, manifest, 'teacher.pt')
        def encode(images, modality):
            images = images.to(dtype=dtype)
            return teacher.encode_image(images) if controller is None else controller(images, modality)
        for modality, current in paths.items():
            for start, name in zip(range(0, len(current), args.region_shard_size), names[modality]):
                if valid_file(directory, manifest, name):
                    print('[Correspondence Cache] reuse', name, flush=True)
                    continue
                end = min(start + args.region_shard_size, len(current))
                data = DataLoader(TeacherFeatureDataset(current[start:end], args.max_size),
                                  batch_size=args.region_teacher_batch_size, num_workers=args.workers, shuffle=False)
                parts = {'global': [], 'crops': [], 'visibility': []}
                for images in tqdm(data, desc='[Correspondence Cache] ' + name, mininterval=5):
                    images = images.to(device)
                    full = F.normalize(encode(images, modality).float(), dim=-1)
                    crops = crop_regions(images, args.region_grid)
                    local = torch.stack([F.normalize(encode(crops[:, r], modality).float(), dim=-1)
                                         for r in range(regions)], 1)
                    parts['global'].append(full.half().cpu())
                    parts['crops'].append(local.half().cpu())
                    parts['visibility'].append(observed_visibility(images, modality, args.region_grid).cpu())
                shard = {k: torch.cat(v) for k, v in parts.items()}
                validate_targets(shard, end - start, regions, modality)
                atomic_save(directory / name, shard)
                record_file(directory, manifest, name)
        if need_audit:
            # Unseen metrics are diagnostic only, after seen-only checkpoint selection.
            audit = {'seen': evaluate_encoder(encode, dataset, args, device),
                     'unseen': evaluate_encoder(encode, dataset, args, device, 'unseen'),
                     'selection': 'held_out_seen_only', 'unseen_used_for_selection': False,
                     'teacher_crop_correspondence_quality': 'not_assumed; inspect acceptance/evidence and matched controls'}
            atomic_json(directory / 'teacher_audit.json', audit)
            record_file(directory, manifest, 'teacher_audit.json')
            print('[Correspondence Teacher Audit]', json.dumps(audit), flush=True)
        del teacher, controller
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    dataset.region_targets = {}
    for modality, current in paths.items():
        shards = [torch.load(directory / name, map_location='cpu', weights_only=True) for name in names[modality]]
        for start, shard in zip(range(0, len(current), args.region_shard_size), shards):
            validate_targets(shard, min(args.region_shard_size, len(current) - start), regions, modality)
        dataset.region_targets[modality] = {k: torch.cat([s[k] for s in shards]) for k in shards[0]}
    manifest['completed'] = True
    atomic_json(manifest_path, manifest)
    args.teacher_target_metadata = metadata | {'file_hashes': dict(manifest['files'])}
    print('[Correspondence Cache] complete; raw 1024-D targets, no alignment, no teacher in student optimizer', flush=True)
