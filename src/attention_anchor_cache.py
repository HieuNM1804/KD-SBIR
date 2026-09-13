"""Seen-only vocabulary and compact teacher descriptor cache for anchor KD."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.dataset import TeacherFeatureDataset
from src.attention_anchor import AnchorCapture, anchor_distribution, fit_vocabulary, hellinger_descriptor

BASELINE_COMMIT = 'b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def data_hash(dataset, root):
    digest = hashlib.sha256()
    for path in tqdm(dataset.all_sketches_path + dataset.all_photo_paths, desc='[Anchor Cache] image hashes'):
        digest.update(Path(path).relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def balanced_paths(dataset, per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for modality, paths in [('sketch', dataset.all_sketches_path), ('photo', dataset.all_photo_paths)]:
        groups = {}
        for path in paths:
            groups.setdefault(Path(path).parent.name, []).append(path)
        for category in sorted(groups):
            members = groups[category]
            ids = torch.randperm(len(members), generator=generator)[:per_class].tolist()
            selected.extend((modality, members[i]) for i in ids)
    return selected


def validate_cache(payload, metadata, ns, np_):
    if payload.get('metadata') != metadata:
        raise ValueError('Anchor cache provenance/config differs. Use a new --anchor_cache_path.')
    count = metadata['anchor_count']
    for modality, size in [('sketch', ns), ('photo', np_)]:
        p = payload.get(modality)
        if not isinstance(p, torch.Tensor) or p.shape != (size, count) or p.dtype != torch.float16:
            raise ValueError(f'Invalid {modality} anchor probabilities')
        for part in p.split(2048):
            if not torch.isfinite(part).all() or (part < 0).any() or not torch.allclose(
                    part.float().sum(-1), torch.ones(len(part)), atol=.002, rtol=0):
                raise ValueError(f'Invalid {modality} anchor probability values')
    anchors, center = payload.get('anchors'), payload.get('center')
    if not isinstance(anchors, torch.Tensor) or anchors.shape != (count, metadata['teacher_width']):
        raise ValueError('Invalid teacher anchors')
    if not isinstance(center, torch.Tensor) or center.shape != (metadata['teacher_width'],):
        raise ValueError('Invalid teacher centering vector')
    if not torch.isfinite(anchors).all() or not torch.isfinite(center).all() or (anchors.norm(dim=-1) < 1e-8).any():
        raise ValueError('Non-finite/zero teacher vocabulary')


def load_teacher(payload):
    import open_clip
    from src.teacher_prompts import TeacherPromptController
    m = payload['metadata']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = open_clip.create_model(m['teacher_model'], pretrained=m['teacher_pretrained'],
                                    precision='fp16' if device == 'cuda' else 'fp32', device=device)
    teacher.eval().requires_grad_(False)
    controller = TeacherPromptController(teacher.visual, m['teacher_n_ctx_visual'], m['teacher_prompt_depth'],
                                         m['teacher_prompt_std'], m['teacher_prompt_seed'])
    controller.load_state_dict(payload['teacher_prompt_state_dict'], strict=True)
    controller.eval().requires_grad_(False)
    return teacher, controller


def batches(paths, args):
    return DataLoader(TeacherFeatureDataset(paths, args.max_size), batch_size=args.anchor_teacher_batch_size,
                      shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(args.seed+610))


def evidence(teacher, controller, images, modality):
    parameter = teacher.visual.conv1.weight
    with torch.no_grad(), AnchorCapture(teacher.visual) as capture:
        global_features = controller(images.to(device=parameter.device, dtype=parameter.dtype), modality)
    if len(capture.values) != 1:
        raise RuntimeError('Expected one teacher attention capture')
    return *capture.values[0], global_features


def save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = sum(v.numel()*v.element_size() for v in payload.values() if isinstance(v, torch.Tensor))
    if shutil.disk_usage(path.parent).free < size+64*1024**2:
        raise OSError(f'Insufficient disk for compact anchor cache: need ~{size/1024**2:.1f} MiB plus overhead')
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, tmp)
        if path.exists():
            raise FileExistsError(f'Refusing to replace existing anchor cache: {path}')
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


@torch.no_grad()
def prepare_anchor_cache(args, dataset, expected_teacher_metadata):
    teacher_path = Path(args.teacher_cache_path)
    payload = torch.load(teacher_path, map_location='cpu', weights_only=True)
    if payload['metadata'] != expected_teacher_metadata or not payload.get('teacher_prompt_state_dict'):
        raise ValueError('Need the matching tuned teacher cache, including its saved prompt weights')
    source_path = Path(__file__).with_name('attention_anchor.py')
    metadata = dict(version=1, definition='last_projected_V_mean_CLS_patch_attention_visual_vocabulary_v1',
                    baseline_commit=BASELINE_COMMIT, teacher_sha256=file_hash(teacher_path),
                    teacher_metadata=payload['metadata'], image_sha256=data_hash(dataset, args.root),
                    implementation_sha256=file_hash(source_path), cache_builder_sha256=file_hash(__file__),
                    teacher_controller_sha256=file_hash(Path(__file__).with_name('teacher_prompts.py')),
                    anchor_count=args.anchor_count, temperature=args.anchor_temperature, pooling=args.anchor_pooling,
                    fit_images_per_class=args.anchor_fit_images_per_class, fit_tokens_per_image=args.anchor_fit_tokens_per_image,
                    fit_iterations=args.anchor_fit_iterations, seed=args.seed, teacher_width=1280,
                    fit_split='seen_training_only', torch=str(torch.__version__))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(args.anchor_cache_path) if args.anchor_cache_path else teacher_path.parent/f'{args.dataset}_anchors_{key}.pt'
    args.anchor_cache_path = str(path)
    ns, np_ = len(dataset.all_sketches_path), len(dataset.all_photo_paths)
    print('[Anchor Cache]', path, flush=True)
    if path.is_file():
        result = torch.load(path, map_location='cpu', weights_only=True)
        validate_cache(result, metadata, ns, np_)
    else:
        samples = balanced_paths(dataset, args.anchor_fit_images_per_class, args.seed+611)
        teacher, controller = load_teacher(payload)
        pool = []
        generator = torch.Generator().manual_seed(args.seed+612)
        for modality in ('sketch', 'photo'):
            paths = [p for mod,p in samples if mod == modality]
            for images in tqdm(batches(paths, args), desc=f'[Anchor Fit] {modality}'):
                tokens, attention, _ = evidence(teacher, controller, images, modality)
                # Uniformly sample patch positions for vocabulary fitting. This
                # fixes the same vocabulary for attention vs uniform pooling.
                for token in tokens.cpu():
                    ids = torch.randperm(len(token), generator=generator)[:args.anchor_fit_tokens_per_image]
                    pool.append(token[ids])
        print('[Anchor Fit] fitting shared vocabulary on CPU', flush=True)
        anchors, center, fit = fit_vocabulary(torch.cat(pool), args.anchor_count, args.anchor_fit_iterations, args.seed+613)
        print('[Anchor Fit]', fit, flush=True)
        del pool
        device = teacher.visual.conv1.weight.device
        c, mean = anchors.to(device), center.to(device)
        result = dict(metadata=metadata, anchors=anchors, center=center, fit=fit, fit_paths=samples)
        for modality, paths in [('sketch', dataset.all_sketches_path), ('photo', dataset.all_photo_paths)]:
            probs = torch.empty(len(paths), args.anchor_count, dtype=torch.float16)
            offset = 0
            for images in tqdm(batches(paths, args), desc=f'[Anchor Cache] {modality}'):
                tokens, attention, _ = evidence(teacher, controller, images, modality)
                p = anchor_distribution(tokens, attention, c, mean, args.anchor_temperature, args.anchor_pooling)
                probs[offset:offset+len(p)] = p.cpu().half()
                offset += len(p)
            result[modality] = probs
        validate_cache(result, metadata, ns, np_)
        save_atomic(path, result)
        del teacher, controller
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    dataset.set_anchor_features(result['sketch'], result['photo'])
    args.anchor_target_metadata = metadata
    print(f'[Anchor Cache] ready; {args.anchor_count}-D probabilities, {path.stat().st_size/1024**2:.1f} MiB', flush=True)
    return result


@torch.no_grad()
def evaluate_teacher(args, cache, sketch_loader, photo_loader, report_path):
    """Full validation, fixed vocabulary; evaluation never fits/tunes anchors."""
    from src.model import _retrieval_metrics
    payload = torch.load(args.teacher_cache_path, map_location='cpu', weights_only=True)
    if file_hash(args.teacher_cache_path) != cache['metadata']['teacher_sha256']:
        raise ValueError('Teacher changed after anchor-cache creation')
    teacher, controller = load_teacher(payload)
    device = teacher.visual.conv1.weight.device
    anchors, center = cache['anchors'].to(device), cache['center'].to(device)
    outputs, labels = {}, {}
    for modality, original in [('sketch', sketch_loader), ('photo', photo_loader)]:
        gathered = {name: [] for name in ('global', 'attention', 'uniform')}
        # Use the small teacher batch size, not the student's validation batch.
        loader = DataLoader(original.dataset, batch_size=args.anchor_teacher_batch_size, shuffle=False,
                            num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed+615))
        targets = []
        for images, category in tqdm(loader, desc=f'[Anchor Teacher Eval] {modality}'):
            tokens, a, features = evidence(teacher, controller, images, modality)
            gathered['global'].append(F.normalize(features.float(), dim=-1).cpu())
            for pooling in ('attention', 'uniform'):
                p = anchor_distribution(tokens, a, anchors, center, args.anchor_temperature, pooling)
                gathered[pooling].append(hellinger_descriptor(p).cpu())
            targets.append(category.cpu())
        outputs[modality] = {name: torch.cat(values) for name,values in gathered.items()}
        labels[modality] = torch.cat(targets)
    del teacher, controller
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    rows = []
    for name in ('global', 'attention', 'uniform'):
        ap, precision, map_k, p_k = _retrieval_metrics(outputs['sketch'][name], outputs['photo'][name],
                                                     labels['sketch'], labels['photo'], args.dataset)
        row = dict(descriptor=name, mAP=float(ap), precision=float(precision), map_k=map_k, p_k=p_k)
        rows.append(row)
        print('[Anchor Teacher Eval]', row, flush=True)
    report = dict(baseline_commit=BASELINE_COMMIT, cache_metadata=cache['metadata'], fit=cache['fit'],
                  queries=len(labels['sketch']), gallery=len(labels['photo']), results=rows,
                  notes=['Full validation; vocabulary fitted only on seen training images.',
                         'Global teacher is a reference, not the main student baseline.',
                         'Uniform and attention descriptors use the same fixed teacher vocabulary.',
                         'These metrics do not demonstrate student performance or novelty.'])
    report_path = Path(report_path); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
