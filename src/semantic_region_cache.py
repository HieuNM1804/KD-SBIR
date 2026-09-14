"""Resumable crop-semantic cache. Alignment fitting uses seen images only."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.dataset import TeacherFeatureDataset, sample_seed
from src.semantic_region import crop_regions, content_prior, semantic_weights, fit_alignment


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    required = sum(t.numel()*t.element_size() for t in value.values() if torch.is_tensor(t)) + 8*1024**2
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(f'Not enough disk space to save {path}; need {required / 1024**2:.1f} MiB free')
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise


def state_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if 'region_head.' in name or name == 'region_alignment':
            continue
        h.update(name.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def image_fingerprint(paths, root):
    h = hashlib.sha256()
    for p in tqdm(paths, desc='[Region Cache] image hashes', mininterval=5):
        h.update(Path(p).relative_to(root).as_posix().encode())
        h.update(file_hash(p).encode())
    return h.hexdigest()


def valid_file(directory, manifest, name):
    path = directory / name
    return name in manifest['files'] and path.is_file() and file_hash(path) == manifest['files'][name]


def record_file(directory, manifest, name):
    manifest['files'][name] = file_hash(directory / name)
    atomic_json(directory / 'manifest.json', manifest)


def loader(paths, args, batch_size):
    return DataLoader(TeacherFeatureDataset(paths, args.max_size), batch_size=batch_size,
                      num_workers=args.workers, shuffle=False, pin_memory=True,
                      generator=torch.Generator().manual_seed(args.seed+912), persistent_workers=False)


def calibration_indices(paths, args):
    categories = {}
    for i, p in enumerate(paths):
        categories.setdefault(Path(p).parent.name, []).append(i)
    rng = np.random.default_rng(args.seed+910)
    return [int(i) for c in sorted(categories)
            for i in rng.permutation(categories[c])[:args.region_calibration_per_class]]


@torch.no_grad()
def prepare_region_cache(module, dataset, val_sketch, val_photo):
    args = module.args
    directory = Path(args.region_cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(module.model.clip_model.parameters()).device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    module.to(device).eval()
    paths = {'sketch': dataset.all_sketches_path, 'photo': dataset.all_photo_paths}
    metadata = {'format': 1, 'definition': 'teacher_global_crops_bilinear_seen_procrustes_student_dense_local_v_ffn',
                'teacher_sha256': file_hash(args.teacher_cache_path),
                'teacher_metadata': module.model._teacher_cache_metadata(dataset),
                'student_initial_sha256': state_fingerprint(module.model),
                'image_fingerprints': {m: image_fingerprint(p, args.root) for m,p in paths.items()},
                'source_sha256': {n: args.training_source_sha256[n] for n in
                    ('src/model.py','src/dataset.py','src/semantic_region.py','src/semantic_region_cache.py',
                     'src/teacher_prompts.py','clip/model.py')},
                'region_grid': args.region_grid, 'temperature': args.region_temperature,
                'calibration_per_class': args.region_calibration_per_class, 'shard_size': args.region_shard_size,
                'seed': args.seed, 'fit_data': 'seen_categories_only', 'target_dimension': 512,
                'gate_definition': 'cosine_of_teacher_crop_and_full_global_times_sketch_ink_mass_prior',
                'descriptor_target': 'normalized_teacher_global_times_fixed_alignment',
                'test_labels_used_for_region_weights': False}
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest['metadata'] != metadata:
            raise ValueError('Region cache configuration/source/images differ. Use another --region_cache_dir; existing data retained.')
    else:
        manifest = {'metadata': metadata, 'files': {}, 'completed': False}
        atomic_json(manifest_path, manifest)
    width, regions = 512, args.region_grid**2
    # Preflight only the missing output. Source bundles/model weights stay untouched.
    remaining = sum(len(p)*width*(regions+1)*2 for p in paths.values())
    already = sum((directory/n).stat().st_size for n in manifest['files'] if (directory/n).is_file())
    if shutil.disk_usage(directory).free < max(0, remaining-already) + 64*1024**2:
        raise OSError('Insufficient free disk for region cache (about %.2f GiB remaining)' % (max(0,remaining-already)/1024**3))
    references = {}
    for modality, current_paths in paths.items():
        name = 'reference_' + modality + '.pt'
        if not valid_file(directory, manifest, name):
            parts = []
            for images in tqdm(loader(current_paths,args,64), desc='[Region Cache] initial student '+modality, mininterval=5):
                parts.append(module.model.encode_student_image(images.to(device), modality).half().cpu())
            atomic_save(directory/name, {'reference': torch.cat(parts)})
            record_file(directory,manifest,name)
        references[modality] = torch.load(directory/name,weights_only=True)['reference']
        if references[modality].shape != (len(current_paths),width):
            raise ValueError('Reference shape mismatch')
    payload = torch.load(args.teacher_cache_path, map_location='cpu', weights_only=True)
    if payload['metadata'] != metadata['teacher_metadata']:
        raise ValueError('Teacher cache metadata differs')
    full = {'sketch': payload['teacher_sketch_features'], 'photo': payload['teacher_photo_features']}
    if not valid_file(directory,manifest,'alignment.pt'):
        fit_ids = {m:calibration_indices(p,args) for m,p in paths.items()}
        t = torch.cat([full[m][fit_ids[m]] for m in paths])
        s = torch.cat([references[m][fit_ids[m]] for m in paths])
        q = fit_alignment(t,s)
        fit_cosine = F.cosine_similarity(t.float()@q,s.float()).mean().item()
        atomic_save(directory/'alignment.pt',{'matrix':q,'fit_indices':fit_ids,'mean_fit_cosine':fit_cosine})
        record_file(directory,manifest,'alignment.pt')
    alignment = torch.load(directory/'alignment.pt',weights_only=True)
    q = alignment['matrix']
    if q.shape != (1024,width) or not torch.allclose(q.T@q,torch.eye(width),atol=2e-5):
        raise ValueError('Invalid fixed alignment')
    print('[Region Alignment] fit cosine=%.4f; seen-only samples=%d; fixed 1024->512 map' %
          (alignment['mean_fit_cosine'],sum(len(v) for v in alignment['fit_indices'].values())),flush=True)
    module.model.region_alignment.copy_(q.to(device))
    names = {m:[f'{m}_{start:07d}.pt' for start in range(0,len(p),args.region_shard_size)] for m,p in paths.items()}
    need_teacher = any(not valid_file(directory,manifest,n) for ns in names.values() for n in ns)
    eval_needed = args.region_eval_teacher and not valid_file(directory,manifest,'teacher_evaluation.json')
    teacher = controller = None
    if need_teacher or eval_needed:
        import open_clip
        from src.teacher_prompts import build_teacher_prompt_controller
        from src.model import DFN5B_MODEL, DFN5B_PRETRAINED
        print('[Region Teacher] reload DFN5B for semantic crops; original tuned prompts restored',flush=True)
        teacher = open_clip.create_model(DFN5B_MODEL, pretrained=DFN5B_PRETRAINED,
                                        precision='fp16', device=device).eval().requires_grad_(False)
        if not payload.get('teacher_prompt_state_dict'):
            raise ValueError('Teacher cache must contain its tuned prompt state')
        controller = build_teacher_prompt_controller(teacher,args.teacher_n_ctx_visual,
                      args.teacher_prompt_depth,args.teacher_prompt_std,args.teacher_prompt_seed).eval()
        controller.load_state_dict(payload['teacher_prompt_state_dict'],strict=True)
        controller.requires_grad_(False)
    def encode_crops(images, modality):
        crops = crop_regions(images,args.region_grid)
        return torch.stack([F.normalize(controller(crops[:,r].to(dtype=teacher.visual.conv1.weight.dtype),
                          modality).float(),dim=-1) for r in range(regions)], dim=1)
    checked = set()
    for modality, current_paths in paths.items():
        for start, name in zip(range(0,len(current_paths),args.region_shard_size),names[modality]):
            if valid_file(directory,manifest,name):
                print('[Region Cache] reuse',name,flush=True)
                continue
            end = min(start+args.region_shard_size,len(current_paths))
            parts, sems, priors, randoms = [],[],[],[]
            offset = start
            for images in tqdm(loader(current_paths[start:end],args,args.region_teacher_batch_size),
                               desc='[Region Cache] '+name,mininterval=5):
                images = images.to(device)
                n = len(images)
                ft = F.normalize(full[modality][offset:offset+n].to(device).float(),dim=-1)
                if modality not in checked:
                    actual = F.normalize(controller(images.to(dtype=teacher.visual.conv1.weight.dtype),modality).float(),dim=-1)
                    if F.cosine_similarity(ft,actual).min().item() < .999:
                        raise ValueError('Restored teacher full-image features do not match original cache')
                    checked.add(modality)
                tc = encode_crops(images,modality)
                prior = content_prior(images,modality,args.region_grid)
                sem = semantic_weights(ft,tc,prior,args.region_temperature)
                random = torch.stack([torch.from_numpy(np.random.default_rng(sample_seed(args.seed+913,0,i))
                                     .uniform(.1,1,regions)).float() for i in range(offset,offset+n)]).to(device) * prior
                random = random / random.sum(-1,keepdim=True)
                parts.append(F.normalize(tc @ q.to(device),dim=-1).half().cpu())
                sems.append(sem.cpu()); priors.append(prior.cpu()); randoms.append(random.cpu())
                offset += n
            shard = {'crops':torch.cat(parts),'semantic':torch.cat(sems),
                     'prior':torch.cat(priors),'random':torch.cat(randoms)}
            validate_shard(shard,end-start,regions,width)
            atomic_save(directory/name,shard)
            record_file(directory,manifest,name)
            print('[Region Cache] committed',name,flush=True)
    if eval_needed:
        results = teacher_evaluation(module,teacher,controller,q, val_sketch,val_photo,encode_crops)
        atomic_json(directory/'teacher_evaluation.json',results)
        record_file(directory,manifest,'teacher_evaluation.json')
    if teacher is not None:
        del controller, teacher
        torch.cuda.empty_cache()
    dataset.region_targets = {}
    for modality, current_paths in paths.items():
        shards = [torch.load(directory/n,weights_only=True) for n in names[modality]]
        for start,shard in zip(range(0,len(current_paths),args.region_shard_size),shards):
            validate_shard(shard,min(args.region_shard_size,len(current_paths)-start),regions,width)
        dataset.region_targets[modality] = {k:torch.cat([v[k] for v in shards]) for k in shards[0]}
        dataset.region_targets[modality]['reference'] = references[modality]
    manifest['completed'] = True
    atomic_json(manifest_path,manifest)
    args.region_target_metadata = metadata | {'alignment_sha256':manifest['files']['alignment.pt']}
    print('[Region Cache] complete:',directory,flush=True)


def validate_shard(shard,count,regions,width):
    expected = {'crops':(count,regions,width),'semantic':(count,regions),
                'prior':(count,regions),'random':(count,regions)}
    for key,shape in expected.items():
        if key not in shard or tuple(shard[key].shape) != shape or not torch.isfinite(shard[key]).all():
            raise ValueError('Invalid shard '+key)
    for key in ('semantic','prior','random'):
        v = shard[key]
        if (v<0).any() or not torch.allclose(v.sum(-1),torch.ones(count),atol=1e-5):
            raise ValueError('Invalid distribution '+key)
    if (shard['crops'].float().norm(dim=-1)<.99).any():
        raise ValueError('Degenerate crop targets')


@torch.no_grad()
def teacher_evaluation(module,teacher,controller,q,sketch_loader,photo_loader,encode_crops):
    from src.model import _retrieval_metrics
    args = module.args
    device = next(module.parameters()).device
    outputs,labels = {},{}
    for modality, original in [('sketch',sketch_loader),('photo',photo_loader)]:
        data = DataLoader(original.dataset,batch_size=args.region_teacher_batch_size,
                          num_workers=args.workers,shuffle=False,
                          generator=torch.Generator().manual_seed(args.seed+914))
        parts = {k:[] for k in ('teacher_global','teacher_aligned','teacher_region_uniform','teacher_region_semantic','student_initial_native')}
        ys = []
        for images,y in tqdm(data,desc='[Region Oracle] full unseen '+modality,mininterval=5):
            images = images.to(device)
            ft = F.normalize(controller(images.to(dtype=teacher.visual.conv1.weight.dtype),modality).float(),dim=-1)
            tc = encode_crops(images,modality)
            prior = content_prior(images,modality,args.region_grid)
            sem = semantic_weights(ft,tc,prior,args.region_temperature)
            values = {'teacher_global':ft,'teacher_aligned':F.normalize(ft@q.to(device),dim=-1),
                      'teacher_region_uniform':F.normalize(torch.einsum('br,brd->bd',prior,tc)@q.to(device),dim=-1),
                      'teacher_region_semantic':F.normalize(torch.einsum('br,brd->bd',sem,tc)@q.to(device),dim=-1),
                      'student_initial_native':module.model.encode_student_image(images,modality).float()}
            for k,v in values.items(): parts[k].append(v.cpu())
            ys.append(y)
        outputs[modality] = {k:torch.cat(v) for k,v in parts.items()}
        labels[modality] = torch.cat(ys)
    results = []
    for variant in outputs['sketch']:
        ap,p,mk,pk = _retrieval_metrics(outputs['sketch'][variant].to(device).float(),
                    outputs['photo'][variant].to(device).float(),labels['sketch'],labels['photo'],args.dataset)
        row = {'variant':variant,'mAP':ap.item(),'precision':p.item(),'map_k':mk,'p_k':pk,
               'queries':len(labels['sketch']),'gallery':len(labels['photo'])}
        print('[Region Oracle]',json.dumps(row),flush=True)
        results.append(row)
    return {'evaluation':'full_unseen; labels only used for metrics, never region selection',
            'results':results,'alignment_fit_split':'seen_only'}
