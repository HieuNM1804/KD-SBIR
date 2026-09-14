"""Compact teacher counterfactual features; no local token cache or student head."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.dataset import TeacherFeatureDataset, sample_seed
from src.attention_output_cache import file_sha256
from src.attention_output_kd import PatchContributionCapture
from src.counterfactual_retrieval_kd import (
    CACHE_FORMAT_VERSION, attention_proposals, random_mass_matched_proposals,
    patch_ink_mass, erase_ink_box, counterfactual_effect,
)


def add_arguments(parser):
    parser.add_argument('--lambda_avcrd', type=float, default=0.)
    parser.add_argument('--avcrd_selection', choices=['verified','random','attention_first','random_first'], default='verified')
    parser.add_argument('--avcrd_objective', choices=['field','shuffled_effect'], default='field')
    parser.add_argument('--avcrd_clean_weight', type=float, default=1.)
    parser.add_argument('--avcrd_effect_weight', type=float, default=1.)
    parser.add_argument('--avcrd_magnitude_weight', type=float, default=.25)
    parser.add_argument('--avcrd_window', type=int, default=4,
                        help='Square ink-erasure window in teacher patches; 4 on a 16x16 grid = 56px.')
    parser.add_argument('--avcrd_proposals', type=int, default=3)
    parser.add_argument('--avcrd_photo_bank', type=int, default=512,
                        help='Fixed seen photo reference count for teacher region verification only.')
    parser.add_argument('--avcrd_teacher_batch_size', type=int, default=8)
    parser.add_argument('--avcrd_cache_path', default='')
    parser.add_argument('--avcrd_prepare_only', action='store_true')
    parser.add_argument('--avcrd_audit_per_class', type=int, default=0,
                        help='0 builds full training targets. >0 is an audit only; no student fitting.')
    parser.add_argument('--avcrd_ink_threshold', type=float, default=.08)
    parser.add_argument('--avcrd_ink_softness', type=float, default=.12)
    parser.add_argument('--avcrd_diagnostic_batch_size', type=int, default=32)
    parser.add_argument('--avcrd_diagnostic_examples', type=int, default=8)


def validate_arguments(parser,args):
    import math
    weights=(args.lambda_avcrd,args.avcrd_clean_weight,args.avcrd_effect_weight,args.avcrd_magnitude_weight)
    if any(not math.isfinite(x) or x<0 for x in weights):
        parser.error('AVCRD weights must be finite and nonnegative')
    if args.lambda_avcrd<=0 and not args.avcrd_prepare_only:
        return
    if args.avcrd_clean_weight+args.avcrd_effect_weight<=0:
        parser.error('At least one AVCRD field weight must be positive')
    if min(args.avcrd_window,args.avcrd_proposals,args.avcrd_teacher_batch_size,args.avcrd_photo_bank,args.avcrd_diagnostic_batch_size)<1:
        parser.error('AVCRD window, proposal, bank, and batch sizes must be positive')
    if args.batch_size<2 or args.avcrd_diagnostic_batch_size<2:
        parser.error('AVCRD retrieval fields require at least two photos/queries')
    if args.avcrd_audit_per_class<0 or args.avcrd_diagnostic_examples<0:
        parser.error('AVCRD audit/example counts must be nonnegative')
    if args.avcrd_audit_per_class and not args.avcrd_prepare_only:
        parser.error('Partial AVCRD caches are audit only; use --avcrd_prepare_only')
    if args.teacher_pretrain_epochs<1:
        parser.error('AVCRD needs the matching prompt-tuned teacher; keep teacher_pretrain_epochs >=1')
    if args.lambda_av>0 or args.lambda_global_feature>0:
        parser.error('Run AVCRD and old projector KD as separate controls')
    if not 0<=args.avcrd_ink_threshold<1 or not 0<args.avcrd_ink_softness<=1:
        parser.error('Ink threshold must be in [0,1), softness in (0,1]')


def _selected_indices(dataset,count,seed):
    if count==0:
        return list(range(len(dataset)))
    generator=torch.Generator().manual_seed(seed)
    selected=[]
    for category in dataset.all_categories:
        ids=[i for i,p in enumerate(dataset.all_sketches_path) if Path(p).parent.name==category]
        order=torch.randperm(len(ids),generator=generator)[:count].tolist()
        selected.extend(ids[i] for i in order)
    return sorted(selected)


def _content_fingerprint(paths,root):
    digest=hashlib.sha256()
    for path in tqdm(paths,desc='[AVCRD Cache] sketch content',mininterval=5):
        digest.update(os.path.relpath(path,root).replace('\\','/').encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def validate_payload(payload,metadata,count,width):
    if payload.get('metadata')!=metadata:
        raise ValueError('AVCRD cache metadata differs. Choose a NEW --avcrd_cache_path; existing targets are never overwritten.')
    shapes={'clean':(count,width),'masked':(count,2,width),'boxes':(count,2,4),
            'first_masked':(count,2,width),'first_boxes':(count,2,4),
            'candidate_rms':(count,2,metadata['proposals']),
            'candidate_mass':(count,2,metadata['proposals']),
            'selected':(count,2),'attention':(count,metadata['teacher_patch_count']),
            'saliency':(count,metadata['teacher_patch_count'])}
    for name,shape in shapes.items():
        value=payload.get(name)
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise ValueError('Invalid AVCRD tensor: '+name)
    if any(payload[k].dtype!=torch.float16 for k in ('masked','clean','first_masked')):
        raise ValueError('Counterfactual feature targets must be float16')
    if any((payload[k].float().norm(dim=-1)<1e-8).any() for k in ('masked','first_masked','clean')):
        raise ValueError('Zero AVCRD features')
    boxes=torch.cat([payload['boxes'],payload['first_boxes']],dim=1); size=metadata['max_size']
    if ((boxes[...,:2]<0).any() or (boxes[...,2:]>size).any()
            or (boxes[...,2:]<=boxes[...,:2]).any()):
        raise ValueError('Invalid AVCRD pixel boxes')
    if (payload['selected']<0).any() or (payload['selected']>=metadata['proposals']).any():
        raise ValueError('Invalid selected proposal indices')


def prepare_cache(args,dataset,report_dir):
    preparation_started=time.perf_counter()
    import open_clip
    import torchvision
    import PIL
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION
    from src.teacher_prompts import TeacherPromptController
    path_teacher=Path(args.teacher_cache_path)
    if not path_teacher.is_file():
        raise FileNotFoundError('Prepare the main global teacher cache before AVCRD')
    teacher_payload=torch.load(path_teacher,map_location='cpu',weights_only=True)
    teacher_meta=teacher_payload['metadata']
    if (teacher_meta.get('format_version')!=TEACHER_CACHE_FORMAT_VERSION
            or teacher_meta.get('dataset')!=args.dataset
            or teacher_meta.get('max_size')!=dataset.max_size
            or teacher_meta.get('teacher_model')!=DFN5B_MODEL
            or teacher_meta.get('teacher_pretrained')!=DFN5B_PRETRAINED
            or not teacher_payload.get('teacher_prompt_state_dict')):
        raise ValueError('AVCRD needs the same tuned DFN5B teacher as main')
    ids=_selected_indices(dataset,args.avcrd_audit_per_class,args.seed+8100)
    paths=[dataset.all_sketches_path[i] for i in ids]
    photo_features=F.normalize(teacher_payload['teacher_photo_features'].float(),dim=-1)
    generator=torch.Generator().manual_seed(args.seed+8101)
    bank_ids=torch.randperm(len(photo_features),generator=generator)[:args.avcrd_photo_bank]
    bank=photo_features[bank_ids]
    if len(bank)<2:
        raise ValueError('Teacher verification needs at least two seen photo references')
    # DFN5B uses 14px patches; reject incompatible input instead of approximating.
    if args.max_size%14:
        raise ValueError('DFN5B AVCRD input size must be divisible by 14')
    grid=args.max_size//14
    if args.avcrd_window>=grid//2:
        raise ValueError('Use an erasure window smaller than half the teacher patch grid')
    metadata={'version':CACHE_FORMAT_VERSION,'definition':'final_CLS_per_patch_AVWO_norm_sqrt_ink;verified_centered_global_score_delta;random_equal_budget',
              'teacher_sha256':file_sha256(path_teacher),'teacher_metadata':teacher_meta,
              'image_sha256':_content_fingerprint(paths+dataset.all_photo_paths,args.root),'indices':ids,
              'photo_bank_indices':bank_ids.tolist(),'seed':args.seed,'max_size':args.max_size,
              'teacher_patch_count':grid*grid,'window':args.avcrd_window,'proposals':args.avcrd_proposals,
              'ink_threshold':args.avcrd_ink_threshold,'ink_softness':args.avcrd_ink_softness,
              'view_definition':'original_main_resize;soft_ink_whitening;no_crop;no_prompt_masking',
              'teacher_batch_size':args.avcrd_teacher_batch_size,
              'torch':str(torch.__version__),'open_clip':getattr(open_clip,'__version__','unknown'),
              'torchvision':str(torchvision.__version__),'pillow':str(PIL.__version__),
              'view_source_sha256':{name:file_sha256(Path(__file__).resolve().parent.parent/name)
                    for name in ('src/dataset.py','src/teacher_prompts.py','src/attention_output_kd.py',
                                 'src/counterfactual_retrieval_kd.py','src/counterfactual_cache.py')}}
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:16]
    path=Path(args.avcrd_cache_path) if args.avcrd_cache_path else path_teacher.parent/f'{args.dataset}_avcrd_{key}.pt'
    args.avcrd_cache_path=str(path);args.avcrd_target_metadata=metadata
    count=len(ids);width=photo_features.shape[1]
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        validate_payload(payload,metadata,count,width)
        print('[AVCRD Cache] reused; DFN5B counterfactual encoding skipped:',path,flush=True)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        estimate=count*(5*width*2+2*grid*grid*2+16*args.avcrd_proposals+48)
        free=shutil.disk_usage(path.parent).free
        if free<estimate+2*1024**3:
            raise OSError(f'AVCRD cache needs about {estimate/1024**2:.1f} MiB plus 2 GiB reserve; free={free/1024**3:.2f} GiB')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher=open_clip.create_model(DFN5B_MODEL,pretrained=DFN5B_PRETRAINED,
                                      precision='fp16' if device.type=='cuda' else 'fp32',device=device).eval().requires_grad_(False)
        if teacher.visual.positional_embedding.shape[0]-1!=grid*grid:
            raise ValueError('Unexpected teacher patch geometry')
        controller=TeacherPromptController(teacher.visual,teacher_meta['teacher_n_ctx_visual'],
                    teacher_meta['teacher_prompt_depth'],teacher_meta['teacher_prompt_std'],teacher_meta['teacher_prompt_seed'])
        controller.load_state_dict(teacher_payload['teacher_prompt_state_dict'],strict=True)
        controller.eval().requires_grad_(False)
        dtype=teacher.visual.conv1.weight.dtype
        payload={'metadata':metadata,'clean':torch.empty(count,width,dtype=torch.float16),
                 'masked':torch.empty(count,2,width,dtype=torch.float16),'boxes':torch.empty(count,2,4,dtype=torch.int16),
                 'first_masked':torch.empty(count,2,width,dtype=torch.float16),'first_boxes':torch.empty(count,2,4,dtype=torch.int16),
                 'candidate_rms':torch.empty(count,2,args.avcrd_proposals),'candidate_mass':torch.empty(count,2,args.avcrd_proposals),
                 'selected':torch.empty(count,2,dtype=torch.int16),'attention':torch.empty(count,grid*grid,dtype=torch.float16),
                 'saliency':torch.empty(count,grid*grid,dtype=torch.float16),'clean_cache_cosine':torch.empty(count)}
        loader=DataLoader(TeacherFeatureDataset(paths,dataset.max_size),batch_size=args.avcrd_teacher_batch_size,
                          shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda',
                          generator=torch.Generator().manual_seed(args.seed+8102))
        bank=bank.to(device)
        offset=0
        with torch.no_grad():
            for images in tqdm(loader,desc='[AVCRD Cache] verified sketch views',mininterval=5):
                with PatchContributionCapture(teacher.visual) as capture:
                    clean=controller(images.to(device,dtype=dtype),'sketch')
                if len(capture.values)!=1:
                    raise RuntimeError('Expected one teacher attention contribution capture')
                # AVWO norm is used as a proposal map, never as a retrieval descriptor.
                saliency=capture.values[0].norm(dim=-1).cpu()
                attention=capture.attention[0].cpu()
                ink=patch_ink_mass(images,grid,args.avcrd_ink_threshold,args.avcrd_ink_softness)
                for local,image in enumerate(images):
                    row=offset+local
                    proposals=attention_proposals(saliency[local].view(grid,grid),ink[local],
                               args.avcrd_window,args.avcrd_proposals,args.max_size)
                    if len(proposals)!=args.avcrd_proposals:
                        raise ValueError('Not enough disjoint attention proposals; reduce --avcrd_proposals or window')
                    rng=torch.Generator().manual_seed(sample_seed(args.seed+8103,0,ids[row]))
                    controls=random_mass_matched_proposals(ink[local],proposals,args.avcrd_window,args.max_size,rng)
                    candidates=[item[0] for item in proposals]+[item[0] for item in controls]
                    views=torch.stack([erase_ink_box(image,box,args.avcrd_ink_threshold,args.avcrd_ink_softness) for box in candidates])
                    encoded=[]
                    for part in views.split(args.avcrd_teacher_batch_size):
                        encoded.append(controller(part.to(device,dtype=dtype),'sketch'))
                    masked=torch.cat(encoded)
                    _,rms=counterfactual_effect(clean[local:local+1].expand(len(masked),-1),masked,bank)
                    rms=rms.view(2,args.avcrd_proposals)
                    best=rms.argmax(dim=-1)
                    payload['clean'][row]=clean[local].half().cpu()
                    payload['attention'][row]=attention[local].half()
                    payload['saliency'][row]=saliency[local].half()
                    payload['candidate_rms'][row]=rms.cpu()
                    payload['candidate_mass'][row]=torch.tensor([[p[2] for p in proposals],[c[1] for c in controls]])
                    payload['selected'][row]=best.cpu().short()
                    for variant in range(2):
                        chosen=variant*args.avcrd_proposals+int(best[variant])
                        payload['masked'][row,variant]=masked[chosen].half().cpu()
                        payload['boxes'][row,variant]=torch.tensor(candidates[chosen])
                        first=variant*args.avcrd_proposals
                        payload['first_masked'][row,variant]=masked[first].half().cpu()
                        payload['first_boxes'][row,variant]=torch.tensor(candidates[first])
                    original=teacher_payload['teacher_sketch_features'][ids[row]].float()
                    payload['clean_cache_cosine'][row]=F.cosine_similarity(original[None],clean[local:local+1].float().cpu()).item()
                offset+=len(images)
        if (payload['clean_cache_cosine']<.999).any():
            raise RuntimeError('Re-encoded clean teacher differs from main cache; inspect teacher/source provenance')
        payload['preparation_seconds']=time.perf_counter()-preparation_started
        validate_payload(payload,metadata,count,width)
        temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            torch.save(payload,temporary)
            if path.exists():
                raise FileExistsError('Another process created the AVCRD cache; refusing overwrite')
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
        del controller,teacher
        if torch.cuda.is_available():torch.cuda.empty_cache()
        print(f'[AVCRD Cache] saved {path}; {path.stat().st_size/1024**2:.1f} MiB; teacher released',flush=True)
    del teacher_payload
    if len(ids)==len(dataset):
        dataset.set_counterfactual_targets(payload)
    from src.counterfactual_diagnostics import cache_report
    cache_report(payload,dataset,Path(report_dir),args)
    return payload
