"""Paired attention/random interventions on seen images, with compact targets."""
import hashlib
import csv
import json
import os
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import TeacherFeatureDataset
from src.mask_guided import TeacherAttention, make_masks, apply_mask

BASELINE_COMMIT='b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def dataset_hash(dataset,root):
    h=hashlib.sha256()
    for path in tqdm(dataset.all_sketches_path+dataset.all_photo_paths,desc='[Mask Cache] image hashes'):
        h.update(Path(path).relative_to(root).as_posix().encode()+b'\0')
        h.update(bytes.fromhex(file_hash(path)))
    return h.hexdigest()


def load_teacher(payload):
    import open_clip
    from src.teacher_prompts import TeacherPromptController
    m=payload['metadata']; device='cuda' if torch.cuda.is_available() else 'cpu'
    teacher=open_clip.create_model(m['teacher_model'],pretrained=m['teacher_pretrained'],
                                  precision='fp16' if device=='cuda' else 'fp32',device=device)
    teacher.eval().requires_grad_(False)
    controller=TeacherPromptController(teacher.visual,m['teacher_n_ctx_visual'],m['teacher_prompt_depth'],
                                      m['teacher_prompt_std'],m['teacher_prompt_seed'])
    controller.load_state_dict(payload['teacher_prompt_state_dict'],strict=True)
    controller.eval().requires_grad_(False)
    return teacher,controller


def validate_cache(cache,metadata,lengths):
    if cache.get('metadata')!=metadata:raise ValueError('Mask cache provenance/config mismatch; choose a new --mask_cache_path')
    grid=metadata['grid']; count=metadata['masked_cells']
    for mod,n in lengths.items():
        entry=cache[mod]
        for name in ('full','attention_target','random_target'):
            value=entry[name]
            if value.shape!=(n,1024) or value.dtype!=torch.float16:raise ValueError('Invalid mask target shape/dtype')
            for batch in value.split(2048):
                if not torch.isfinite(batch).all() or not torch.allclose(batch.float().norm(dim=-1),torch.ones(len(batch)),atol=.002,rtol=0):
                    raise ValueError('Invalid mask target values')
        for name in ('attention_mask','random_mask'):
            value=entry[name]
            if value.shape!=(n,grid,grid) or value.dtype!=torch.bool or not (value.sum((-1,-2))==count).all():
                raise ValueError('Invalid mask size/area')


def save_cache(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload,tmp)
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def visualize(images,attention,guided,random,paths,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.dataset import CLIP_MEAN,CLIP_STD
    mean=torch.tensor(CLIP_MEAN)[None,:,None,None];std=torch.tensor(CLIP_STD)[None,:,None,None]
    def rgb(x):return (x.cpu()*std+mean).clamp(0,1).permute(0,2,3,1).numpy()
    original=rgb(images); ga=rgb(apply_mask(images,guided));ra=rgb(apply_mask(images,random))
    fig,axes=plt.subplots(len(images),4,figsize=(12,3*len(images)),squeeze=False)
    for i in range(len(images)):
        for j,pixels in ((0,original[i]),(2,ga[i]),(3,ra[i])):axes[i,j].imshow(pixels)
        axes[i,1].imshow(attention[i].reshape(guided.shape[-2:]).cpu().numpy(),cmap='magma')
        axes[i,0].set_title(Path(paths[i]).parent.name)
        for j in range(4):axes[i,j].axis('off')
    for j,title in enumerate(('Original','Teacher CLS attention','Attention mask','Random mask')):axes[0,j].set_title(title)
    fig.suptitle('Seen-image supervision examples; masks have equal area; fill = CLIP mean RGB')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


@torch.no_grad()
def prepare_mask_cache(args,dataset,expected_metadata,report_dir):
    path=Path(args.teacher_cache_path)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['metadata']!=expected_metadata or not payload.get('teacher_prompt_state_dict'):
        raise ValueError('Matching tuned teacher cache with prompt weights is required')
    grid=args.mask_grid;count=max(1,min(grid*grid-1,round(args.mask_ratio*grid*grid)))
    metadata={'version':1,'baseline_commit':BASELINE_COMMIT,'teacher_sha256':file_hash(path),
              'teacher_metadata':payload['metadata'],'image_sha256':dataset_hash(dataset,args.root),
              'grid':grid,'ratio':args.mask_ratio,'masked_cells':count,'seed':args.seed,'max_size':args.max_size,
              'definition':'mean_head_last_CLS_patch_attention_then_area_pool_topk_v1',
              'fill':'zero_in_CLIP_normalized_space','fit_split':'seen_training_only',
              'torch':str(torch.__version__),
              'sources':{name:file_hash(Path(__file__).with_name(name)) for name in
                         ('mask_guided.py','mask_guided_cache.py','teacher_prompts.py','dataset.py')}}
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:16]
    path=Path(args.mask_cache_path) if args.mask_cache_path else path.parent/f'{args.dataset}_mask_response_{key}.pt'
    args.mask_cache_path=str(path)
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    lengths={'sketch':len(dataset.all_sketches_path),'photo':len(dataset.all_photo_paths)}
    if path.is_file():
        cache=torch.load(path,map_location='cpu',weights_only=True);validate_cache(cache,metadata,lengths)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        estimated=sum(lengths.values())*(3*1024*2+2*grid*grid)+128*1024**2
        if shutil.disk_usage(path.parent).free<estimated:raise OSError(f'Mask cache needs ~{estimated/1024**3:.2f} GiB free')
        teacher,controller=load_teacher(payload);parameter=teacher.visual.conv1.weight
        cache={'metadata':metadata,'summary':{},'examples':{},'audit':{}}
        for mod,paths in [('photo',dataset.all_photo_paths),('sketch',dataset.all_sketches_path)]:
            loader=DataLoader(TeacherFeatureDataset(paths,args.max_size),batch_size=args.mask_teacher_batch_size,
                              num_workers=args.workers,shuffle=False,generator=torch.Generator().manual_seed(args.seed+811))
            n=len(paths);entry={name:torch.empty(n,1024,dtype=torch.float16) for name in ('full','attention_target','random_target')}
            entry.update({name:torch.empty(n,grid,grid,dtype=torch.bool) for name in ('attention_mask','random_mask')})
            audit={name:[] for name in ('attention_delta','random_delta','attention_mass','random_mass','attention_entropy',
                                        'attention_dark_fraction','random_dark_fraction')}
            # Examples spread across ordered training paths, rather than only the first class.
            example_ids=set(torch.linspace(0,n-1,min(4,n)).long().tolist());examples=[];offset=0
            for images in tqdm(loader,desc=f'[Mask Cache] {mod} original + attention + random'):
                device_images=images.to(device=parameter.device,dtype=parameter.dtype)
                with TeacherAttention(teacher.visual) as capture:full=controller(device_images,mod)
                if len(capture.values)!=1:raise RuntimeError('Expected one teacher attention capture')
                batch_paths=paths[offset:offset+len(images)]
                relative=[Path(p).relative_to(args.root).as_posix() for p in batch_paths]
                scores,guided,random=make_masks(capture.values[0],relative,grid,args.mask_ratio,args.seed+812)
                full=F.normalize(full.float(),dim=-1).cpu()
                entry['full'][offset:offset+len(images)]=full.half()
                from src.dataset import CLIP_MEAN,CLIP_STD
                rgb=images*torch.tensor(CLIP_STD)[None,:,None,None]+torch.tensor(CLIP_MEAN)[None,:,None,None]
                dark=(rgb.mean(1)<.8).float() # Descriptive only; dark pixels are not a semantic foreground label.
                for strategy,mask in [('attention',guided),('random',random)]:
                    masked=controller(apply_mask(device_images,mask),mod)
                    masked=F.normalize(masked.float(),dim=-1).cpu()
                    entry[strategy+'_target'][offset:offset+len(images)]=masked.half()
                    entry[strategy+'_mask'][offset:offset+len(images)]=mask
                    audit[strategy+'_delta'].append((full-masked).norm(dim=-1))
                    audit[strategy+'_mass'].append((scores*mask.flatten(1)).sum(-1))
                    pixels=mask.repeat_interleave(args.max_size//grid,-2).repeat_interleave(args.max_size//grid,-1)
                    audit[strategy+'_dark_fraction'].append((dark*pixels).sum((-1,-2))/dark.sum((-1,-2)).clamp_min(1))
                audit['attention_entropy'].append(-(scores*scores.clamp_min(1e-12).log()).sum(-1))
                for i in range(len(images)):
                    if offset+i in example_ids:examples.append((images[i],scores[i],guided[i],random[i],batch_paths[i]))
                offset+=len(images)
            audit={name:torch.cat(values) for name,values in audit.items()}
            cache['audit'][mod]=audit
            cache['summary'][mod]={name:{'mean':v.mean().item(),'q10':v.quantile(.1).item(),'median':v.median().item(),'q90':v.quantile(.9).item()} for name,v in audit.items()}
            cache['summary'][mod]['fraction_attention_delta_gt_random']=(audit['attention_delta']>audit['random_delta']).float().mean().item()
            cache[mod]=entry
            cache['examples'][mod]={'images':torch.stack([e[0] for e in examples]),'attention':torch.stack([e[1] for e in examples]),
                                    'guided':torch.stack([e[2] for e in examples]),'random':torch.stack([e[3] for e in examples]),
                                    'paths':[e[4] for e in examples]}
        validate_cache(cache,metadata,lengths);save_cache(path,cache)
        del teacher,controller
        if torch.cuda.is_available():torch.cuda.empty_cache()
    args.mask_target_metadata=metadata
    # Same full targets for embedding-only, random and guided ablations.
    dataset.set_teacher_features(cache['sketch']['full'],cache['photo']['full'])
    if args.lambda_response>0:dataset.set_mask_features(cache,args.mask_strategy)
    else:dataset.mask_features=None
    report={'metadata':metadata,'summary':cache['summary'],
            'notes':['Seen images only; these are perturbation statistics, not retrieval gains.',
                     'Attention and random masks have equal area, not necessarily equal foreground/stroke coverage.',
                     'dark_fraction is fraction of pixels below grayscale 0.8 removed, not a foreground annotation.',
                     'Larger teacher response alone does not prove better supervision.',
                     'One fixed attention and random mask per image; no independent mask resampling per epoch.']}
    (report_dir/'teacher_mask_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for mod,paths in [('photo',dataset.all_photo_paths),('sketch',dataset.all_sketches_path)]:
        values=cache['audit'][mod]
        with (report_dir/f'{mod}_teacher_mask_samples.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['path']+list(values))
            writer.writeheader()
            for i,path_name in enumerate(paths):
                writer.writerow({'path':Path(path_name).relative_to(args.root).as_posix(),
                                 **{key:value[i].item() for key,value in values.items()}})
    for mod,example in cache['examples'].items():
        visualize(example['images'],example['attention'],example['guided'],example['random'],example['paths'],report_dir/f'{mod}_mask_examples.png')
        summary=cache['summary'][mod]
        print(f'[Mask Teacher] {mod}: delta attention={summary["attention_delta"]["mean"]:.4f}, random={summary["random_delta"]["mean"]:.4f}; attention > random fraction={summary["fraction_attention_delta_gt_random"]:.3f}',flush=True)
    print(f'[Mask Cache] ready: {path}; {path.stat().st_size/1024**2:.1f} MiB; diagnostics: {report_dir}',flush=True)
    return cache
