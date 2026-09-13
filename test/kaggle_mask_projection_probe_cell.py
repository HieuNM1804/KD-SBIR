"""Paste this entire file into one Kaggle cell. Evaluate existing mask checkpoints only."""
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

# Leave empty to find one completed run for each method in the CURRENT project.
# If a method has several runs, put the exact final.ckpt paths here.
CHECKPOINTS = []
PROJECT = '/kaggle/working/KD-SBIR-AVKD'
ROOT_OVERRIDE = ''  # Only needed if the Sketchy input directory moved.

RUNNER_SOURCE = r'''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
from argparse import Namespace
import csv
import hashlib
import json
import math
import sys
import zipfile
from unittest.mock import patch
import torch
from torch.nn import functional as F


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024**2), b''): h.update(part)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def select_checkpoints(project, explicit):
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths = []
        for method in ('embedding', 'random', 'guided'):
            matches = sorted((project/'saved_models').glob(f'mask_{method}_*/final.ckpt'))
            if len(matches) > 1:
                raise ValueError(f'Several {method} runs exist. Set CHECKPOINTS explicitly:\n' + '\n'.join(map(str, matches)))
            paths.extend(matches)
    if not paths:
        raise FileNotFoundError('No final mask checkpoints found. Set CHECKPOINTS to existing paths; do not retrain.')
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    return paths


def frozen_state(saved):
    prefix = 'model.clip_model.'
    result = {k[len(prefix):]: v for k,v in saved['state_dict'].items() if k.startswith(prefix)}
    if not result: raise ValueError('Checkpoint does not contain frozen student CLIP weights')
    return result


def construct_initial(saved):
    # Construct from checkpoint CLIP, retaining the ORIGINAL seeded prompt and W
    # initialization. No teacher, teacher cache, or pretrained download is needed.
    from clip.model import build_model
    import src.model as source
    from src.train import seed_everything
    args = Namespace(**saved['experiment_config']['args'])
    seed_everything(args.seed)
    with patch.object(source, '_load_teacher', return_value=None), \
         patch.object(source, '_load_clip_model', side_effect=lambda _: build_model(dict(frozen_state(saved)))):
        module = source.ZS_SBIR(args, saved['hyper_parameters']['classnames']).eval()
    return module.requires_grad_(False)


@torch.no_grad()
def encode(module, loaders, unprompted=False):
    from tqdm.auto import tqdm
    outputs, labels = {}, {}
    for mod, loader in loaders.items():
        parts, targets = [], []
        for images, category in tqdm(loader, desc='[Probe] encode '+mod, mininterval=2):
            images = images.cuda(non_blocking=True)
            if unprompted:
                z = module.model.clip_model.encode_image(images)
                z = z/z.norm(dim=-1, keepdim=True)
            else:
                z = module.model.encode_student_image(images, mod)
            if not torch.isfinite(z).all(): raise ValueError('Nonfinite image descriptor')
            # Retain native dtype for the projection, exactly as in training.
            parts.append(z.cpu()); targets.append(category.cpu())
        outputs[mod], labels[mod] = torch.cat(parts), torch.cat(targets)
    return outputs, labels


@torch.no_grad()
def project_features(features, weight):
    weight = weight.cuda().float()
    return {mod: torch.cat([F.normalize(F.linear(part.cuda().float(), weight), dim=-1).cpu()
                           for part in x.split(1024)]) for mod,x in features.items()}


@torch.no_grad()
def retrieval(features, labels, dataset):
    # Use the original main scoring formula and TorchMetrics AP/P implementations.
    # FP32 for ALL descriptors to avoid a half/float comparison confound.
    from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision
    from tqdm.auto import tqdm
    query, gallery = features['sketch'].cuda().float(), features['photo'].cuda().float()
    map_k = 200 if dataset == 'sketchy_2' else 0
    p_k = 200 if dataset in ('sketchy_2','quickdraw') else 100
    aps, ps = [], []
    for i,q in enumerate(tqdm(query, desc='[Probe] full retrieval', mininterval=5)):
        score = ((F.cosine_similarity(q[None], gallery).cpu()+1)*.5).clamp(min=torch.finfo(torch.float32).eps,max=1)
        target = labels['photo'].eq(labels['sketch'][i])
        ap = retrieval_average_precision(score, target, top_k=min(map_k,len(gallery)) if map_k else None)
        aps.append(ap.item()); ps.append(retrieval_precision(score,target,top_k=p_k).item())
    return {'mAP': sum(aps)/len(aps), 'precision': sum(ps)/len(ps), 'map_k':map_k, 'p_k':p_k,
            'queries':len(query),'gallery':len(gallery)}


@torch.no_grad()
def feature_statistics(features, labels):
    # Full-set centered covariance, NOT the within-batch training statistic.
    x = F.normalize(features.float(), dim=-1).double().cpu()
    n,d = x.shape
    mean = x.mean(0); x = x-mean
    covariance = (x.T@x)/n
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum()
    p = eigenvalues/total.clamp_min(1e-30)
    effective_rank = math.exp(-(p[p>0]*p[p>0].log()).sum().item()) if total>1e-20 else 0.
    rank95 = int(torch.searchsorted(p.cumsum(0),p.new_tensor(.95)).item()+1) if total>1e-20 else 0
    between = sum((labels==c).sum().item()/n * x[labels==c].mean(0).square().sum().item() for c in labels.unique())
    row = {'samples':n,'dimensions':d,'centroid_norm':mean.norm().item(),'total_spread':total.item(),
           'mean_off_diagonal_cosine':(n*mean.square().sum().item()-1)/(n-1) if n>1 else None,
           'effective_covariance_rank':effective_rank,'rank95':rank95,
           'between_class_spread':between,'within_class_spread':max(0.,total.item()-between)}
    spectrum = [{'index':i+1,'centered_singular_value':math.sqrt(n*v.item()),
                 'variance_fraction':p[i].item()} for i,v in enumerate(eigenvalues)]
    return row, spectrum


def figure(out, metrics, stats):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(17,max(6,len(metrics)*.32)))
    names = [r['run']+' / '+r['variant'] for r in metrics]
    y = list(range(len(names)))
    axes[0].barh(y,[100*r['mAP'] for r in metrics],label='mAP')
    axes[0].set_yticks(y,names,fontsize=7); axes[0].invert_yaxis(); axes[0].set_xlabel('Full validation mAP (%)')
    for mod,offset in [('photo',-.18),('sketch',.18)]:
        values = [next(s['effective_covariance_rank'] for s in stats if s['run']==r['run'] and s['variant']==r['variant'] and s['modality']==mod) for r in metrics]
        axes[1].barh([i+offset for i in y],values,height=.35,label=mod)
    axes[1].invert_yaxis(); axes[1].set_yticks(y,[]); axes[1].set_xlabel('Effective centered covariance rank');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'projection_probe.png',dpi=160);plt.close(fig)


def main():
    config = json.loads(os.environ['MASK_PROBE_CONFIG'])
    project, out = Path(config['project']), Path(config['out'])
    sys.path.insert(0,str(project))
    from src.dataset import ValidDataset
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available(): raise RuntimeError('Enable a Kaggle GPU')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    paths = select_checkpoints(project, config['checkpoints'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'runner.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    manifest = {'completed':False,'checkpoints':[],'notes':[
        'Initial prompts and projection reconstructed from their local deterministic seeds.',
        'Frozen CLIP state must match across checkpoints; no teacher is loaded.',
        'All retrieval uses full validation, original main metric formula, FP32 scoring.',
        'Native features retain prompts; CLIP_no_prompts is a separate reference.',
        'Swapped prompt/projection combinations are interventions, not trained models.',
        'Centered covariance statistics are computed on the full unseen set per modality.',
        'Effective rank = exp(entropy of normalized covariance eigenvalues).',
        'No checkpoint file is overwritten; no optimizer step is performed.']}
    metrics, stats, spectra, drift = [],[],[],[]
    reference_args = reference_classes = frozen_hash = initial = w0 = None
    labels = None
    def evaluate(run, variant, features, args):
        print(f'[Probe] {run} / {variant}',flush=True)
        result = {'run':run,'variant':variant,**retrieval(features,labels,args.dataset)}
        metrics.append(result)
        for mod,x in features.items():
            row, spectrum = feature_statistics(x,labels[mod])
            stats.append({'run':run,'variant':variant,'modality':mod,**row})
            spectra.extend({'run':run,'variant':variant,'modality':mod,**s} for s in spectrum)
        write_csv(out/'retrieval.csv',metrics);write_csv(out/'feature_statistics.csv',stats)
        write_csv(out/'feature_spectra.csv',spectra)
        print('[Probe] result:',json.dumps(result),flush=True)
        return result
    for path in paths:
        print('[Probe] checkpoint:',path,flush=True)
        digest = file_hash(path)
        saved = torch.load(path,map_location='cpu',weights_only=False)
        experiment = saved.get('experiment_config',{})
        if experiment.get('retrieval_head')!='mask_guided': raise ValueError('Expected mask-guided checkpoint')
        args = Namespace(**experiment['args'])
        for name,expected in args.training_source_sha256.items():
            if file_hash(project/name)!=expected: raise ValueError('Training source differs: '+name)
        signature = {k:getattr(args,k) for k in ('dataset','backbone','max_size','seed','n_ctx_visual','prompt_depth')}
        classes = saved['hyper_parameters']['classnames']
        digest_frozen = state_hash(frozen_state(saved))
        if reference_args is not None and (signature!=reference_args or classes!=reference_classes or digest_frozen!=frozen_hash):
            raise ValueError('Runs have different initialization/backbone/class configuration; probe separately')
        module = construct_initial(saved).cuda().eval()
        if config.get('root_override'): args.root=config['root_override']
        loaders = {mod:DataLoader(ValidDataset(args,mod),batch_size=256,num_workers=4,shuffle=False,
                                 generator=torch.Generator().manual_seed(args.seed+900),pin_memory=True)
                   for mod in ('sketch','photo')}
        if any(len(loader.dataset)==0 for loader in loaders.values()): raise ValueError('Empty validation dataset; check ROOT_OVERRIDE')
        if reference_args is None:
            reference_args,reference_classes,frozen_hash = signature,classes,digest_frozen
            initial,labels = encode(module,loaders)
            w0 = module.model.retrieval_projection.weight.detach().cpu().clone()
            manifest['validation_paths_sha256']={mod:hashlib.sha256('\n'.join(loader.dataset.paths).encode()).hexdigest() for mod,loader in loaders.items()}
            plain,plain_labels = encode(module,loaders,unprompted=True)
            evaluate('initial','CLIP_no_prompts',plain,args); del plain
            evaluate('initial','native_with_initial_prompts',initial,args)
            initial_metrics=evaluate('initial','initial_prompts_initial_W',project_features(initial,w0),args)
        else:
            if not torch.equal(w0,module.model.retrieval_projection.weight.detach().cpu()):raise ValueError('Initial W differs')
            if any(hashlib.sha256('\n'.join(loader.dataset.paths).encode()).hexdigest()!=manifest['validation_paths_sha256'][mod] for mod,loader in loaders.items()):raise ValueError('Validation path order differs')
        module.load_state_dict(saved['state_dict'],strict=True)
        trained,current_labels = encode(module,loaders)
        if any(not torch.equal(labels[m],current_labels[m]) for m in labels):raise ValueError('Validation labels differ')
        wt = module.model.retrieval_projection.weight.detach().cpu().clone()
        run = path.parent.name
        for mod in trained:
            cosine = F.cosine_similarity(initial[mod].float(),trained[mod].float(),dim=-1)
            drift.append({'run':run,'modality':mod,'mean_native_initial_final_cosine':cosine.mean().item()})
        evaluate(run,'trained_prompts_native',trained,args)
        evaluate(run,'initial_prompts_trained_W',project_features(initial,wt),args)
        evaluate(run,'trained_prompts_initial_W',project_features(trained,w0),args)
        deployed=evaluate(run,'trained_prompts_trained_W',project_features(trained,wt),args)
        # Save singular spectra of the linear map itself, separately from features.
        for stage,weight in [('initial',w0),('trained',wt)]:
            singular=torch.linalg.svdvals(weight.double())
            write_csv(out/(run+'_'+stage+'_W_spectrum.csv'),[{'index':i+1,'singular_value':v.item()} for i,v in enumerate(singular)])
        provenance={'path':str(path),'sha256':digest,'epoch':saved['epoch'],'global_step':saved['global_step'],
                    'deployed_metrics':deployed,'initialization_signature':signature,'frozen_CLIP_sha256':digest_frozen}
        logged=project/'tb_logs'/run
        candidates=list(logged.glob('version_*/mask_diagnostics/epochs.csv'))
        if len(candidates)==1:
            with candidates[0].open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            match=[r for r in rows if int(r['global_step'])==saved['global_step']]
            if len(match)==1:
                provenance['reproduction_difference']={k:deployed[k]-float(match[0][k]) for k in ('mAP','precision')}
                print('[Probe] difference from logged final metrics:',provenance['reproduction_difference'],flush=True)
                provenance['final_metrics_reproduced_within_0.05pp']=all(abs(v)<=.0005 for v in provenance['reproduction_difference'].values())
                if not provenance['final_metrics_reproduced_within_0.05pp']:
                    print('[Probe] WARNING: final metrics differ by >0.05 percentage points; inspect provenance before interpreting swaps.',flush=True)
            initial_log=candidates[0].with_name('initial_validation.json')
            if initial_log.is_file():
                original=json.loads(initial_log.read_text(encoding='utf-8'))
                provenance['initial_reproduction_difference']={k:initial_metrics[k]-original[k] for k in ('mAP','precision')}
        if file_hash(path)!=digest:raise RuntimeError('Checkpoint file changed during probe')
        provenance['checkpoint_unchanged']=True
        manifest['checkpoints'].append(provenance)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        write_csv(out/'native_feature_drift.csv',drift)
        del saved,module,trained,wt;torch.cuda.empty_cache()
    figure(out,metrics,stats)
    manifest['completed']=True
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.is_file():archive.write(p,p.name)
    print('[Probe] ZIP:',out.with_suffix('.zip'),flush=True)


if __name__=='__main__': main()
'''

if __name__ == '__main__':
    import json
    name='mask_projection_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=Path('/kaggle/working')/name
    runner=out.with_name(name+'_runner.py')
    compile(RUNNER_SOURCE,str(runner),'exec')
    runner.write_text(RUNNER_SOURCE,encoding='utf-8')
    env=os.environ.copy()
    env['MASK_PROBE_CONFIG']=json.dumps({'project':PROJECT,'out':str(out),'checkpoints':CHECKPOINTS,'root_override':ROOT_OVERRIDE})
    log=runner.with_suffix('.log')
    print('Probe log:',log,flush=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([sys.executable,'-u',str(runner)],cwd=PROJECT,env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            stream.write(line);stream.flush();print(line.rstrip(),flush=True)
        code=process.wait()
    if code:raise RuntimeError(f'Projection probe failed; full log: {log}')
    from IPython.display import FileLink,Image,display
    display(Image(filename=str(out/'projection_probe.png')))
    display(FileLink(str(out.with_suffix('.zip'))))
