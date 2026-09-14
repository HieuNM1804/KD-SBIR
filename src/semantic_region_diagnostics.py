"""Read-only measurements; full-set covariance is distinct from minibatch spread."""
from pathlib import Path
import csv
import json
import math
import torch
from torch.nn import functional as F


def directory(module):
    root = Path(module.logger.log_dir) if module.logger else Path(module.trainer.default_root_dir)
    out = root / 'region_diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    return out


def append(path, row):
    present = path.is_file()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not present:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def covariance_statistics(features):
    x = F.normalize(features.float(), dim=-1).cpu().double()
    mean = x.mean(0)
    centered = x - mean
    cov = centered.T @ centered / len(x)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    spread = eig.sum().item()
    p = eig / max(spread, 1e-30)
    rank = math.exp(-(p[p > 0] * p[p > 0].log()).sum().item()) if spread > 1e-20 else 0.
    return {'effective_rank': rank, 'spread': spread, 'centroid_norm': mean.norm().item()}


def gradient_measurements(module, components, out):
    named = [(n,p) for n,p in module.model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    gradients = {}
    for key in ('descriptor', 'region', 'gate', 'spread'):
        loss = .5 * sum(v[key] for v in components.values()) * getattr(module.args, 'lambda_' + key)
        if loss.requires_grad:
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            gradients[key] = [torch.zeros_like(p) if v is None else v.detach() for p,v in zip(params,g)]
    groups = {'head':[i for i,(n,_) in enumerate(named) if 'region_head.' in n],
              'prompts':[i for i,(n,_) in enumerate(named) if '_visual_prompt.' in n]}
    for group, ids in groups.items():
        if not ids:
            continue
        flat = {key:torch.cat([value[i].flatten().double() for i in ids]) for key,value in gradients.items()}
        for key,g in flat.items():
            base = flat['descriptor']
            norm, bn = g.norm().item(), base.norm().item()
            append(out/'gradients.csv', {'epoch':module.current_epoch,'step':module.global_step,
                   'group':group,'component':key,'weighted_gradient_norm':norm,
                   'cosine_with_descriptor':(g@base).item()/(norm*bn) if norm*bn > 1e-20 else None})


@torch.no_grad()
def attention_figure(out, batch, outputs, epoch, grid, mode='semantic'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,4,figsize=(13,11))
    mean = torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
    std = torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
    for row,(modality,index,gate_index) in enumerate([('photo',0,7),('sketch',1,8)]*2):
        sample = min(row//2,len(batch[index])-1)
        image = (batch[index][sample].detach().float().cpu()*std+mean).clamp(0,1).permute(1,2,0)
        values = outputs[modality]
        axes[row,0].imshow(image); axes[row,0].set_title(modality+' input')
        target = batch[gate_index][sample].detach().float().cpu().reshape(grid,grid)
        predicted = values['weights'][sample].detach().float().cpu().reshape(grid,grid)
        axes[row,1].imshow(target,vmin=0,vmax=1,cmap='magma');axes[row,1].set_title('Teacher semantic weights')
        axes[row,2].imshow(predicted,vmin=0,vmax=1,cmap='magma');axes[row,2].set_title('Student weights')
        heat = torch.einsum('r,rn->n', values['weights'][sample],values['attention'][sample]).detach().float().cpu()
        side = int(math.sqrt(len(heat)))
        im=axes[row,3].imshow(heat.reshape(side,side),vmin=0,vmax=max(heat.max().item(),1e-6),cmap='magma')
        fig.colorbar(im,ax=axes[row,3],fraction=.04,pad=.02)
        axes[row,3].set_title('Student patch pooling mass')
        for column,matrix in ((1,target),(2,predicted)):
            for y in range(grid):
                for x in range(grid):
                    axes[row,column].text(x,y,'%.3f'%matrix[y,x].item(),ha='center',va='center',color='white',fontsize=10)
        for axis in axes[row]:axis.axis('off')
    fig.suptitle('Epoch %d; mode=%s; semantic teacher reference and student pooling weights'%(epoch,mode))
    fig.tight_layout();fig.savefig(out/('attention_epoch_%02d.png'%epoch),dpi=140);plt.close(fig)


def training_diagnostics(module,batch,batch_idx,outputs,components,main_loss):
    # Validation precedes Lightning's train-epoch logging: keep our own detached
    # sample-weighted sums so epochs.csv contains this epoch, not the previous one.
    if getattr(module,'_region_epoch_index',None) != module.current_epoch:
        module._region_epoch_index=module.current_epoch
        module._region_epoch_sums={}
        module._region_epoch_count=0
    count=len(batch[0])
    module._region_epoch_count+=count
    for modality,values in components.items():
        for key,value in values.items():
            name=modality+'_'+key
            contribution=value.detach()*count
            module._region_epoch_sums[name]=module._region_epoch_sums.get(name,0)+contribution
    step = module.global_step
    if batch_idx != 0 and step % module.args.region_diagnostic_interval:
        return
    out = directory(module)
    with torch.no_grad():
        for modality,teacher_idx in [('photo',2),('sketch',3)]:
            value = outputs[modality]
            weights = value['weights'].detach()
            target = F.normalize(batch[teacher_idx].float()@module.model.region_alignment,dim=-1)
            row = {'epoch':module.current_epoch,'step':step,'modality':modality,
                   **{k:v.detach().item() for k,v in components[modality].items()},
                   'main_loss':main_loss.detach().item(),
                   'weight_entropy':-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean().item(),
                   'mean_correction_norm':value['correction'].detach().norm(dim=-1).mean().item(),
                   'descriptor_native_cosine':F.cosine_similarity(value['descriptor'],value['native']).mean().item(),
                   'batch_descriptor_spread':value['descriptor'].detach().var(0,correction=0).sum().item(),
                   'batch_target_spread':target.var(0,correction=0).sum().item()}
            append(out/'steps.csv',row)
    if batch_idx == 0:
        gradient_measurements(module,components,out)
        attention_figure(out,batch,outputs,module.current_epoch,module.args.region_grid,module.args.region_mode)
    print('[Region Diagnostics] epoch=%d step=%d; descriptor=%.4f region=%.4f gate=%.4f' %
          (module.current_epoch,step,*[.5*sum(v[k].detach().item() for v in components.values())
                                    for k in ('descriptor','region','gate')]),flush=True)


@torch.no_grad()
def validation_diagnostics(module,sketch,photo,sketch_labels,photo_labels,ap,precision):
    from src.model import _retrieval_metrics
    out = directory(module)
    native_sketch = torch.cat(module.val_native_sk)
    native_photo = torch.cat(module.val_native_ph)
    nap,np,_,_ = _retrieval_metrics(native_sketch.float(),native_photo.float(),sketch_labels,photo_labels,module.args.dataset)
    row = {'epoch':module.current_epoch,'global_step':module.global_step,
           'mAP':ap.item(),'precision':precision.item(),'native_mAP':nap.item(),'native_precision':np.item()}
    for modality,x,native in [('sketch',sketch,native_sketch),('photo',photo,native_photo)]:
        for variant,features in [('descriptor',x),('native',native)]:
            row.update({modality+'_'+variant+'_'+k:v for k,v in covariance_statistics(features).items()})
    # Our detached epoch sums are current sample-weighted means.
    for modality in ('photo','sketch'):
        for key in ('descriptor','region','gate','reference','spread'):
            value = getattr(module,'_region_epoch_sums',{}).get(modality+'_'+key)
            row[modality+'_'+key] = value.item()/module._region_epoch_count if value is not None else None
    append(out/'epochs.csv',row)
    if module.global_step == 0:
        (out/'initial_validation.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('[Region Validation] descriptor mAP=%.2f native mAP=%.2f; effective rank sketch/photo=%.2f/%.2f' %
          (100*ap.item(),100*nap.item(),row['sketch_descriptor_effective_rank'],row['photo_descriptor_effective_rank']),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'epochs.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    x=[int(r['global_step']) for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for key in ('mAP','native_mAP','precision'):
        axes[0].plot(x,[100*float(r[key]) for r in rows],'o-',label=key)
    for key in ('photo_descriptor_effective_rank','sketch_descriptor_effective_rank'):
        axes[1].plot(x,[float(r[key]) for r in rows],'o-',label=key)
    for key in ('photo_descriptor','sketch_descriptor','photo_region','sketch_region'):
        points=[(int(r['global_step']),float(r[key])) for r in rows if r.get(key)]
        if points:axes[2].plot(*zip(*points),'o-',label=key)
    for ax,title in zip(axes,('Full validation (%)','Full-set centered covariance rank','Training epoch mean losses')):
        ax.set_title(title);ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'training_diagnostics.png',dpi=140);plt.close(fig)
