"""Teacher falsification and fixed-batch/full-validation AVCRD diagnostics."""
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
from pytorch_lightning import Callback
from src.counterfactual_retrieval_kd import (
    CLIP_MEAN, CLIP_STD, ink_alpha, erase_ink_box, erase_ink_batch, similarity_field,
)
from src.losses import loss_fn


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _rgb(image):
    image=image.float().cpu()
    return (image*image.new_tensor(CLIP_STD).view(3,1,1)+image.new_tensor(CLIP_MEAN).view(3,1,1)).clamp(0,1).permute(1,2,0).numpy()


def cache_report(payload,dataset,out,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.dataset import TeacherFeatureDataset
    out.mkdir(parents=True,exist_ok=True)
    rms=payload['candidate_rms'].float()
    attention_first=rms[:,0,0];random_first=rms[:,1,0]
    verified=rms[:,0].max(-1).values;random_verified=rms[:,1].max(-1).values
    ratio=float(attention_first.median()/random_first.median().clamp_min(1e-8))
    wins=float((attention_first>random_first).float().mean())
    selected_ratio=float(verified.median()/random_verified.median().clamp_min(1e-8))
    summary={'samples':len(rms),'teacher_only':True,'cache_preparation_seconds':payload.get('preparation_seconds'),'attention_first_median_rms':float(attention_first.median()),
             'random_first_median_rms':float(random_first.median()),'attention_first_median_ratio':ratio,
             'attention_first_win_fraction':wins,'equal_budget_selected_median_ratio':selected_ratio,
             'equal_budget_selected_win_fraction':float((verified>random_verified).float().mean()),
             'exploratory_attention_gate_pass':ratio>=1.2 and wins>=.65,
             'notes':['The gate uses first proposals BEFORE response-based verification selection.',
                      'Selected comparison gives attention/random the same teacher encoding budget.',
                      'Random windows have equal area and approximate ink-mass matching; inspect the exported residuals.',
                      'Response size measures influence, not correct class evidence or improved retrieval.',
                      'Bank consists exclusively of fixed seen training photo features; no unseen labels enter selection.'],
             'metadata':payload['metadata']}
    rows=[]
    for row,index in enumerate(payload['metadata']['indices']):
        for variant,name in enumerate(('attention','random')):
            selected=int(payload['selected'][row,variant])
            mass=payload['candidate_mass'][row,variant,selected].item()
            other_mass=payload['candidate_mass'][row,1-variant,int(payload['selected'][row,1-variant])].item()
            rows.append({'sketch_index':index,'selection':name,'first_proposal_rms':rms[row,variant,0].item(),
                         'selected_proposal':selected,'selected_rms':rms[row,variant,selected].item(),
                         'selected_ink_mass':mass,'selected_pair_relative_ink_difference':abs(mass-other_mass)/max(mass,1e-6),
                         'top':int(payload['boxes'][row,variant,0]),'left':int(payload['boxes'][row,variant,1]),
                         'bottom':int(payload['boxes'][row,variant,2]),'right':int(payload['boxes'][row,variant,3])})
    write_csv(out/'teacher_effects.csv',rows)
    candidate_rows=[]
    for row,index in enumerate(payload['metadata']['indices']):
        for proposal in range(payload['metadata']['proposals']):
            mass_a=payload['candidate_mass'][row,0,proposal].item()
            mass_b=payload['candidate_mass'][row,1,proposal].item()
            candidate_rows.append({'sketch_index':index,'proposal':proposal,
                'attention_rms':rms[row,0,proposal].item(),'random_rms':rms[row,1,proposal].item(),
                'attention_ink_mass':mass_a,'random_ink_mass':mass_b,
                'relative_ink_difference':abs(mass_a-mass_b)/max(mass_a,1e-6)})
    write_csv(out/'candidate_effects.csv',candidate_rows)
    (out/'teacher_probe.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,x in [('attention first',attention_first),('random first',random_first)]:axes[0].hist(x.numpy(),bins=40,alpha=.5,label=name)
    axes[0].set(title=f'Before verification: median ratio {ratio:.2f}; wins {wins:.1%}',xlabel='Centered retrieval response RMS');axes[0].legend()
    axes[1].scatter(random_verified.numpy(),verified.numpy(),s=3,alpha=.2)
    maximum=max(float(verified.max()),float(random_verified.max()),1e-5)
    axes[1].plot([0,maximum],[0,maximum],color='gray');axes[1].set(xlabel='Random selected response RMS',ylabel='Attention selected response RMS',title='Equal verification budget')
    mass_a=payload['candidate_mass'][:,0].gather(1,payload['selected'][:,0,None].long())[:,0]
    mass_b=payload['candidate_mass'][:,1].gather(1,payload['selected'][:,1,None].long())[:,0]
    axes[2].scatter(mass_b.numpy(),mass_a.numpy(),s=3,alpha=.2);axes[2].set(xlabel='Random selected ink mass',ylabel='Attention selected ink mass',title='Ink-mass matching audit')
    fig.tight_layout();fig.savefig(out/'teacher_effects.png',dpi=160);plt.close(fig)
    count=min(args.avcrd_diagnostic_examples,len(rms))
    if count:
        generator=torch.Generator().manual_seed(args.seed+8190)
        chosen=torch.randperm(len(rms),generator=generator)[:count].tolist()
        data=TeacherFeatureDataset(dataset.all_sketches_path,dataset.max_size)
        fig,axes=plt.subplots(count,5,figsize=(15,3*count),squeeze=False)
        grid=math.isqrt(payload['metadata']['teacher_patch_count'])
        for line,row in enumerate(chosen):
            image=data[payload['metadata']['indices'][row]];rgb=_rgb(image)
            axes[line,0].imshow(rgb);axes[line,0].set_title(f'Sketch index {payload["metadata"]["indices"][row]}')
            for col,key,title in [(1,'attention','Raw CLS attention'),(2,'saliency','AVWO proposal norm')]:
                axes[line,col].imshow(rgb)
                heat=F.interpolate(payload[key][row].float().view(1,1,grid,grid),size=image.shape[-2:],mode='bilinear',align_corners=False)[0,0].numpy()
                axes[line,col].imshow(heat,cmap='inferno',alpha=.6);axes[line,col].set_title(title)
            for variant,name in enumerate(('Verified attention','Matched random')):
                view=erase_ink_box(image,payload['boxes'][row,variant],args.avcrd_ink_threshold,args.avcrd_ink_softness)
                axes[line,variant+3].imshow(_rgb(view));axes[line,variant+3].set_title(f'{name}; RMS {rms[row,variant].max():.4f}')
            for axis in axes[line]:axis.axis('off')
        fig.tight_layout();fig.savefig(out/'teacher_interventions.png',dpi=130);plt.close(fig)
    print('[AVCRD Probe]',json.dumps({k:v for k,v in summary.items() if k not in ('metadata','notes')}),flush=True)
    return summary


def feature_statistics(features, labels=None):
    x=F.normalize(features.detach().float().cpu(),dim=-1).double()
    mean=x.mean(0);x=x-mean
    covariance=x.T@x/max(1,len(x))
    values=torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total=values.sum();p=values/total.clamp_min(1e-20)
    rank=math.exp(-(p[p>0]*p[p>0].log()).sum().item()) if total>1e-15 else 0.
    stats = {'samples':len(x),'centroid_norm':mean.norm().item(),'total_spread':total.item(),
            'effective_rank':rank,'rank95':int(torch.searchsorted(p.cumsum(0),p.new_tensor(.95)))+1 if total>1e-15 else 0}
    if labels is not None:
        labels=labels.detach().cpu()
        between=sum((labels==c).sum().item()/len(x)*x[labels==c].mean(0).square().sum().item() for c in labels.unique())
        stats.update(between_class_spread=between,within_class_spread=max(0.,total.item()-between))
    return stats


def _gradient_statistics(a,b):
    na=math.sqrt(sum(x.square().sum().item() for x in a));nb=math.sqrt(sum(x.square().sum().item() for x in b))
    dot=sum((x*y).sum().item() for x,y in zip(a,b))
    return {'main_norm':na,'weighted_cf_norm':nb,'cf_over_main':nb/na if na>1e-12 else None,
            'cosine':dot/(na*nb) if na*nb>1e-20 else None,'dot':dot}


class CounterfactualDiagnostics(Callback):
    """No optimizer steps, no RNG consumption, no overwriting accumulated grads."""
    def _setup(self,trainer,module):
        if hasattr(self,'out'):return
        self.out=Path(trainer.log_dir or trainer.default_root_dir)/'avcrd_diagnostics'
        self.out.mkdir(parents=True,exist_ok=True);self.epochs=[];self.gradients=[];self.fixed=[]
        (self.out/'configuration.json').write_text(json.dumps(vars(module.args),indent=2),encoding='utf-8')

    def on_train_start(self,trainer,module):
        self._setup(trainer,module)
        dataset=trainer.train_dataloader.dataset
        generator=torch.Generator().manual_seed(module.args.seed+8200)
        self.indices=torch.randperm(len(dataset),generator=generator)[:module.args.avcrd_diagnostic_batch_size].tolist()
        self.batch=default_collate([dataset[(0,i)] for i in self.indices])
        self.initial_native=None
        (self.out/'fixed_batch.json').write_text(json.dumps({'sample_epoch':0,'indices':self.indices,
            'notes':'Seen training images only. Raw gradients before optimizer momentum; native inference descriptors.'},indent=2),encoding='utf-8')
        self.measure(trainer,module,'initial')

    def on_train_epoch_start(self,trainer,module):
        self.epoch_started=time.perf_counter()

    def on_validation_start(self,trainer,module):
        self.validation_started=time.perf_counter()

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        self._setup(trainer,module)
        current=module._avcrd_validation_statistics
        row={'epoch':int(trainer.current_epoch)+1 if trainer.global_step else 0,
             'global_step':int(trainer.global_step),'mAP':float(trainer.callback_metrics['mAP']),
             'precision':float(trainer.callback_metrics['precision']),
             'train_plus_validation_seconds':time.perf_counter()-getattr(self,'epoch_started',self.validation_started)}
        for modality,stats in current.items():row.update({modality+'_'+k:v for k,v in stats.items()})
        self.epochs.append(row);write_csv(self.out/'epochs.csv',self.epochs)
        if module._avcrd_retrieval_rows:
            per_query=module._avcrd_retrieval_rows
            write_csv(self.out/f'queries_epoch_{row["epoch"]}.csv',per_query)
            from src.data_config import UNSEEN_CLASSES
            class_rows=[]
            for label in sorted(set(r['label'] for r in per_query)):
                selected=[r for r in per_query if r['label']==label]
                class_rows.append({'label':label,'category':UNSEEN_CLASSES[module.args.dataset][label],
                    'queries':len(selected),'mAP':sum(r['AP'] for r in selected)/len(selected),
                    'precision':sum(r['precision'] for r in selected)/len(selected)})
            write_csv(self.out/f'classes_epoch_{row["epoch"]}.csv',class_rows)
        if trainer.global_step and hasattr(self,'batch'):self.measure(trainer,module,'epoch_'+str(row['epoch']))
        self.figure()
        print(f'[AVCRD Diagnostics] epoch={row["epoch"]} mAP={row["mAP"]:.4f} P={row["precision"]:.4f} rank(photo/sketch)={row["photo_effective_rank"]:.1f}/{row["sketch_effective_rank"]:.1f}',flush=True)

    def measure(self,trainer,module,stage):
        from src.counterfactual_retrieval_kd import avcrd_loss
        # Lightning validation hooks run inside inference_mode. Merely enabling
        # grad would still yield empty epoch-end gradient probes. Disable it for
        # this independent differentiable forward, then restore the outer mode.
        with torch.inference_mode(False), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            batch=module.transfer_batch_to_device(self.batch,module.device,0)
            named=[(n,p) for n,p in module.named_parameters() if p.requires_grad and '_visual_prompt.' in n]
            params=[p for _,p in named]
            with torch.enable_grad():
                features=module(batch[:5]);main,_=loss_fn(module.args,features)
                cf=features[0].sum()*0
                stats={}
                if getattr(module,'lambda_avcrd',0)>0:
                    cf,stats,masked=module.counterfactual_loss(batch,features,return_masked=True)
                    cf=cf*module.lambda_avcrd
                def gradient(loss):
                    if not loss.requires_grad:return [torch.zeros_like(p,dtype=torch.float32) for p in params]
                    values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                    return [torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float() for p,g in zip(params,values)]
                gm,gc=gradient(main),gradient(cf)
                groups={'all':list(range(len(named))),
                        'photo':[i for i,(n,_) in enumerate(named) if 'photo_visual_prompt.' in n],
                        'sketch':[i for i,(n,_) in enumerate(named) if 'sketch_visual_prompt.' in n]}
                groups.update({name:[i] for i,(name,_) in enumerate(named)})
                for group,indices in groups.items():
                    self.gradients.append({'stage':stage,'global_step':int(trainer.global_step),'group':group,
                        'main_loss':float(main.detach()),'weighted_cf_loss':float(cf.detach()),
                        **_gradient_statistics([gm[i] for i in indices],[gc[i] for i in indices])})
            native={modality:features[i].detach().float().cpu() for i,modality in enumerate(('photo','sketch'))}
            if self.initial_native is None:self.initial_native={k:v.clone() for k,v in native.items()}
            record={'stage':stage,'global_step':int(trainer.global_step),'main_loss':float(main.detach()),'weighted_cf_loss':float(cf.detach())}
            record.update({k:float(v) for k,v in stats.items()})
            for mod,x in native.items():record[mod+'_initial_native_cosine']=F.cosine_similarity(x,self.initial_native[mod]).mean().item()
            self.fixed.append(record)
            # Examples are fixed by seed, not cherry-picked strongest responses.
            if getattr(module,'lambda_avcrd',0)>0:
                self.field_figure(module,batch,features,masked,stage)
            write_csv(self.out/'gradient_interaction.csv',self.gradients)
            write_csv(self.out/'fixed_batch.csv',self.fixed)
            print('[AVCRD Fixed Batch]',json.dumps(record,allow_nan=False),flush=True)

    def field_figure(self,module,batch,features,masked,stage):
        import matplotlib.pyplot as plt
        selected={'verified':0,'random':1,'attention_first':2,'random_first':3}[module.args.avcrd_selection]
        with torch.no_grad():
            sc=similarity_field(features[1],features[0]).cpu();sm=similarity_field(masked,features[0]).cpu()
            tc=similarity_field(batch[-1]['clean'],features[2]).cpu()
            tm=similarity_field(batch[-1]['masked'][:,selected],features[2]).cpu()
            sd=(sc-sm).numpy();actual_td=(tc-tm)
            supervised_td=actual_td.roll(1,dims=0) if module.args.avcrd_objective=='shuffled_effect' else actual_td
            td=supervised_td.numpy()
            rows=[{'query_position':i,'gallery_position':j,
                   'sketch_index':int(batch[-1]['sketch_index'][i]),
                   'teacher_actual_delta':float(actual_td[i,j]),
                   'teacher_supervised_delta':float(supervised_td[i,j]),
                   'student_delta':float(sd[i,j])}
                  for i in range(len(sd)) for j in range(sd.shape[1])]
            write_csv(self.out/('pair_effects_'+stage+'.csv'),rows)
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for axis,field,title in [(axes[0,0],tc.numpy(),'Teacher clean field'),(axes[0,1],sc.numpy(),'Student clean field'),
                                 (axes[1,0],td,'Teacher supervised delta'),(axes[1,1],sd,'Student counterfactual delta')]:
            lim=max(float(torch.stack([tc.abs().max(),sc.abs().max()]).max()),1e-6) if 'clean' in title else max(float(abs(td).max()),float(abs(sd).max()),1e-6)
            im=axis.imshow(field,cmap='RdBu_r',vmin=-lim,vmax=lim)
            axis.set(title=title,xlabel='Photo position',ylabel='Sketch position');fig.colorbar(im,ax=axis,shrink=.7)
        axes[0,2].scatter(td.flatten(),sd.flatten(),s=6,alpha=.3);axes[0,2].set(xlabel='Teacher delta',ylabel='Student delta',title='Per-pair response alignment')
        image=batch[1][0].detach().cpu();box=batch[-1]['boxes'][0,selected].cpu()
        view=erase_ink_box(image,box,module.args.avcrd_ink_threshold,module.args.avcrd_ink_softness)
        axes[1,2].imshow(np.concatenate([_rgb(image),_rgb(view)],axis=1));axes[1,2].axis('off');axes[1,2].set_title('Fixed sketch / selected ink erasure')
        fig.suptitle(stage+'; fixed seen batch; batch-sized gallery');fig.tight_layout()
        fig.savefig(self.out/('fields_'+stage+'.png'),dpi=130);plt.close(fig)

    def figure(self):
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        epochs=[r['epoch'] for r in self.epochs]
        for key in ('mAP','precision'):axes[0,0].plot(epochs,[100*r[key] for r in self.epochs],'o-',label=key)
        for mod in ('photo','sketch'):axes[0,1].plot(epochs,[r[mod+'_effective_rank'] for r in self.epochs],'o-',label=mod)
        steps=[r['global_step'] for r in self.fixed]
        for key in ('main_loss','weighted_cf_loss'):axes[0,2].plot(steps,[r[key] for r in self.fixed],label=key)
        for key in ('clean_cosine','effect_cosine'):
            if self.fixed and key in self.fixed[0]:axes[1,0].plot(steps,[r[key] for r in self.fixed],label=key)
        if self.fixed and 'effect_magnitude_ratio' in self.fixed[0]:axes[1,1].plot(steps,[r['effect_magnitude_ratio'] for r in self.fixed],label='student/teacher RMS')
        for group in ('all','photo','sketch'):
            rows=[r for r in self.gradients if r['group']==group and r['cosine'] is not None]
            if rows:axes[1,2].plot([r['global_step'] for r in rows],[r['cosine'] for r in rows],label=group)
        titles=('Full unseen retrieval (%)','Full unseen effective covariance rank','Fixed batch loss',
                'Fixed batch field cosine','Counterfactual response magnitude ratio','Main vs weighted AVCRD gradient cosine')
        for axis,title in zip(axes.flat,titles):
            axis.set_title(title);axis.grid(alpha=.2)
            if axis.lines:axis.legend(fontsize=8)
        fig.tight_layout();fig.savefig(self.out/'training_diagnostics.png',dpi=160);plt.close(fig)
        layers=list(dict.fromkeys(r['group'] for r in self.gradients if r['group'] not in ('all','photo','sketch')))
        stages=list(dict.fromkeys(r['stage'] for r in self.gradients))
        if layers:
            fig,axes=plt.subplots(1,2,figsize=(max(12,len(stages)*2),max(6,len(layers)*.27)))
            for axis,key,title in [(axes[0],'cosine','Main vs weighted AVCRD gradient cosine'),
                                   (axes[1],'cf_over_main','log10(weighted AVCRD / main gradient norm)')]:
                matrix=np.full((len(layers),len(stages)),np.nan)
                for i,layer in enumerate(layers):
                    for j,stage in enumerate(stages):
                        record=next(r for r in self.gradients if r['group']==layer and r['stage']==stage)
                        value=record[key]
                        if value is not None:matrix[i,j]=math.log10(max(value,1e-8)) if key=='cf_over_main' else value
                im=axis.imshow(matrix,cmap='RdBu_r' if key=='cosine' else 'viridis',aspect='auto',
                               **({'vmin':-1,'vmax':1} if key=='cosine' else {}))
                axis.set_xticks(range(len(stages)),stages,rotation=30,ha='right')
                axis.set_yticks(range(len(layers)),[x.replace('model.','') for x in layers],fontsize=7)
                axis.set_title(title+'; n/a if main is inactive');fig.colorbar(im,ax=axis,shrink=.6)
            fig.tight_layout();fig.savefig(self.out/'gradient_layers.png',dpi=160);plt.close(fig)

