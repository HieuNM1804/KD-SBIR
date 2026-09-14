"""Paste this entire file into one Kaggle cell after AVCRD experiments.

Exports diagnostics/figures and a comparison table. Does not copy checkpoints,
model weights, image caches, or teacher counterfactual tensors into the ZIP.
"""
from datetime import datetime
from pathlib import Path
import csv
import json
import zipfile

PROJECT=Path('/kaggle/working/KD-SBIR-AVKD')
# Optionally list exact run names to restrict the report. Empty includes avcrd_*.
RUNS=[]

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import FileLink,Image,display
    roots=[]
    for root in sorted((PROJECT/'tb_logs').glob('avcrd_*/version_*')):
        if RUNS and root.parent.name not in RUNS:continue
        if (root/'avcrd_diagnostics').is_dir() or (root/'avcrd_teacher_probe').is_dir():roots.append(root)
    if not roots:raise FileNotFoundError('No AVCRD logs found in the current project. Set PROJECT to the project used for training.')
    OUT=Path('/kaggle/working')/('avcrd_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    OUT.mkdir(parents=True,exist_ok=False)
    summary=[];manifest={'created':datetime.now().isoformat(),'project':str(PROJECT),'runs':[],
        'notes':['Unseen retrieval uses the full original validation gallery and native global CLIP descriptors.',
                 'Best row is selected by P@K, matching the existing main checkpoint policy; the final epoch is also exported.',
                 'The existing main teacher/student checkpoint policy uses unseen labels for model selection.',
                 'Fixed-batch influence/gradient diagnostics use seen training data, not the unseen set.',
                 'Lower response loss is not a guarantee of higher retrieval metrics. Compare matched controls.']}
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            run=root.parent.name;version=root.name;record={'run':run,'version':version}
            for name in ('avcrd_diagnostics','avcrd_teacher_probe'):
                directory=root/name
                if not directory.is_dir():continue
                for path in sorted(directory.rglob('*')):
                    if path.is_file():archive.write(path,path.relative_to(PROJECT/'tb_logs'))
            path=root/'avcrd_diagnostics/epochs.csv'
            if path.is_file():
                rows=read_csv(path);trained=[r for r in rows if int(r['global_step'])>0]
                if trained:
                    best=max(trained,key=lambda r:float(r['precision']));final=trained[-1]
                    initial=next((r for r in rows if int(r['global_step'])==0),None)
                    item={'run':run,'version':version,'epochs':len(trained),'selected_epoch':int(best['epoch']),
                          'selected_mAP':float(best['mAP']),'selected_precision':float(best['precision']),
                          'final_mAP':float(final['mAP']),'final_precision':float(final['precision']),
                          'initial_mAP':float(initial['mAP']) if initial else None,
                          'initial_precision':float(initial['precision']) if initial else None,
                          'final_photo_effective_rank':float(final['photo_effective_rank']),
                          'final_sketch_effective_rank':float(final['sketch_effective_rank']),
                          'train_validation_seconds':sum(float(r['train_plus_validation_seconds']) for r in trained)}
                    config_path=root/'avcrd_diagnostics/configuration.json'
                    if config_path.is_file():
                        config=json.loads(config_path.read_text(encoding='utf-8'))
                        for key in ('seed','lambda_domain','lambda_modality','lambda_avcrd','avcrd_clean_weight',
                                    'avcrd_effect_weight','avcrd_selection','avcrd_objective'):
                            item[key]=config.get(key)
                    summary.append(item);record['metrics']=item
                    curve=root/'avcrd_diagnostics/training_diagnostics.png'
                    if curve.is_file():display(Image(filename=str(curve)))
            path=root/'avcrd_teacher_probe/teacher_probe.json'
            if path.is_file():
                probe=json.loads(path.read_text(encoding='utf-8'));record['teacher_probe']={k:v for k,v in probe.items() if k!='metadata'}
                if not (root/'avcrd_diagnostics/epochs.csv').is_file():
                    for name in ('teacher_effects.png','teacher_interventions.png'):
                        figure=root/'avcrd_teacher_probe'/name
                        if figure.is_file():display(Image(filename=str(figure)))
            manifest['runs'].append(record)
        if summary:
            path=OUT/'comparison_summary.csv'
            with path.open('w',newline='',encoding='utf-8') as stream:
                fields=list(dict.fromkeys(k for r in summary for k in r))
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(summary)
            archive.write(path,path.name)
            fig,axes=plt.subplots(1,2,figsize=(17,max(4,len(summary)*.6)))
            labels=[r['run']+' / '+r['version'] for r in summary]
            y=list(range(len(summary)))
            axes[0].barh([i-.17 for i in y],[100*r['selected_mAP'] for r in summary],height=.3,label='Selected mAP')
            axes[0].barh([i+.17 for i in y],[100*r['final_mAP'] for r in summary],height=.3,label='Final mAP')
            axes[0].set_yticks(y,labels,fontsize=7);axes[0].invert_yaxis();axes[0].set_xlabel('Full unseen mAP (%)');axes[0].legend()
            for mod,offset in [('photo',-.17),('sketch',.17)]:
                axes[1].barh([i+offset for i in y],[r['final_'+mod+'_effective_rank'] for r in summary],height=.3,label=mod)
            axes[1].invert_yaxis();axes[1].set_yticks(y,[]);axes[1].set_xlabel('Final full unseen covariance effective rank');axes[1].legend()
            fig.tight_layout();path=OUT/'comparison.png';fig.savefig(path,dpi=160);plt.close(fig)
            archive.write(path,path.name);display(Image(filename=str(path)))
        path=OUT/'manifest.json';path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8');archive.write(path,path.name)
    print('Send this ZIP:',OUT.with_suffix('.zip'))
    print('Checkpoints remain in:',PROJECT/'saved_models')
    display(FileLink(str(OUT.with_suffix('.zip'))))
