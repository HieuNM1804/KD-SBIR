"""Persistent epoch measurements; all validation metrics use full galleries."""
import csv
import json
from pathlib import Path

from pytorch_lightning import Callback
from pytorch_lightning.trainer.states import TrainerFn


def write_rows(path, rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


class MaskDiagnostics(Callback):
    def __init__(self, directory):
        super().__init__()
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.rows=[]

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking or not hasattr(module,'mask_validation'):return
        if trainer.state.fn == TrainerFn.VALIDATING:
            row=dict(stage='initial_full_validation',**module.mask_validation)
            (self.directory/'initial_validation.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            print('[Mask Initial Full Validation]',row,flush=True)

    def on_train_epoch_end(self, trainer, module):
        n=module.mask_epoch_samples
        if not n:return
        row={'epoch':int(module.current_epoch),'global_step':trainer.global_step,
             'strategy':module.args.mask_strategy,'lambda_response':module.args.lambda_response,
             'lr':trainer.optimizers[0].param_groups[0]['lr'],
             **{k:v/n for k,v in module.mask_epoch_sums.items()},**getattr(module,'mask_validation',{})}
        self.rows.append(row)
        write_rows(self.directory/'epochs.csv',self.rows)
        write_rows(self.directory/'gradient_audit.csv',module.mask_gradient_rows)
        print('[Mask Epoch]',json.dumps(row),flush=True)
        self.plot()

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        epoch=[r['epoch']+1 for r in self.rows]
        for key in ('embedding_loss','response_loss'):
            axes[0].plot(epoch,[r[key] for r in self.rows],'o-',label=key)
        for key in ('mAP','precision'):
            if key in self.rows[0]:axes[1].plot(epoch,[100*r[key] for r in self.rows],'o-',label=key)
        for key in ('photo_teacher_delta','photo_student_delta','sketch_teacher_delta','sketch_student_delta'):
            if key in self.rows[0]:axes[2].plot(epoch,[r[key] for r in self.rows],'o-',label=key)
        for axis,title in zip(axes,('Unweighted train objectives','Full validation (%)','Mean embedding response norm')):
            axis.set(title=title,xlabel='Completed epoch');axis.grid(alpha=.2)
            if axis.lines:axis.legend(fontsize=7)
        fig.tight_layout();fig.savefig(self.directory/'training_diagnostics.png',dpi=140);plt.close(fig)

    def on_fit_end(self, trainer, module):
        (self.directory/'completion.json').write_text(json.dumps({'fit_completed':True,
            'global_step':trainer.global_step,'epochs_logged':len(self.rows),
            'notes':['Loss averages and validation metrics correspond to the same epoch.',
                     'Gradient audit uses the first training batch each epoch; it is not a full-dataset estimate.',
                     'Gradient norms/cosines are for unweighted embedding and response losses.',
                     'Teacher/student response stats are on seen training batches, not unseen retrieval performance.']},indent=2),encoding='utf-8')
