"""Paste the whole file into one Kaggle cell: fresh student, sketch-only cosine AV."""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import zipfile

PROJECT = Path('/kaggle/working/KD-SBIR-AVKD')
RUN = 'av_sketch_only_s42_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT = Path('/kaggle/working') / RUN

RUNNER_SOURCE = r'''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import ast
import hashlib
import inspect
import json
import runpy
import sys
import textwrap


def sketch_only_step(original, namespace):
    # Replace just the AV loss assignment; preserve the production main losses,
    # forward/capture, logging and global-feature ablation behavior.
    tree = ast.parse(textwrap.dedent(original))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'av_loss' for t in n.targets)]
    if len(matches) != 1:
        raise ValueError('Expected one AV loss assignment in production training_step')
    matches[0].value = ast.parse(
        '0.5 * feature_cosine_kd(capture.values[1], batch[6], self.model.av_projector)',
        mode='eval').body
    ast.fix_missing_locations(tree)
    scope = dict(namespace)
    exec(compile(tree, '<sketch_only_training_step>', 'exec'), scope)
    return scope['training_step'], ast.unparse(tree)


def main():
    project, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(project))
    import torch
    import src.model as model_source
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU')
    teacher = Path('/kaggle/working/teacher_cache/sketchy1_teacher_1ep.pt')
    av = Path('/kaggle/working/teacher_cache/sketchy1_av_output.pt')
    for p in (teacher, av, Path('/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy/sketch'),
              Path('/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy/photo')):
        if not p.exists():
            raise FileNotFoundError(f'Missing existing input: {p}')
    source_files = ('src/model.py', 'src/train.py', 'src/losses.py', 'src/dataset.py',
                    'src/attention_output_kd.py', 'src/attention_output_cache.py',
                    'src/teacher_prompts.py', 'clip/model.py')
    hashes = {}
    for n in source_files:
        text = (project/n).read_text(encoding='utf-8')
        target = out/'source'/n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        hashes[n] = hashlib.sha256(text.encode()).hexdigest()
    cls = model_source.ZS_SBIR
    changed, source = sketch_only_step(inspect.getsource(cls.training_step), vars(model_source))
    effective_hash = hashlib.sha256(source.encode()).hexdigest()
    (out/'effective_training_step.py').write_text(source+'\n', encoding='utf-8')
    cls.training_step = changed
    original_save = cls.on_save_checkpoint
    def save_metadata(self, checkpoint):
        original_save(self, checkpoint)
        args = dict(vars(self.args))
        args.update(av_objective='cosine', av_modality='sketch_only', av_sketch_factor=0.5)
        checkpoint['experiment_config'] = {
            'args': args, 'av_objective': 'cosine', 'av_modality': 'sketch_only',
            'loss_formula': '3*domain + photo_text + sketch_text + 0.5*lambda_av*cosine(projector(AV_sketch),teacher_AV_sketch)',
            'av_target_metadata': getattr(self.args, 'av_target_metadata', cache_metadata),
            'source_sha256': hashes,
            'effective_training_step_sha256': effective_hash,
        }
        checkpoint.setdefault('hyper_parameters', {}).update(
            args=args, classnames=list(self.model.classnames))
    # Older bundle code does not copy target metadata into args.
    payload = torch.load(av, map_location='cpu', weights_only=True)
    cache_metadata = payload['metadata']
    del payload
    cls.on_save_checkpoint = save_metadata
    argv = [
        '--root', '/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy',
        '--dataset', 'sketchy_1', '--epochs', '5', '--workers', '8',
        '--batch_size', '64', '--test_batch_size', '1024',
        '--n_ctx_visual', '3', '--prompt_depth', '12',
        '--teacher_pretrain_epochs', '1', '--teacher_cache_path', str(teacher),
        '--teacher_pretrain_batch_size', '64', '--teacher_n_ctx_visual', '10',
        '--teacher_prompt_depth', '12', '--teacher_prompt_std', '0.02',
        '--teacher_prompt_lr', '3e-2', '--teacher_prompt_seed', '42',
        '--teacher_prompt_gradient_checkpointing', '--teacher_momentum', '0.9',
        '--teacher_weight_decay', '1e-3', '--lambda_teacher_retrieval', '1.5',
        '--teacher_triplet_margin', '0.2', '--lambda_domain', '3', '--lambda_modality', '1',
        '--photo_text_kd_temperature', '0.15', '--sketch_text_kd_temperature', '0.02',
        '--lambda_av', '1', '--lambda_global_feature', '0', '--av_cache_path', str(av),
        '--av_teacher_batch_size', '8', '--lr', '1e-2', '--momentum', '0.9',
        '--weight_decay', '5e-4', '--seed', '42', '--exp_name', name, '--ckpt_path', '', '--progress',
    ]
    # ca92ba2 only implements cosine; later versions expose its explicit flag.
    train_text = (project/'src/train.py').read_text(encoding='utf-8')
    if '--av_objective' in train_text:
        argv.extend(['--av_objective', 'cosine'])
    manifest = {'run': name, 'arguments': argv, 'source_sha256': hashes,
                'av_objective': 'cosine', 'av_modality': 'sketch_only', 'sketch_factor': 0.5,
                'student_initialization': 'Pretrained CLIP + freshly seeded prompts/projector; no resume',
                'teacher_cache': str(teacher), 'av_cache': str(av),
                'av_target_metadata': cache_metadata, 'completed': False,
                'effective_training_step': source,
                'effective_training_step_sha256': effective_hash}
    (out/'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('[Sketch-only AV] fresh student; domain=3, modality=1, sketch AV=0.5, photo AV=0.', flush=True)
    sys.argv = ['src.train'] + argv
    namespace = runpy.run_module('src.train', run_name='__main__')
    trainer = namespace['trainer']
    final = project/'saved_models'/name/'final.ckpt'
    trainer.save_checkpoint(str(final))
    saved = torch.load(final, map_location='cpu', weights_only=False)
    if saved['global_step'] != trainer.global_step or saved['experiment_config']['av_modality'] != 'sketch_only':
        raise RuntimeError('Final checkpoint verification failed')
    manifest.update(completed=True, final_checkpoint=str(final), global_step=saved['global_step'],
                    final_epoch=saved['epoch'], best_checkpoint=trainer.checkpoint_callback.best_model_path,
                    best_P100=float(trainer.checkpoint_callback.best_model_score),
                    lightning_version=saved.get('pytorch-lightning_version'))
    (out/'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('[Sketch-only AV] final checkpoint verified:', final, 'step=', saved['global_step'], flush=True)


if __name__ == '__main__':
    main()
'''


def collect_scalars(root):
    import struct
    from tensorboard.compat.proto.event_pb2 import Event
    metrics = {}
    for path in sorted(root.rglob('events.out.tfevents.*')):
        with path.open('rb') as stream:
            while True:
                header = stream.read(12)
                if len(header) < 12:
                    break
                length = struct.unpack('<Q', header[:8])[0]
                if length > 64 * 1024**2:
                    raise ValueError(f'Invalid event record: {path}')
                data, checksum = stream.read(length), stream.read(4)
                if len(data) != length or len(checksum) != 4:
                    break
                event = Event.FromString(data)
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        metrics.setdefault(value.tag, []).append({'step': event.step, 'value': value.simple_value})
    return metrics


if __name__ == '__main__':
    if not (PROJECT/'src/train.py').is_file():
        raise FileNotFoundError(f'Run offline setup first: {PROJECT}')
    OUT.mkdir(parents=True, exist_ok=False)
    runner = OUT/'train_sketch_only.py'
    compile(RUNNER_SOURCE, str(runner), 'exec')
    runner.write_text(RUNNER_SOURCE, encoding='utf-8')
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['MPLBACKEND'] = 'Agg'
    # Direct output as in main; no line-by-line replay of tqdm.
    subprocess.run([sys.executable, '-u', str(runner), str(PROJECT), str(OUT), RUN],
                   cwd=PROJECT, env=env, check=True)
    metrics = collect_scalars(PROJECT/'tb_logs'/RUN)
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag in ('mAP', 'precision'):
        values = metrics.get(tag, [])
        if values:
            axes[0].plot(range(1, len(values)+1), [100*r['value'] for r in values], 'o-', label=tag)
    for tag in ('AV_KD', 'train_loss'):
        values = metrics.get(tag, [])
        if values:
            axes[1].plot(range(1, len(values)+1), [r['value'] for r in values], 'o-', label=tag)
    axes[0].set(title='Sketch-only cosine AV: full validation', xlabel='Completed epoch', ylabel='%')
    axes[1].set(title='AV_KD includes the sketch factor 0.5', xlabel='Completed epoch', ylabel='Loss')
    for ax in axes:
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'training_curves.png', dpi=160); plt.close(fig)
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(OUT))
    from IPython.display import FileLink, Image, display
    display(Image(filename=str(OUT/'training_curves.png')))
    display(FileLink(str(archive)))
    print('Send this ZIP:', archive)
    print('Checkpoints:', PROJECT/'saved_models'/RUN)
