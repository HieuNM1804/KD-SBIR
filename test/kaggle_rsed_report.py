from datetime import datetime
from pathlib import Path
import csv
import json
import zipfile

PROJECT = Path('/kaggle/working/KD-SBIR-AVKD')
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT = Path('/kaggle/working') / ('rsed_diagnostics_' + STAMP)
OUT.mkdir(parents=True, exist_ok=False)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def event_scalars(version):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    result = {}
    accumulator = EventAccumulator(str(version), size_guidance={'scalars': 0})
    accumulator.Reload()
    for tag in accumulator.Tags().get('scalars', []):
        result[tag] = [{'step': item.step, 'value': item.value, 'wall_time': item.wall_time}
                       for item in accumulator.Scalars(tag)]
    return result


rows = []
curves = []
versions = []
for pattern in ('main_baseline_*', 'rsed_*'):
    for run in sorted((PROJECT / 'tb_logs').glob(pattern)):
        for version in sorted(run.glob('version_*')):
            if not list(version.glob('events.out.tfevents.*')):
                continue
            versions.append(version)
            metrics = event_scalars(version)
            m = metrics.get('mAP', [])
            p = metrics.get('precision', [])
            native_m = metrics.get('native_mAP', [])
            native_p = metrics.get('native_precision', [])
            if not m or not p:
                continue
            row = {
                'run': run.name,
                'version': version.name,
                'final_mAP': m[-1]['value'],
                'best_mAP': max(item['value'] for item in m),
                'final_precision': p[-1]['value'],
                'best_precision': max(item['value'] for item in p),
                'final_native_mAP': native_m[-1]['value'] if native_m else None,
                'final_native_precision': native_p[-1]['value'] if native_p else None,
                'completed_validation_epochs': len(m),
            }
            rows.append(row)
            for tag, values in metrics.items():
                if tag in ('mAP', 'precision', 'native_mAP', 'native_precision',
                           'RSED', 'RSED_WEIGHTED', 'main_loss', 'train_loss'):
                    curves.extend({'run': run.name, 'version': version.name, 'tag': tag, **item}
                                  for item in values)
write_csv(OUT / 'comparison_summary.csv', rows)
write_csv(OUT / 'scalar_curves.csv', curves)

checkpoint_rows = []
for run in sorted((PROJECT / 'saved_models').glob('*')):
    if not (run.name.startswith('main_baseline_') or run.name.startswith('rsed_')):
        continue
    for path in sorted(run.glob('*.ckpt')):
        checkpoint_rows.append({'run': run.name, 'checkpoint': path.name,
                                'path': str(path), 'size_MiB': path.stat().st_size / 1024**2})
write_csv(OUT / 'checkpoint_inventory.csv', checkpoint_rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if rows:
    ordered = sorted(rows, key=lambda row: row['final_mAP'])
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(ordered) * .45)))
    names = [row['run'] for row in ordered]
    y = range(len(ordered))
    axes[0].barh(list(y), [100 * row['final_mAP'] for row in ordered])
    axes[0].set_yticks(list(y), names, fontsize=8)
    axes[0].set_xlabel('Final full unseen mAP (%)')
    axes[1].barh(list(y), [100 * row['final_precision'] for row in ordered])
    axes[1].set_yticks(list(y), [])
    axes[1].set_xlabel('Final P@100 (%)')
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)

manifest = {
    'created': datetime.now().isoformat(),
    'project': str(PROJECT),
    'included_versions': [str(path) for path in versions],
    'runs_with_metrics': len(rows),
    'notes': [
        'Metrics are full unseen Sketchy-1 retrieval values logged by the training code.',
        'RSED fixed-batch diagnostics are seen-training measurements and are not validation metrics.',
        'No model checkpoint or teacher tensor is copied into this archive.',
        'Compare matched main and RSED at the same seed before interpreting controls.',
    ],
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

archive = OUT.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            bundle.write(path, Path('report') / path.relative_to(OUT))
    for version in versions:
        diagnostic = version / 'rsed_diagnostics'
        if diagnostic.is_dir():
            for path in sorted(diagnostic.rglob('*')):
                if path.is_file():
                    bundle.write(path, Path('tb_logs') / version.parent.name / version.name /
                                 'rsed_diagnostics' / path.relative_to(diagnostic))

from IPython.display import FileLink, Image, display
if (OUT / 'comparison.png').is_file():
    display(Image(filename=str(OUT / 'comparison.png')))
for version in versions:
    image = version / 'rsed_diagnostics' / 'training_diagnostics.png'
    if image.is_file():
        display(Image(filename=str(image)))
print('Send this ZIP:', archive)
display(FileLink(str(archive)))
