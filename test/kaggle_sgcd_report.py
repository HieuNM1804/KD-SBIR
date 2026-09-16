"""Collect SGCD audit/training diagnostics for analysis; no checkpoints are copied."""
from datetime import datetime
from pathlib import Path
import csv
import json
import zipfile

PROJECT = Path('/kaggle/working/KD-SBIR-AVKD')
OUT = Path('/kaggle/working') / ('pcsgcd_diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUT.mkdir(parents=True, exist_ok=False)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def scalars(version):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(version), size_guidance={'scalars': 0})
    accumulator.Reload()
    return {tag: [{'step': value.step, 'value': value.value, 'wall_time': value.wall_time}
                  for value in accumulator.Scalars(tag)]
            for tag in accumulator.Tags().get('scalars', [])}


versions = []
rows = []
curves = []
for pattern in ('main_baseline_*', 'sgcd_*', 'pcsgcd_*'):
    for run in sorted((PROJECT / 'tb_logs').glob(pattern)):
        for version in sorted(run.glob('version_*')):
            diagnostic = version / 'sgcd_diagnostics'
            events = list(version.glob('events.out.tfevents.*'))
            if not events and not diagnostic.is_dir():
                continue
            versions.append(version)
            metrics = scalars(version) if events else {}
            m, p = metrics.get('mAP', []), metrics.get('precision', [])
            native_m, native_p = metrics.get('native_mAP', []), metrics.get('native_precision', [])
            if m and p:
                rows.append({
                    'run': run.name, 'version': version.name,
                    'final_mAP': m[-1]['value'], 'best_mAP': max(x['value'] for x in m),
                    'final_precision': p[-1]['value'], 'best_precision': max(x['value'] for x in p),
                    'final_native_mAP': native_m[-1]['value'] if native_m else None,
                    'final_native_precision': native_p[-1]['value'] if native_p else None,
                    'completed_validation_epochs': len(m),
                })
            for tag in ('mAP', 'precision', 'native_mAP', 'native_precision',
                        'SGCD', 'SGCD_WEIGHTED', 'SGCD_rank', 'SGCD_rank_margin',
                        'SGCD_rank_violation_rate', 'SGCD_teacher_selected_effect',
                        'main_loss', 'train_loss'):
                curves.extend({'run': run.name, 'version': version.name, 'tag': tag, **value}
                              for value in metrics.get(tag, []))
write_csv(OUT / 'comparison_summary.csv', rows)
write_csv(OUT / 'scalar_curves.csv', curves)

baseline_rows = [row for row in rows if row['run'].startswith('main_baseline_')]
delta_rows = []
if baseline_rows:
    baseline = sorted(baseline_rows, key=lambda row: (row['run'], row['version']))[-1]
    (OUT / 'baseline_reference.json').write_text(
        json.dumps(baseline, indent=2), encoding='utf-8'
    )
    for row in rows:
        delta_rows.append({
            **row,
            'baseline_run': baseline['run'],
            'delta_final_mAP': row['final_mAP'] - baseline['final_mAP'],
            'delta_final_precision': row['final_precision'] - baseline['final_precision'],
            'delta_best_mAP': row['best_mAP'] - baseline['best_mAP'],
            'delta_best_precision': row['best_precision'] - baseline['best_precision'],
        })
write_csv(OUT / 'comparison_deltas.csv', delta_rows)

checkpoint_rows = []
for run in sorted((PROJECT / 'saved_models').glob('*')):
    if not (run.name.startswith('main_baseline_') or run.name.startswith('sgcd_')
            or run.name.startswith('pcsgcd_')):
        continue
    for checkpoint in sorted(run.glob('*.ckpt')):
        checkpoint_rows.append({'run': run.name, 'checkpoint': checkpoint.name,
                                'path': str(checkpoint),
                                'size_MiB': checkpoint.stat().st_size / 1024**2})
write_csv(OUT / 'checkpoint_inventory.csv', checkpoint_rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if rows:
    ordered = sorted(rows, key=lambda row: row['final_mAP'])
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(ordered) * .45)))
    names = [row['run'] for row in ordered]
    y = list(range(len(ordered)))
    axes[0].barh(y, [100 * row['final_mAP'] for row in ordered])
    axes[0].set_yticks(y, names, fontsize=8); axes[0].set_xlabel('Final full unseen mAP (%)')
    axes[1].barh(y, [100 * row['final_precision'] for row in ordered])
    axes[1].set_yticks(y, []); axes[1].set_xlabel('Final P@100 (%)')
    fig.tight_layout(); fig.savefig(OUT / 'comparison.png', dpi=160); plt.close(fig)
if delta_rows:
    ordered = sorted(delta_rows, key=lambda row: row['delta_final_mAP'])
    fig, axis = plt.subplots(figsize=(12, max(5, len(ordered) * .42)))
    names = [row['run'] for row in ordered]
    values = [100 * row['delta_final_mAP'] for row in ordered]
    colors = ['#2ca02c' if value >= 0 else '#d62728' for value in values]
    axis.barh(range(len(ordered)), values, color=colors)
    axis.set_yticks(range(len(ordered)), names, fontsize=8)
    axis.axvline(0, color='black', linewidth=.8)
    axis.set_xlabel('Final mAP change from latest matched baseline (percentage points)')
    fig.tight_layout(); fig.savefig(OUT / 'comparison_deltas.png', dpi=160); plt.close(fig)

manifest = {
    'created': datetime.now().isoformat(), 'project': str(PROJECT),
    'included_versions': [str(path) for path in versions],
    'runs_with_metrics': len(rows),
    'notes': [
        'Pairwise teacher audit files are included even when the gate stops before training.',
        'Full unseen retrieval metrics and fixed seen-batch diagnostics have different scope.',
        'No model checkpoint, teacher cache or target tensor is copied.',
    ],
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
archive = OUT.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob('*')):
        if path.is_file(): bundle.write(path, Path('report') / path.relative_to(OUT))
    for version in versions:
        diagnostic = version / 'sgcd_diagnostics'
        if diagnostic.is_dir():
            for path in sorted(diagnostic.rglob('*')):
                if path.is_file():
                    bundle.write(path, Path('tb_logs') / version.parent.name / version.name /
                                 'sgcd_diagnostics' / path.relative_to(diagnostic))

from IPython.display import FileLink, Image, display
for name in ('comparison.png', 'comparison_deltas.png'):
    if (OUT / name).is_file(): display(Image(filename=str(OUT / name)))
for version in versions:
    for name in ('teacher_audit.png', 'teacher_audit_examples.png',
                 'teacher_stroke_graph_examples.png', 'training_diagnostics.png'):
        image = version / 'sgcd_diagnostics' / name
        if image.is_file(): display(Image(filename=str(image)))
print('Send this diagnostics ZIP:', archive)
display(FileLink(str(archive)))
