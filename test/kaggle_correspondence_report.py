"""Print matched run summaries and ZIP diagnostics (checkpoints excluded)."""
from datetime import datetime
from pathlib import Path
import json
import zipfile

root = Path('/kaggle/working/correspondence_runs')
rows = []
for path in sorted((root / 'saved_models').glob('*/run.json')):
    run = json.loads(path.read_text(encoding='utf-8'))
    unseen = run.get('unseen_deployed') or {}
    row = {'run': path.parent.name, 'mode': run['args']['correspondence_mode'],
           'protocol': run['args']['retrieval_protocol'], 'seed': run['args']['seed'],
           'seen_primary': run['best_seen_primary'],
           'unseen_mAP': unseen.get('mAP'), 'unseen_Acc1': unseen.get('Acc1'),
           'unseen_Acc5': unseen.get('Acc5'), 'gallery_scope': unseen.get('gallery_scope'),
           'gallery_candidates': unseen.get('gallery_candidates_max')}
    rows.append(row)
    print(json.dumps(row))
destination = root / ('correspondence_diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('summary.json', json.dumps(rows, indent=2))
    for folder in ('saved_models', 'tb_logs', 'correspondence_cache'):
        for path in (root / folder).rglob('*'):
            if path.is_file() and (path.suffix in ('.json', '.csv') or path.name.startswith('events.out')):
                archive.write(path, path.relative_to(root).as_posix())
print('Download diagnostics:', destination)
print('Download best/last/final .ckpt files separately; no weights/data included in this ZIP.')
