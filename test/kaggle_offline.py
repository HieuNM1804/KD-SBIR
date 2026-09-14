"""Restore pinned correspondence source/weights and run deterministic GPU tests."""
from datetime import datetime
from pathlib import Path
from importlib import metadata
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

WORKING = Path('/kaggle/working')
PROJECT = WORKING / 'KD-SBIR'
EXPECTED_BRANCH = 'experiment/sketch-region-correspondence-kd'
EXPECTED_COMMIT = 'SOURCE_COMMIT_PENDING'
EXPECTED_TASK = 'sketch_region_correspondence_kd'
EXPECTED_ENTRYPOINT = 'src.train_correspondence'
DFN_REPO = 'apple/DFN5B-CLIP-ViT-H-14'
DFN_REVISION = '11738501a1db6d5e0a3451a71ba100be02e577e6'
DFN_FILENAME = 'open_clip_pytorch_model.bin'
DFN_SHA256 = 'd67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae'
STUDENT_FILENAME = 'ViT-B-32.pt'
STUDENT_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
matches = []
reports = []
for depth in range(1, 6):
    for path in glob.glob('/kaggle/input/' + '*/' * depth + 'correspondence_bundle/bundle_manifest.json'):
        candidate = json.loads(Path(path).read_text(encoding='utf-8'))
        reports.append((path, candidate.get('branch'), candidate.get('commit')))
        if (candidate.get('branch') == EXPECTED_BRANCH and candidate.get('commit') == EXPECTED_COMMIT
                and candidate.get('task') == EXPECTED_TASK and candidate.get('entrypoint') == EXPECTED_ENTRYPOINT):
            matches.append((Path(path).parent, candidate))
if len(matches) != 1:
    raise RuntimeError(f'Attach exactly one matching correspondence_bundle; matches={len(matches)}, inspected={reports}')
bundle, manifest = matches[0]
if list(sys.version_info[:2]) != manifest['python_minor']:
    raise RuntimeError('Online/offline Python minor versions differ; rebuild wheels in the matching Kaggle image')
if (manifest['teacher_repo'] != DFN_REPO or manifest['teacher_revision'] != DFN_REVISION
        or manifest['teacher_filename'] != DFN_FILENAME or manifest['student_filename'] != STUDENT_FILENAME):
    raise ValueError('Unexpected teacher/student checkpoint definitions')
requirements = bundle / 'requirements.txt'
if sha256(requirements) != manifest['requirements_sha256']:
    raise RuntimeError('Requirements hash mismatch')
for name, digest in manifest['wheel_sha256'].items():
    if Path(name).name != name or not name.endswith('.whl') or sha256(bundle / 'wheels' / name) != digest:
        raise RuntimeError('Invalid/mismatched wheel: ' + name)
if not manifest['wheel_sha256']:
    raise RuntimeError('No dependency wheels attached')
teacher = bundle / 'dfn5b_openclip' / DFN_FILENAME
student = bundle / 'clip_cache' / STUDENT_FILENAME
for path, digest, size in ((teacher, DFN_SHA256, manifest['teacher_size']),
                           (student, STUDENT_SHA256, manifest['student_size'])):
    if path.stat().st_size != size or sha256(path) != digest:
        raise RuntimeError('Checkpoint size/SHA256 mismatch: ' + str(path))
source = bundle / 'source' / 'KD-SBIR'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
if commit != EXPECTED_COMMIT:
    raise RuntimeError('Attached source commit mismatch')
for name, digest in manifest['source_sha256'].items():
    if '..' in Path(name).parts or Path(name).is_absolute() or sha256(source / name) != digest:
        raise RuntimeError('Source hash mismatch: ' + name)

stack = {name: metadata.version(name) for name in ('torch', 'torchvision')}
constraint = WORKING / 'correspondence_stack_constraints.txt'
constraint.write_text('\n'.join(f'{name}=={version}' for name, version in stack.items()) + '\n', encoding='utf-8')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(bundle / 'wheels'),
                '-r', str(requirements), '-c', str(constraint)], check=True)
if stack != {name: metadata.version(name) for name in stack}:
    raise RuntimeError('Kaggle CUDA stack changed')

clip_cache = Path.home() / '.cache' / 'clip'
clip_cache.mkdir(parents=True, exist_ok=True)
shutil.copy2(student, clip_cache / STUDENT_FILENAME)
hf_home = WORKING / 'huggingface'
repo_cache = hf_home / 'hub' / ('models--' + DFN_REPO.replace('/', '--'))
snapshot = repo_cache / 'snapshots' / DFN_REVISION
snapshot.mkdir(parents=True, exist_ok=True)
shutil.copy2(teacher, snapshot / DFN_FILENAME)
(repo_cache / 'refs').mkdir(parents=True, exist_ok=True)
(repo_cache / 'refs' / 'main').write_text(DFN_REVISION, encoding='utf-8')
os.environ.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home / 'hub'), HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
if PROJECT.exists():
    if PROJECT.resolve().parent != WORKING.resolve():
        raise ValueError('Unexpected working project path; refusing to move it')
    backup = WORKING / ('KD-SBIR_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    PROJECT.rename(backup)
    print('Previous project/checkpoints retained:', backup)
shutil.copytree(source, PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(), 'Enable Kaggle GPU'; print('GPU:', torch.cuda.get_device_name(0))"], check=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_correspondence*.py', '-v'], check=True)
subprocess.run([sys.executable, '-m', EXPECTED_ENTRYPOINT, '--help'], check=True, stdout=subprocess.DEVNULL)
print('CORRESPONDENCE SETUP COMPLETE:', PROJECT)
print('Branch:', EXPECTED_BRANCH, 'source:', EXPECTED_COMMIT)
print('Run the preparation cell, then GT/global/teacher controls from docs/region_correspondence.md.')
print('Category root: /kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy')
print('FG root: /kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg (must directly contain sketch/ and photo/)')
