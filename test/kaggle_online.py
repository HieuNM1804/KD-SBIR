"""Build a pinned correspondence bundle; paste into Internet-enabled Kaggle.

An uploaded sketch-region-correspondence-source.bundle takes precedence over
GitHub. This makes the local, not-yet-pushed branch runnable on Kaggle.
"""
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

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
WORKING = Path('/kaggle/working')
BUNDLE = WORKING / 'correspondence_bundle'
BRANCH = 'experiment/sketch-region-correspondence-kd'
COMMIT = 'SOURCE_COMMIT_PENDING'
REPO = 'https://github.com/HieuNM1804/KD-SBIR.git'
TASK = 'sketch_region_correspondence_kd'
ENTRYPOINT = 'src.train_correspondence'
REQUIREMENTS = ('open-clip-torch==3.2.0', 'pytorch-lightning==2.6.0', 'torchmetrics==1.8.2',
                'lightning-utilities==0.15.2', 'huggingface-hub==0.36.2', 'ftfy', 'regex',
                'tensorboard', 'packaging', 'tqdm', 'numpy==2.2.6', 'pillow==10.4.0')
DFN_REPO = 'apple/DFN5B-CLIP-ViT-H-14'
DFN_REVISION = '11738501a1db6d5e0a3451a71ba100be02e577e6'
DFN_FILENAME = 'open_clip_pytorch_model.bin'
DFN_SHA256 = 'd67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae'
STUDENT_FILENAME = 'ViT-B-32.pt'
STUDENT_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'


def run(command, cwd=WORKING):
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def installed_dependency_closure(requirements):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    excluded = {'torch', 'torchvision', 'torchaudio', 'triton'}
    pending = [Requirement(item) for item in requirements]
    processed, pinned = set(), {}
    while pending:
        req = pending.pop()
        name = canonicalize_name(req.name)
        if name in excluded or name.startswith(('nvidia-', 'cuda-')):
            continue
        key = (name, tuple(sorted(req.extras)))
        if key in processed:
            continue
        processed.add(key)
        dist = metadata.distribution(name)
        if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            raise RuntimeError(f'Installed dependency does not satisfy {req}: {dist.version}')
        pinned[name] = dist.version
        for item in dist.requires or ():
            child = Requirement(item)
            if child.marker is None or any(child.marker.evaluate({'extra': extra}) for extra in ('', *req.extras)):
                pending.append(child)
    return pinned


WORKING.mkdir(parents=True, exist_ok=True)
if BUNDLE.exists():
    if BUNDLE.resolve().parent != WORKING.resolve():
        raise ValueError('Unexpected bundle target path')
    backup = WORKING / ('correspondence_bundle_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    BUNDLE.rename(backup)
    print('Previous bundle retained:', backup)
for directory in ('wheels', 'source', 'clip_cache', 'dfn5b_openclip'):
    (BUNDLE / directory).mkdir(parents=True)
requirements = BUNDLE / 'requirements.txt'
requirements.write_text('\n'.join(REQUIREMENTS) + '\n', encoding='utf-8')

# Respect the installed Kaggle CUDA stack; fail rather than replace it.
stack = {}
for name in ('torch', 'torchvision', 'torchaudio', 'triton'):
    try:
        stack[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
if 'torch' not in stack or 'torchvision' not in stack:
    raise RuntimeError('Use a Kaggle image with its preinstalled torch/torchvision stack')
constraints = BUNDLE / 'online_stack_constraints.txt'
constraints.write_text('\n'.join(f'{name}=={version}' for name, version in stack.items()) + '\n', encoding='utf-8')
run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements), '-c', str(constraints)])
assert stack == {name: metadata.version(name) for name in stack}, 'CUDA stack changed'

# Download the installed dependency closure WITHOUT ever downloading a second
# torch/CUDA stack. Extras (e.g. fsspec[http]) are propagated through metadata.
pinned = installed_dependency_closure(REQUIREMENTS)
closure = BUNDLE / 'wheel_closure.txt'
closure.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(pinned.items())) + '\n', encoding='utf-8')
run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
     '--dest', str(BUNDLE / 'wheels'), '-r', str(closure)])
print('[1/4] Pinned dependency wheels:', len(list((BUNDLE / 'wheels').glob('*.whl'))))

source_bundles = []
for depth in range(1, 6):
    source_bundles.extend(Path(p) for p in glob.glob('/kaggle/input/' + '*/' * depth + 'sketch-region-correspondence-source.bundle'))
source_bundles = sorted(set(source_bundles))
if len(source_bundles) > 1:
    raise RuntimeError('Attach exactly one correspondence source.bundle; multiple versions found')
project = BUNDLE / 'source' / 'KD-SBIR'
if source_bundles:
    run(['git', 'bundle', 'verify', str(source_bundles[0])])
    origin = str(source_bundles[0])
    print('Using uploaded local branch:', origin)
else:
    origin = REPO
    print('No local source.bundle found; GitHub branch must already be pushed')
run(['git', 'clone', '--branch', BRANCH, '--single-branch', origin, str(project)])
run(['git', 'checkout', '--detach', COMMIT], project)
actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
if actual != COMMIT:
    raise RuntimeError('Source commit mismatch')
sys.path.insert(0, str(project))
from src.train_correspondence import SOURCE_FILES
source_hashes = {name: sha256(project / name) for name in SOURCE_FILES}
for name in ('tests/test_correspondence.py', 'tests/test_correspondence_setup.py', 'docs/region_correspondence.md',
             'test/kaggle_online.py', 'test/kaggle_offline.py', 'test/kaggle_correspondence_prepare.ipy',
             'test/kaggle_correspondence_train.ipy', 'test/kaggle_correspondence_report.py', 'test/RUN_ORDER.txt'):
    source_hashes[name] = sha256(project / name)
print('[2/4] Source validated:', actual)

from huggingface_hub import hf_hub_download
teacher = BUNDLE / 'dfn5b_openclip' / DFN_FILENAME
shutil.copy2(hf_hub_download(DFN_REPO, filename=DFN_FILENAME, revision=DFN_REVISION), teacher)
if sha256(teacher) != DFN_SHA256:
    raise RuntimeError('DFN5B checksum mismatch')
for name in list(sys.modules):
    if name == 'clip' or name.startswith('clip.'):
        del sys.modules[name]
from clip import clip as project_clip
student = BUNDLE / 'clip_cache' / STUDENT_FILENAME
shutil.copy2(project_clip.download_model('ViT-B/32'), student)
if sha256(student) != STUDENT_SHA256:
    raise RuntimeError('Student checksum mismatch')
print('[3/4] Teacher/student checkpoint hashes verified')

manifest = {'repository': REPO, 'branch': BRANCH, 'commit': COMMIT, 'task': TASK, 'entrypoint': ENTRYPOINT,
            'protocols': ['category', 'fg'], 'teacher_repo': DFN_REPO, 'teacher_revision': DFN_REVISION,
            'teacher_filename': DFN_FILENAME, 'teacher_sha256': DFN_SHA256, 'teacher_size': teacher.stat().st_size,
            'student_filename': STUDENT_FILENAME, 'student_sha256': STUDENT_SHA256, 'student_size': student.stat().st_size,
            'source_sha256': source_hashes, 'requirements_sha256': sha256(requirements),
            'wheel_sha256': {p.name: sha256(p) for p in (BUNDLE / 'wheels').glob('*.whl')},
            'python_minor': list(sys.version_info[:2]), 'kaggle_stack': stack, 'dependency_versions': pinned}
(BUNDLE / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('[4/4] CORRESPONDENCE BUNDLE COMPLETE:', BUNDLE)
print('Save Version -> Save & Run All -> Always save output.')
print('Attach this correspondence_bundle output to the offline GPU notebook.')
