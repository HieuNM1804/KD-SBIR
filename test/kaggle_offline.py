"""Run as one offline Kaggle cell; restores all four exact pretrained teachers."""

from pathlib import Path
import ast
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/clip-kd-teacher-ensemble4"
EXPECTED_BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
EXPECTED_COMMIT = "d218b11a8bdc7f16d4d697a7157b4cece4e4d034"
EXPECTED_TASK = "clip_kd_teacher_ensemble4"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"
EXPECTED_ENTRYPOINT = "src.train"
RUNTIME_REQUIREMENTS = [
    "open-clip-torch==3.2.0", "pytorch-lightning", "torchmetrics",
    "lightning-utilities", "huggingface-hub", "safetensors", "ftfy", "regex",
    "tensorboard", "packaging", "tqdm", "numpy", "pillow", "matplotlib",
]
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR"
SKETCHY = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_environment(*directories):
    """Trust only these repository paths for this cell's child processes.

    Kaggle input/output mounts can have a different owner. Command-scope Git
    configuration also reaches Git calls inside the unittest subprocess without
    changing the user's global configuration or trusting every repository.
    """
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    for directory in directories:
        environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
        environment[f"GIT_CONFIG_VALUE_{count}"] = Path(directory).resolve().as_posix()
        count += 1
    environment["GIT_CONFIG_COUNT"] = str(count)
    return environment


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
paths = set()
for depth in range(1, 5):
    paths.update(Path(p) for p in glob.glob("/kaggle/input/" + "*/" * depth + "offline_bundle/bundle_manifest.json"))
matching, reports = [], []
for path in sorted(paths):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        reports.append(f"{path}: unreadable ({error})")
        continue
    reports.append(f"{path}: branch={manifest.get('branch')}, commit={manifest.get('commit')}, task={manifest.get('task')}")
    if all(manifest.get(key) == value for key, value in {
        "repository": EXPECTED_REPOSITORY, "branch": EXPECTED_BRANCH,
        "commit": EXPECTED_COMMIT, "base_commit": EXPECTED_BASE_COMMIT,
        "task": EXPECTED_TASK, "dataset": EXPECTED_DATASET, "entrypoint": EXPECTED_ENTRYPOINT,
    }.items()):
        matching.append((path.parent, manifest))
if not matching:
    raise FileNotFoundError("No matching ensemble4 bundle. Expected source " + EXPECTED_COMMIT
                            + "\nInspected:\n" + "\n".join(reports))
if len(matching) > 1:
    print("Multiple matching bundles; using first.")
bundle, manifest = matching[0]
source = bundle / "source" / "KD-SBIR"
print("Selected bundle:", bundle)
required = [source / ".git", bundle / "requirements.txt", bundle / "wheels",
            source / "src" / "teacher_ensemble.py", source / "src" / "model.py",
            source / "src" / "losses.py", source / "src" / "dataset.py",
            source / "src" / "train.py", source / "src" / "teacher_prompts.py",
            source / "clip" / "model.py", source / "tests" / "test_teacher_ensemble.py"]
for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Incomplete bundle: {path}")
source_git_env = git_environment(source)
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True, env=source_git_env).strip() != EXPECTED_COMMIT:
    raise RuntimeError("Source commit mismatch")
if subprocess.check_output(["git", "merge-base", EXPECTED_COMMIT, EXPECTED_BASE_COMMIT], cwd=source, text=True, env=source_git_env).strip() != EXPECTED_BASE_COMMIT:
    raise RuntimeError("Source does not match expected main base")
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True, env=source_git_env).strip():
    raise RuntimeError("Bundle source has tracked modifications")
tree = ast.parse((source / "src" / "teacher_ensemble.py").read_text(encoding="utf-8"))
catalogue = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "TEACHERS" for t in node.targets)))
records = manifest["teachers"]
if [r["key"] for r in records] != list(catalogue):
    raise RuntimeError("Bundle must contain the four specified teachers in catalogue order")
for record in records:
    if any(record.get(k) != v for k, v in catalogue[record["key"]].items()):
        raise RuntimeError(f"Teacher specification mismatch: {record['key']}")
    path = bundle / "teachers" / record["key"] / record["filename"]
    if not path.is_file() or path.stat().st_size != record["size"] or file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"Teacher checkpoint integrity failure: {path}")
    print("Verified teacher:", record["key"])
student_source = bundle / "clip_cache" / manifest["student_filename"]
if not student_source.is_file() or student_source.stat().st_size != manifest["student_size"] or file_sha256(student_source) != manifest["student_sha256"]:
    raise RuntimeError("Student checkpoint integrity failure")
if file_sha256(bundle / "requirements.txt") != manifest["requirements_sha256"]:
    raise RuntimeError("Requirements integrity failure")
for record in manifest["wheels"]:
    path = bundle / "wheels" / record["filename"]
    if not path.is_file() or path.stat().st_size != record["size"] or file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"Wheel integrity failure: {path}")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
                str(bundle / "wheels"), "--only-binary=:all:", *RUNTIME_REQUIREMENTS], cwd=WORKING, check=True)

# Recreate exactly the Hugging Face snapshots used by OpenCLIP's pretrained tags.
hf_home = WORKING / "huggingface_ensemble4"
hf_hub = hf_home / "hub"
for record in records:
    repo_cache = hf_hub / ("models--" + record["repo"].replace("/", "--"))
    snapshot = repo_cache / "snapshots" / record["revision"]
    snapshot.mkdir(parents=True, exist_ok=True)
    target = snapshot / record["filename"]
    if not target.is_file() or target.stat().st_size != record["size"] or file_sha256(target) != record["sha256"]:
        shutil.copy2(bundle / "teachers" / record["key"] / record["filename"], target)
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(record["revision"], encoding="utf-8")
    if record["filename"] != "open_clip_model.safetensors":
        no_exist = repo_cache / ".no_exist" / record["revision"]
        no_exist.mkdir(parents=True, exist_ok=True)
        (no_exist / "open_clip_model.safetensors").touch()
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
shutil.copy2(student_source, student_cache / manifest["student_filename"])
os.environ.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_hub), HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1")
if PROJECT.exists():
    backup = WORKING / "KD-SBIR_previous_ensemble4"
    suffix = 1
    while backup.exists():
        backup = WORKING / f"KD-SBIR_previous_ensemble4_{suffix}"
        suffix += 1
    PROJECT.rename(backup)
    print("Previous project preserved:", backup)
shutil.copytree(source, PROJECT, symlinks=False)
for folder in (SKETCHY / "sketch", SKETCHY / "photo"):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {folder}")
check_code = (
    "from src.teacher_ensemble import TEACHERS; "
    "from huggingface_hub import hf_hub_download; import open_clip; "
    "[(print(k, hf_hub_download(s['repo'], filename=s['filename'], local_files_only=True)), "
    "assert_mapping(k, s)) for k,s in TEACHERS.items()]"
)
check_code = (
    "def assert_mapping(k, s):\n"
    "    import open_clip\n"
    "    assert open_clip.get_pretrained_cfg(s['model'], s['pretrained'])['hf_hub'].rstrip('/') == s['repo'], k\n"
) + check_code
verification_env = git_environment(source, PROJECT)
subprocess.run([sys.executable, "-c", check_code], cwd=PROJECT, check=True, env=verification_env)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_teacher_ensemble.py", "-v"],
               cwd=PROJECT, check=True, env=verification_env)
subprocess.run([sys.executable, "-m", EXPECTED_ENTRYPOINT, "--help"], cwd=PROJECT,
               check=True, env=verification_env, stdout=subprocess.DEVNULL)
print("OFFLINE ENSEMBLE4 SETUP COMPLETE")
print("Project:", PROJECT, "Source:", EXPECTED_COMMIT)
print("Run %cd /kaggle/working/KD-SBIR, then the src.train cell from README.md.")
print("Default: dfn5b + openai_l14 + dfn2b_l14 + laion_h14; equal weights.")
