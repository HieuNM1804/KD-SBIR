"""Verified HTTPS downloads with retries and reuse of completed Kaggle weights."""
import hashlib
import http.client
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, expected_sha=None, blob_id=None):
    path = Path(path)
    if not path.is_file():
        return False
    if expected_sha:
        return sha256(path) == expected_sha
    if blob_id:
        data = path.read_bytes()  # Only small HF config files use Git blob hashes.
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob_id
    raise ValueError("A SHA256 or Git blob checksum is required")


def reuse_verified(target, candidates, expected_sha=None, blob_id=None):
    """Only reuse exact pinned bytes, including from interrupted older bundles."""
    target = Path(target)
    if matches(target, expected_sha, blob_id):
        print("Already verified:", target, flush=True)
        return True
    temporary = target.with_name(target.name + ".part")
    for source in map(Path, candidates):
        if source.resolve() == target.resolve() or not matches(source, expected_sha, blob_id):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)  # Same working filesystem: no second multi-GB copy.
            except OSError:
                shutil.copy2(source, temporary)
            if not matches(temporary, expected_sha, blob_id):
                raise RuntimeError(f"Reused checkpoint changed while copying: {source}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print("Reused verified weights:", source, flush=True)
        return True
    return False


def download_verified(urls, target, expected_sha, attempts=3, timeout=60):
    """Try official HTTPS origins, then curl's TLS stack; never disable TLS checks.

    A partial file is never published as the final checkpoint. Even a successful
    HTTP response must pass the ORIGINAL OpenAI SHA256 before atomic replacement.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    target = Path(target)
    if matches(target, expected_sha):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    urls = list(dict.fromkeys(urls))
    errors = []

    def publish():
        if not matches(temporary, expected_sha):
            raise RuntimeError(f"SHA256 mismatch for {target.name}")
        temporary.replace(target)
        return target

    for attempt in range(1, attempts + 1):
        for url in urls:
            print(f"Downloading {target.name}: attempt {attempt}/{attempts}, {url}", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "KD-SBIR-Kaggle/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
                return publish()
            except (OSError, RuntimeError, http.client.HTTPException) as error:
                errors.append(f"urllib {url}: {type(error).__name__}: {error}")
                print("Retryable download failure:", errors[-1], flush=True)
            finally:
                temporary.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(2 ** attempt, 8))

    curl = shutil.which("curl")
    if curl:
        for url in urls:
            print(f"Trying curl for {target.name}: {url}", flush=True)
            try:
                subprocess.run([curl, "--location", "--fail", "--show-error", "--silent",
                                "--connect-timeout", str(timeout), "--max-time", "900",
                                "--retry", "2", "--retry-all-errors", "--output", str(temporary), url],
                               check=True, timeout=1000)
                return publish()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                errors.append(f"curl {url}: {type(error).__name__}: {error}")
            finally:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("Unable to download a verified original checkpoint. Rerun the online cell "
                       "or set REUSE_WEIGHTS for this file.\n" + "\n".join(errors))
