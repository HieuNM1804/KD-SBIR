"""Kaggle wheel resolver reused from the GZS online setup; no CUDA wheels."""

from pathlib import Path
import json
import sys


def download_runtime_wheels(roots, python_versions, bundle, wheels, run):
    """Resolve each Python separately while treating Kaggle's torch stack as provided.

    Temporary metadata-only wheels describe the already installed torch stack to
    pip's resolver. They are NEVER installed, downloaded into wheels/, or bundled.
    Real runtime dependencies are resolved and downloaded for the target ABI.
    """
    from importlib.metadata import distribution
    import tempfile
    import zipfile
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    def provided(name):
        return (name in {"torch", "torchvision", "torchaudio", "triton"}
                or name.startswith(("nvidia-", "cuda-")))

    platforms = ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64")
    inventories = {}
    with tempfile.TemporaryDirectory(prefix="kaggle-provided-stack-") as temporary:
        stack = []
        for name in ("torch", "torchvision"):
            installed = distribution(name)
            version = installed.version
            info = f"{name}-{version}.dist-info"
            path = Path(temporary) / f"{name}-{version}-py3-none-any.whl"
            metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            for specification in installed.requires or []:
                if not provided(canonicalize_name(Requirement(specification).name)):
                    metadata += f"Requires-Dist: {specification}\n"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info + "/METADATA", metadata + "\n")
                archive.writestr(info + "/WHEEL",
                                 "Wheel-Version: 1.0\nGenerator: kaggle-bundle-resolver\n"
                                 "Root-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr(info + "/RECORD", "")
            stack.append(str(path))

        for python_version in python_versions:
            abi = "cp" + python_version.replace(".", "")
            target = ["--python-version", python_version, "--implementation", "cp",
                      "--abi", abi, "--abi", "abi3", "--abi", "none"]
            for platform in platforms:
                target.extend(["--platform", platform])
            report = bundle / f"resolution-{abi}.json"
            print(f"Resolving non-CUDA dependencies for Python {python_version}", flush=True)
            run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                 "--only-binary=:all:", "--report", str(report), *target, *roots, *stack])
            resolved = {}
            for item in json.loads(report.read_text(encoding="utf-8"))["install"]:
                metadata = item["metadata"]
                name = canonicalize_name(metadata["name"])
                if not provided(name):
                    resolved[name] = metadata["version"]
            requirements = bundle / f"requirements-{abi}.txt"
            requirements.write_text(
                "".join(f"{name}=={version}\n" for name, version in sorted(resolved.items())),
                encoding="utf-8",
            )
            run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                 *target, "--dest", str(wheels), "-r", str(requirements)])
            inventories[python_version] = resolved
    # Retain this filename for existing manifest/checksum consumers. This file
    # lists runtime roots; exact resolved inventories live in requirements-cp*.txt.
    requirements = bundle / "requirements.txt"
    requirements.write_text("\n".join(roots) + "\n", encoding="utf-8")
    return inventories
