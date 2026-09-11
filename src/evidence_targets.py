"""Compact, atomic teacher target cache. No student state or hidden patches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from tqdm.auto import tqdm
from src.dataset import TeacherFeatureDataset
from src.evidence_references import balanced_indices, labels_for, prototypes, scores, margin
from src.evidence_views import regions, intervene


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(paths, root):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.relpath(path, root).replace("\\", "/").encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def select_region(clean, masked, cropped, label, args, generator, kind):
    """Return a teacher-selected region, or random region with matched eligibility.

    Random control uses exactly the teacher-eligible images/kinds, but randomly
    assigns one of the equal-area candidates. Selection consumes only local RNG.
    """
    if margin(clean, label) <= 0:
        return None
    drops = margin(clean, label) - margin(masked, label)
    rms = (clean[None] - masked).square().mean(-1).sqrt()
    if kind == "important":
        valid = (drops >= args.evidence_min_drop) & (margin(cropped, label) > 0)
        rank = drops.masked_fill(~valid, -torch.inf)
        index = rank.argmax().item()
    else:
        valid = rms <= args.evidence_stable_rms
        rank = rms.masked_fill(~valid, torch.inf)
        index = rank.argmin().item()
    if not valid.any():
        return None
    if args.evidence_selection == "random":
        index = torch.randint(len(masked), (), generator=generator).item()
    return index


def validate_payload(payload, metadata, count, classes):
    if payload.get("metadata") != metadata:
        raise ValueError("Evidence cache metadata mismatch. Choose a NEW evidence_cache_path; no automatic overwrite.")
    for key, shape in (("clean", (count, classes)),
                       ("masked", (count, 5, classes)), ("cropped", (count, 5, classes))):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid evidence cache tensor: {key}")


def prepare_targets(args, student, dataset):
    if not Path(args.teacher_cache_path).is_file():
        raise ValueError("Materialize the main teacher cache before preparing evidence targets")
    photo_labels = labels_for(dataset.all_photo_paths, dataset.all_categories)
    selected = balanced_indices(photo_labels, args.evidence_photos_per_class, args.seed + 600)
    ref_paths = (dataset.all_sketches_path if args.evidence_reference == "sketch"
                 else dataset.all_photo_paths)
    ref_labels = labels_for(ref_paths, dataset.all_categories)
    ref_ids = balanced_indices(ref_labels, args.evidence_refs_per_class, args.seed + 601)
    references = [ref_paths[i] for i in ref_ids.tolist()]
    photo_paths = [dataset.all_photo_paths[i] for i in selected.tolist()]
    boxes = regions(dataset.max_size, args.evidence_area)
    # Donors come from a different train class. The same donor is used for all
    # candidate regions of an image, and replayed exactly during student training.
    donors = []
    generator = torch.Generator().manual_seed(args.seed + 602)
    for i in selected:
        candidates = torch.where(photo_labels != photo_labels[i])[0]
        if not len(candidates):
            raise ValueError("Evidence targets require at least two train classes")
        donors.append(candidates[torch.randint(len(candidates), (), generator=generator)].item())
    donor_paths = [dataset.all_photo_paths[i] for i in donors]
    import open_clip, PIL
    print("[Evidence] Checking image content and teacher checkpoint fingerprints")
    metadata = {
        "version": 1, "teacher_sha256": sha256(args.teacher_cache_path),
        "teacher_metadata": student._teacher_cache_metadata(dataset),
        "photo_indices": selected.tolist(), "reference_indices": ref_ids.tolist(),
        "reference_modality": args.evidence_reference, "seed": args.seed,
        "photo_content": image_fingerprint(photo_paths, args.root),
        "reference_content": image_fingerprint(references, args.root),
        "donor_indices": donors if args.evidence_fill == "donor" else [],
        "donor_content": image_fingerprint(donor_paths, args.root) if args.evidence_fill == "donor" else "",
        "boxes": [list(b) for b in boxes], "fill": args.evidence_fill,
        "view_definition": "normalized-cpu-fp32-mean-or-donor;crop-bilinear-align_corners_false",
        "torch": str(torch.__version__), "open_clip": open_clip.__version__, "pillow": PIL.__version__,
        "teacher_microbatch": args.evidence_teacher_batch_size,
    }
    cache = Path(args.evidence_cache_path)
    n, c = len(selected), len(dataset.all_categories)
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        validate_payload(payload, metadata, n, c)
        print(f"[Evidence] Reused {cache}; skipped DFN loading")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        estimate = n * 11 * c * 4
        if shutil.disk_usage(cache.parent).free < estimate + 2 * 1024**3:
            raise OSError("Insufficient disk for evidence targets plus 2 GiB reserve")
        main_payload = torch.load(args.teacher_cache_path, map_location="cpu", weights_only=True)
        if main_payload["metadata"] != student._teacher_cache_metadata(dataset):
            raise ValueError("Main teacher metadata mismatch")
        from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, _build_teacher_prompts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = open_clip.create_model(DFN5B_MODEL, pretrained=DFN5B_PRETRAINED,
                                        precision="fp16" if device.type == "cuda" else "fp32", device=device)
        teacher.eval().requires_grad_(False)
        prompts = _build_teacher_prompts(args, teacher)
        prompts.load_state_dict(main_payload["teacher_prompt_state_dict"], strict=True)
        prompts.eval().requires_grad_(False)
        del main_payload
        dtype = teacher.visual.conv1.weight.dtype

        @torch.no_grad()
        def encode(images, modality):
            outputs = []
            for start in range(0, len(images), args.evidence_teacher_batch_size):
                current = torch.stack(images[start:start + args.evidence_teacher_batch_size]).to(device, dtype=dtype)
                outputs.append(prompts(current, modality).float().cpu())
            return torch.cat(outputs)

        reference_data = TeacherFeatureDataset(references, dataset.max_size)
        ref_features = []
        for start in range(0, len(references), args.evidence_teacher_batch_size):
            ref_features.append(encode([reference_data[i] for i in range(start, min(len(references), start + args.evidence_teacher_batch_size))], args.evidence_reference))
        proto = prototypes(torch.cat(ref_features), ref_labels[ref_ids], c)
        payload = {"metadata": metadata, "clean": torch.empty(n, c),
                   "masked": torch.empty(n, 5, c), "cropped": torch.empty(n, 5, c)}
        photos = TeacherFeatureDataset(photo_paths, dataset.max_size)
        donor_data = TeacherFeatureDataset(donor_paths, dataset.max_size)
        for row in tqdm(range(n), desc="Evidence teacher views", disable=not args.progress):
            image = photos[row]
            donor = donor_data[row] if args.evidence_fill == "donor" else None
            pairs = [intervene(image, box, args.evidence_fill, donor) for box in boxes]
            views = [image] + [p[0] for p in pairs] + [p[1] for p in pairs]
            q = scores(encode(views, "photo"), proto)
            payload["clean"][row] = q[0]
            payload["masked"][row] = q[1:6]
            payload["cropped"][row] = q[6:11]
        validate_payload(payload, metadata, n, c)
        temporary = cache.with_name(cache.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, cache)
        del prompts, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Evidence] Saved {cache} ({cache.stat().st_size / 1024**2:.1f} MiB); teacher released")
    return payload, references, ref_labels[ref_ids], photo_labels[selected]
