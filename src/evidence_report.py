"""Train-only audit: eligibility, raw scores and selected/random equal-area views."""
import base64
from html import escape
import io
import json
from pathlib import Path
import torch
from torchvision.transforms.functional import to_pil_image
from src.dataset import CLIP_MEAN, CLIP_STD
from src.evidence_references import margin
from src.evidence_views import intervene


def thumbnail(tensor):
    image = tensor.cpu() * torch.tensor(CLIP_STD)[:, None, None] + torch.tensor(CLIP_MEAN)[:, None, None]
    buffer = io.BytesIO()
    to_pil_image(image.clamp(0, 1)).resize((160, 160)).save(buffer, format="PNG")
    return '<img width="160" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def write_report(runtime, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "report.html").exists():
        raise FileExistsError("Use a new evidence_report_dir to preserve the previous audit")
    args, p = runtime.args, runtime.payload
    records, panels, shown = [], [], {}
    for row, index in enumerate(p["metadata"]["photo_indices"]):
        label = int(runtime.photo_labels[row])
        clean = p["clean"][row]
        drops = margin(clean, label) - margin(p["masked"][row], label)
        rms = (clean - p["masked"][row]).square().mean(-1).sqrt()
        correct = bool(margin(clean, label) > 0)
        important = correct & ((drops >= args.evidence_min_drop) & (margin(p["cropped"][row], label) > 0))
        stable = correct & (rms <= args.evidence_stable_rms)
        record = {"photo_index": index, "class": runtime.dataset.all_categories[label],
                  "teacher_clean_correct": correct, "clean_margin": float(margin(clean, label)),
                  "drops": drops.tolist(), "response_rms": rms.tolist(),
                  "important": important.tolist(), "stable": stable.tolist()}
        records.append(record)
        if shown.get(label, 0) >= 2:
            continue
        shown[label] = shown.get(label, 0) + 1
        region = int(drops.argmax())
        donor = runtime.donors[p["metadata"]["donor_indices"][row]] if args.evidence_fill == "donor" else None
        image = runtime.donors[index]
        masked, crop = intervene(image, p["metadata"]["boxes"][region], args.evidence_fill, donor)
        random_region = (region + 1) % 5
        random_view, _ = intervene(image, p["metadata"]["boxes"][random_region], args.evidence_fill, donor)
        panels.append('<article><h3>' + escape(record["class"]) + f' / photo {index}</h3>'
                      + '<p>Clean / strongest-drop crop / strongest-drop mask / another equal-area mask</p>'
                      + ''.join(thumbnail(v) for v in (image, crop, masked, random_view))
                      + '<pre>' + escape(json.dumps(record, indent=2)) + '</pre></article>')
    summary = {"images": len(records),
               "clean_correct": sum(r["teacher_clean_correct"] for r in records),
               "important_eligible": sum(any(r["important"]) for r in records),
               "stable_eligible": sum(any(r["stable"]) for r in records)}
    report = {"summary": summary, "metadata": p["metadata"],
              "thresholds": {"min_drop": args.evidence_min_drop, "stable_rms": args.evidence_stable_rms},
              "records": records}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Evidence audit</title>'
        '<style>body{font-family:system-ui;max-width:1000px;margin:auto}article{border-top:1px solid #ccc}img{margin:4px}pre{white-space:pre-wrap}</style>'
        '<h1>Teacher evidence audit (train classes only)</h1><p>Regions are candidates, not verified object parts. '
        'Strongest-drop illustrations can fail eligibility. No ground-truth shape annotations are available.</p><pre>'
        + escape(json.dumps(summary, indent=2)) + '</pre>' + ''.join(panels), encoding="utf-8")
    print(f"[Evidence Audit] {summary}; {directory / 'report.html'}")
