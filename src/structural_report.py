"""First-training-batch diagnostics, not semantic correctness annotations."""
import base64
import html
import io
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw


def png_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()


def save_transport_report(directory, modality, image, ct, cs, plan, teacher_identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/(modality+"_transport.html")
    if target.exists():
        raise FileExistsError(f"Choose a new --local_report_dir: {target}")
    mean = image.new_tensor([.48145466,.4578275,.40821073])[:,None,None]
    std = image.new_tensor([.26862954,.26130258,.27577711])[:,None,None]
    pixels = ((image.detach()*std+mean).clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
    original = Image.fromarray(pixels).resize((336,336))
    overlays = []
    for count in (ct.shape[-1],cs.shape[-1]):
        overlay = original.copy();draw = ImageDraw.Draw(overlay)
        grid = math.isqrt(count)
        for i in range(grid+1):
            pos = round(i*336/grid)
            draw.line((pos,0,pos,336),fill="red",width=1)
            draw.line((0,pos,336,pos),fill="red",width=1)
        overlays.append(png_uri(overlay))
    p = plan.detach().float().cpu().numpy()
    intensity = p / max(float(p.max()),1e-12)
    heat = np.stack([255*intensity,70*intensity,255*(1-intensity)],axis=-1).astype(np.uint8)
    heatmap = png_uri(Image.fromarray(heat).resize((cs.shape[-1]*6,ct.shape[-1]*6),
                                                Image.Resampling.NEAREST))
    payload = dict(modality=modality, teacher=teacher_identity, plan=p.tolist(),
                   teacher_structure=ct.detach().float().cpu().tolist(),
                   student_structure=cs.detach().float().cpu().tolist(),
                   row_error=float(abs(p.sum(1)-1/len(p)).max()),
                   column_error=float(abs(p.sum(0)-1/p.shape[1]).max()))
    (directory/(modality+"_transport.json")).write_text(json.dumps(payload),encoding="utf-8")
    text = f"""<!doctype html><meta charset="utf-8"><title>Transport diagnostic</title>
    <style>body{{font:16px system-ui;margin:30px;max-width:1000px}}figure{{display:inline-block;margin:10px}}
    p{{line-height:1.5}}</style><h1>{html.escape(modality)}: first training image</h1>
    <p>Red grids show teacher pooled regions and student patches on the same input.
    Heatmap rows are teacher tokens; columns are student tokens, both in row-major order.
    Red means larger mass relative to the maximum of this plan. This is a solver
    diagnostic, not evidence of correct part correspondence.</p>
    <figure><img src="{overlays[0]}"><figcaption>Teacher regions</figcaption></figure>
    <figure><img src="{overlays[1]}"><figcaption>Student patches</figcaption></figure>
    <p>Marginal errors: rows {payload['row_error']:.3g}, columns {payload['column_error']:.3g}.</p>
    <img src="{heatmap}" alt="Teacher by student transport mass">"""
    target.write_text(text,encoding="utf-8")
