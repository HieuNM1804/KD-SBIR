"""Write training evidence and select trained epochs by precision."""

import json
import math
from pathlib import Path


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def best_validation_epoch(history):
    if not history:
        raise ValueError("No trained validation epochs were recorded.")
    for row in history:
        if not all(math.isfinite(row[key]) for key in ("precision", "mAP")):
            raise ValueError("Non-finite retrieval metrics cannot select a winner.")
    # Preserve the baseline's first-epoch tie handling and checkpoint selection.
    return max(history, key=lambda row: row["precision"])
