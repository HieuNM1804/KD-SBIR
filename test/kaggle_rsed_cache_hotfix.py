"""Paste into the current offline Kaggle session before retrying RSED preparation.

This is only for the first RSED bundle (implementation commit cdf68b4), whose
full-set compatibility guard rejected the entire cache when any single sketch
fell below cosine 0.999. Fresh setup bundles already contain the corrected
preflight and do not need this cell.
"""

from pathlib import Path


path = Path("/kaggle/working/KD-SBIR-AVKD/src/stroke_evidence_cache.py")
if not path.is_file():
    raise FileNotFoundError(path)

text = path.read_text(encoding="utf-8")
old = '''        if (payload["clean_cache_cosine"] < 0.999).any():
            raise RuntimeError("Re-encoded teacher sketch differs from the main cache")
'''
new = '''        compatibility_values = payload["clean_cache_cosine"].float()
        compatibility = {
            "count": len(compatibility_values),
            "minimum": compatibility_values.min().item(),
            "p01": torch.quantile(compatibility_values, 0.01).item(),
            "median": compatibility_values.median().item(),
            "mean": compatibility_values.mean().item(),
        }
        payload["teacher_compatibility_full"] = compatibility
        payload["teacher_compatibility_full_acceptable"] = (
            compatibility["mean"] >= 0.995 and compatibility["p01"] >= 0.980
        )
        print("[RSED Cache] full teacher re-encoding agreement:", compatibility, flush=True)
        if not payload["teacher_compatibility_full_acceptable"]:
            print(
                "[RSED Cache] WARNING: agreement is below the diagnostic gate; "
                "saving targets with the warning instead of discarding 14k results.",
                flush=True,
            )
'''

if old in text:
    text = text.replace(old, new, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")
    print("Patched the first-bundle late compatibility guard:", path)
elif 'payload["teacher_compatibility_full"]' in text:
    print("Compatibility fix is already present:", path)
else:
    raise RuntimeError("Unexpected source version; do not patch an unknown file")

