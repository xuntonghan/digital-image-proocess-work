"""Verify and recompute the sealed 1000-image submission using only Python's standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

FIELDS = {
    "psnr": "psnr_mean", "ssim": "ssim_mean", "lpips": "lpips_mean",
    "delta_e2000_mean": "delta_e2000_mean", "delta_e2000_p95": "delta_e2000_p95_mean",
    "lightness_mae": "lightness_mae_mean", "saturation_mae": "saturation_mae_mean",
    "hue_error_degrees": "hue_error_degrees_mean",
}


def verify(directory: Path) -> dict:
    receipt = json.loads((directory / "verification.json").read_text())
    for name, key in (("metrics.csv", "metrics_sha256"), ("summary.json", "summary_sha256"),
                      ("binding.json", "binding_sha256")):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != receipt[key]:
            raise ValueError(f"Checksum mismatch: {name}")
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000 or len({r["key"] for r in rows}) != 1000:
        raise ValueError("Exactly 1000 unique images are required")
    if Counter(r["split"] for r in rows) != {"train": 832, "val": 92, "test": 76}:
        raise ValueError("The fixed 832/92/76 partition changed")
    if {r["binding_sha256"] for r in rows} != {receipt["binding_sha256"]}:
        raise ValueError("Mixed or incorrect model bindings")
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        if [int(r["index"]) for r in subset] != list(range(len(subset))):
            raise ValueError(f"Invalid split-local indices: {split}")
        for r in subset:
            index = int(r["index"])
            expected = ";".join(str(3407 + index + j * 1000003) for j in range(3))
            if r["seeds"] != expected:
                raise ValueError(f"Seed mismatch: {r['key']}")
    reference = json.loads((directory / "summary.json").read_text())
    answer = {"count": 1000, "splits": {}}
    for label, subset in [("overall", rows), *[(s, [r for r in rows if r['split'] == s])
                                               for s in ("train", "val", "test")]]:
        calculated = {"count": len(subset)}
        for column, name in FIELDS.items():
            values = [float(r[column]) for r in subset]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite metric: {column}")
            calculated[name] = statistics.fmean(values)
            expected = reference["overall" if label == "overall" else "splits"]
            if label != "overall":
                expected = expected[label]
            if not math.isclose(calculated[name], expected[name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Summary mismatch: {label}/{name}")
        if label == "overall":
            answer[label] = calculated
        else:
            answer["splits"][label] = calculated
    answer["verified"] = True
    answer["note"] = "Arithmetic means include training images; not independent generalization."
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent /
                        "reports/assets/enhancement/submission1000_20261024")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
