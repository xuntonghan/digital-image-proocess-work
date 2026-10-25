"""Prepare faithful scientific display panels and verified report data.

Uses saved inference images only. Does not train, infer, change evaluation,
or apply additional exposure/color correction. All display scaling is recorded.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/sied_final_report_20261025"
IMAGES = OUT / "images"
ASSETS = ROOT / "reports/assets/enhancement"
RUN = ROOT / "outputs/sied1000_20261024"
HISTORY = ROOT / "outputs/sied30_resume_snapshot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def display_copy(source: Path, target: Path, crop=None, width=1440) -> dict:
    with Image.open(source) as im:
        im = im.convert("RGB")
        native = im.size
        if crop is not None:
            im = im.crop(crop)
        im.thumbnail((width, width), Image.Resampling.LANCZOS)
        im.save(target, quality=95, subsampling=0, optimize=True)
        display = im.size
    return {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "native_size": native, "crop_xyxy": crop, "display_size": display,
            "display_file": target.name, "display_sha256": sha256(target),
            "processing": "RGB display copy, equal-aspect Lanczos scaling, JPEG95; no color/exposure change"}


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    binding = read_json(ASSETS / "submission1000_20261024/binding.json")
    receipt = read_json(ASSETS / "submission1000_20261024/verification.json")
    summary = read_json(ASSETS / "submission1000_20261024/summary.json")
    baseline = read_json(ASSETS / "full1000_20261024/summary.json")
    for name, field in (("binding.json", "binding_sha256"), ("metrics.csv", "metrics_sha256"),
                        ("summary.json", "summary_sha256")):
        assert sha256(ASSETS / "submission1000_20261024" / name) == receipt[field]
    with (ASSETS / "submission1000_20261024/metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (ASSETS / "full1000_20261024/metrics.csv").open(newline="") as stream:
        base_rows = {r["key"]: r for r in csv.DictReader(stream)}
    assert len(rows) == len({r["key"] for r in rows}) == 1000
    test = {r["key"]: r for r in rows if r["split"] == "test"}
    fixed = ["6", "8", "10", "14", "21", "28"]
    remaining = sorted((k for k in test if k not in fixed),
                       key=lambda k: (float(test[k]["psnr"]) - float(base_rows[k]["psnr"]), k))
    indices = [round(i * (len(remaining) - 1) / 5) for i in range(6)]
    extra = [remaining[i] for i in indices]
    assert len(set(fixed + extra)) == 12
    montage_path = HISTORY / "final/comparison.png"
    montage_meta = read_json(montage_path.with_suffix(".json"))
    assert montage_meta["keys"] == fixed
    panels, examples = [], []
    with Image.open(montage_path) as montage:
        assert montage.size == (3290, 2198)
        for row_index, key in enumerate(fixed):
            for column, label in ((0, "raw"), (1, "isp")):
                target = IMAGES / f"case_{key}_{label}.jpg"
                crop = (90 + 640 * column, 38 + 360 * row_index,
                        90 + 640 * (column + 1), 38 + 360 * (row_index + 1))
                tile = montage.crop(crop)
                tile.save(target, quality=98, subsampling=0)
                panels.append({"source": str(montage_path.relative_to(ROOT)),
                               "source_sha256": sha256(montage_path), "crop_xyxy": crop,
                               "display_file": target.name, "display_sha256": sha256(target),
                               "processing": "Exact saved historical panel crop; no additional enhancement"})
    for key in fixed + extra:
        row, old = test[key], base_rows[key]
        final_path = RUN / "submission1000_native/images/test" / f"{int(row['index']):04d}_{key}.png"
        baseline_path = HISTORY / "final/frozen_test/images" / final_path.name
        gt = HISTORY / "final_reference/gt" / f"{key}.jpg"
        assert sha256(final_path) == row["png_sha256"]
        paths = {"baseline": baseline_path, "final": final_path, "gt": gt}
        for label, path in paths.items():
            panels.append(display_copy(path, IMAGES / f"case_{key}_{label}.jpg"))
            if key in extra:
                panels.append(display_copy(path, IMAGES / f"case_{key}_{label}_crop.jpg",
                                           crop=(1280, 720, 2560, 1440), width=1280))
        examples.append({"key": key, "split": "test", "index": int(row["index"]),
                         "selection": "first six fixed manifest examples" if key in fixed
                         else "six equally spaced ranks in remaining test PSNR-delta distribution",
                         "baseline": {k: float(old[k]) for k in ("psnr", "ssim", "lpips")},
                         "final": {k: float(row[k]) for k in ("psnr", "ssim", "lpips")},
                         "psnr_delta": float(row["psnr"]) - float(old["psnr"]),
                         "ssim_delta": float(row["ssim"]) - float(old["ssim"]),
                         "crop_xyxy": None if key in fixed else [1280, 720, 2560, 1440]})
    own_weights = {
        "canon_naf_phase4.pt": ROOT / "checkpoints/sied30_resume_snapshot/final/best.pt",
        "canon_color_lut_1250.pt": RUN / "research/color_lut_tlc/best_lut.pt",
        "ablation_global3d_750.pt": RUN / "research/iccv_pairwise/global3d/best.pt",
        "ablation_pairwise_rank4_1000.pt": RUN / "research/iccv_pairwise/pairwise_rank4/best.pt",
        "photometric_v2.pt": ASSETS / "refiner_v2/best.pt",
    }
    weights = {name: {"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": sha256(path)} for name, path in own_weights.items()}
    assert weights["canon_naf_phase4.pt"]["sha256"] == binding["refiner_sha256"]
    assert weights["canon_color_lut_1250.pt"]["sha256"] == binding["selection"]["selected"]["checkpoint_sha256"]
    official = ROOT / "checkpoints/sied30_resume_snapshot/official/canon_0.0001.pt"
    assert sha256(official) == binding["official_sha256"]
    weights["official_canon_0.0001.pt"] = {
        "source": str(official.relative_to(ROOT)), "bytes": official.stat().st_size,
        "sha256": binding["official_sha256"], "redistributed": False,
        "obtain_from": "https://github.com/JianghaiSCU/SIED", "reason": "Author checkpoint, obtain from original publisher and verify SHA256"}
    report_data = {"summary": summary, "baseline": baseline, "binding": binding,
                   "examples": examples, "display_panels": panels, "weights": weights,
                   "first_research": read_json(RUN / "research/color_lut_tlc/result.json"),
                   "paired_research": read_json(RUN / "research/iccv_pairwise/result.json"),
                   "real_data_benchmark": read_json(RUN / "research/iccv_pairwise/real_data_benchmark.json"),
                   "selection_policy": "All 1000 contribute to reported scores. Example selection affects display only.",
                   "raw_preview_note": "Historical saved preview exposure1 gamma2.2; not re-read or re-enhanced RAW.",
                   "isp_preview_note": "Historical display ISP gamma2.2 percentile99 target0.9; differs from formal ISP baseline gamma2.0 percentile98.",
                   "metric_note": "All printed scores come from sealed native float metrics, not resized JPEG display panels."}
    (OUT / "report_data.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report_directory": str(OUT), "examples": [e['key'] for e in examples],
                      "image_assets": len(panels), "own_weights_bytes": sum(w['bytes'] for n,w in weights.items() if not n.startswith('official')),
                      "primary_metrics": summary['overall']}, ensure_ascii=False))


if __name__ == "__main__":
    main()
