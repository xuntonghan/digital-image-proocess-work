"""Freeze a validation-selected enhancement pipeline, then score all 1000 images.

This separate submission run never overwrites the historical baseline/test.
All 1000 rows use ONE selected pipeline, including fresh blind test inference.
The overall mean includes training images; it is not a generalization score.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import shutil
import time
from pathlib import Path

from evaluate_sied1000 import METRICS, sha256, summarize


def eligible(metrics: dict, baseline: dict) -> bool:
    """Keep the previously declared validation safeguards, not test-based tuning."""
    if metrics.get("count") != 92 or baseline.get("count") != 92:
        return False
    fields = set(METRICS.values())
    if any(not math.isfinite(float(metrics[field])) for field in fields):
        return False
    return (metrics["psnr_mean"] >= baseline["psnr_mean"] + 0.02
            and metrics["ssim_mean"] >= baseline["ssim_mean"] - 0.002
            and all(metrics[field] <= baseline[field] * 1.02 for field in fields
                    if field not in {"psnr_mean", "ssim_mean"}))


def select_candidate(first: dict, paired: dict, baseline: dict) -> dict:
    """Selection reads complete validation summaries ONLY, never 1000/test scores."""
    if first.get("test_evaluated") is not False or paired.get("test_evaluated") is not False:
        raise ValueError("Research must not have selected using test evaluation")
    if set(paired.get("arms", {})) != {"global3d", "pairwise_rank4"}:
        raise ValueError("Both bounded research arms must finish before final selection")
    candidates = [{"name": "baseline", "checkpoint": None, "checkpoint_sha256": None,
                   "full_val": baseline}]
    if first.get("lut_full_passed") and eligible(first["lut_full_val"], baseline):
        candidates.append({"name": "adaptive_color_lut",
                           "checkpoint": "research/color_lut_tlc/best_lut.pt",
                           "checkpoint_sha256": first["best_lut_sha256"],
                           "full_val": first["lut_full_val"]})
    for arm, result in paired["arms"].items():
        if result.get("full_passed") and eligible(result["full_val"], baseline):
            candidates.append({"name": arm, "checkpoint": f"research/iccv_pairwise/{arm}/best.pt",
                               "checkpoint_sha256": result["checkpoint_sha256"],
                               "full_val": result["full_val"]})
    return {"selected": max(candidates, key=lambda c: (c["full_val"]["psnr_mean"],
                                                      c["full_val"]["ssim_mean"])),
            "candidates": candidates,
            "rule": "full 92-val safeguards; highest PSNR, SSIM breaks exact PSNR ties",
            "test_or_overall_used_for_selection": False}


def require_complete(rows: list[dict], expected: dict[str, list[str]]) -> None:
    if {split: len(keys) for split, keys in expected.items()} != {"train": 832, "val": 92, "test": 76}:
        raise ValueError("Expected the fixed 832/92/76 partition")
    identities = [(split, index, key) for split, keys in expected.items()
                  for index, key in enumerate(keys)]
    if (len(rows) != 1000 or len({r["key"] for r in rows}) != 1000
            or [(r["split"], r["index"], r["key"]) for r in rows] != identities):
        raise ValueError("Exactly 1000 unique, ordered images from all three splits required")
    if len({r["binding_sha256"] for r in rows}) != 1:
        raise ValueError("Cannot mix different pipelines in an overall mean")
    summarize(rows)  # Reject missing/nonfinite metrics; never filter bad images.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-new", type=int, default=0, help="Train-only pilot; no partial overall mean")
    args = parser.parse_args()
    if args.limit_new < 0 or args.limit_new > 3:
        parser.error("pilot limit must be 0..3; never pilot or tune using test images")
    root = args.root.resolve()
    project = root / "project"
    evaluation = root / "evaluation1000_native"
    output = root / "submission1000_native"
    output.mkdir(parents=True, exist_ok=True)
    lock = (root / "logs/gpu_work.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    run_lock = (output / "evaluation.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    import importlib.metadata

    import numpy as np
    import torch

    from dip_course.cli._common import write_json
    from dip_course.cli.infer_official_sied import (
        _autocast_context,
        average_prediction_samples,
        diffusion_seed,
        load_official_model,
        load_refiner,
        predict_official_sied,
    )
    from dip_course.enhancement.adaptive_lut import (
        AdaptiveColorLUT,
        apply_color_lut_chunked,
    )
    from dip_course.enhancement.factorized_lut import PairwiseLowRankLUT
    from dip_course.enhancement.refiner import apply_raw_guided_refiner_tiled
    from dip_course.io import (
        RawSpec,
        load_packed_raw,
        load_rgb,
        read_pair_manifest,
        save_rgb,
    )
    from dip_course.metrics import (
        LPIPSMetric,
        color_fidelity_metrics,
        psnr_rgb,
        ssim_rgb,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Use the provisioned CUDA GPU, not an unexpected CPU fallback")
    torch.set_num_threads(4)
    baseline = json.loads((evaluation / "summary.json").read_text())
    receipt = json.loads((evaluation / "verification.json").read_text())
    original_binding = json.loads((evaluation / "binding.json").read_text())
    for filename, field in (("summary.json", "summary_sha256"), ("metrics.csv", "metrics_sha256"),
                            ("binding.json", "binding_sha256")):
        if sha256(evaluation / filename) != receipt[field]:
            raise ValueError(f"Historical baseline integrity failure: {filename}")
    for filename, expected_hash in original_binding["code_sha256"].items():
        if sha256(project / filename) != expected_hash:
            raise ValueError(f"Historical baseline source changed: {filename}")
    first_path = root / "research/color_lut_tlc/result.json"
    paired_path = root / "research/iccv_pairwise/result.json"
    choice = select_candidate(json.loads(first_path.read_text()), json.loads(paired_path.read_text()),
                              baseline["splits"]["val"])
    chosen = choice["selected"]
    if chosen["checkpoint"] and sha256(root / chosen["checkpoint"]) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected LUT checkpoint hash mismatch")
    protocol = original_binding["protocol"]
    official = root / "checkpoints/official/canon_0.0001.pt"
    refiner_path = root / "final/best.pt"
    if (sha256(official) != original_binding["official_sha256"]
            or sha256(refiner_path) != original_binding["refiner_sha256"]):
        raise ValueError("Underlying pipeline weights changed")
    data = root / "data/SIED_dataset"
    manifests = {s: project / f"manifests/enhancement/canon_{s}.txt"
                 for s in ("train", "val", "test")}
    records = {s: read_pair_manifest(data, path) for s, path in manifests.items()}
    for split, path in manifests.items():
        if sha256(path) != original_binding["split_manifests_sha256"][split]:
            raise ValueError(f"Split changed: {split}")
    source_manifest = project / "manifests/source/canon_all.txt"
    source = read_pair_manifest(data, source_manifest)
    pairs = [(r.low_raw, r.gt_rgb) for rs in records.values() for r in rs]
    if (sha256(source_manifest) != original_binding["all_manifest_sha256"]
            or len(source) != 1000 or len(pairs) != 1000 or len(set(pairs)) != 1000
            or len({r.key for rs in records.values() for r in rs}) != 1000
            or set(pairs) != {(r.low_raw, r.gt_rgb) for r in source}
            or {s: len(rs) for s, rs in records.items()} != {"train": 832, "val": 92, "test": 76}):
        raise ValueError("Manifests must exactly cover the same original 1000 pairs")
    for candidate in choice["candidates"]:
        if candidate["name"] == "baseline":
            continue
        metrics_path = (root / "research/color_lut_tlc/lut_full_val/metrics.json"
                        if candidate["name"] == "adaptive_color_lut" else
                        root / f"research/iccv_pairwise/{candidate['name']}/full_val/metrics.json")
        validation_rows = json.loads(metrics_path.read_text())
        if ([r["key"] for r in validation_rows] != [r.key for r in records["val"]]
                or any(r["split"] != "val" for r in validation_rows)):
            raise ValueError("Candidate validation keys/order do not match the fixed 92 images")
        validation_summary = summarize(validation_rows)
        if any(not math.isclose(validation_summary[field], candidate["full_val"][field],
                                rel_tol=1e-10, abs_tol=1e-10) for field in METRICS.values()):
            raise ValueError("Candidate full-validation result disagrees with its 92 per-image rows")
    binding = {
        "schema": 1, "purpose": "course_submission_1000_image_arithmetic_means",
        "interpretation": "Includes 832 training and 92 official-pretraining-exposed validation images; "
                          "NOT an independent generalization score.",
        "baseline_binding_sha256": receipt["binding_sha256"],
        "baseline_metrics_sha256": receipt["metrics_sha256"],
        "baseline_summary_sha256": receipt["summary_sha256"],
        "sealed_test_csv_sha256": sha256(root / "final/frozen_test/metrics.csv"),
        "selection": choice, "research_results_sha256": {
            "first": sha256(first_path), "paired": sha256(paired_path)},
        "all_manifest_sha256": sha256(source_manifest),
        "split_manifests_sha256": {s: sha256(p) for s, p in manifests.items()},
        "official_sha256": sha256(official), "refiner_sha256": sha256(refiner_path),
        "protocol": protocol, "seed_indexing": "original split-local index",
        "primary_metrics": ["psnr_mean", "ssim_mean"],
        "aggregation": "sum of 1000 per-image metrics / 1000; not PSNR of pooled MSE",
        "prediction_metrics": "native 3840x2160 float RGB before PNG quantization",
        "test_policy": "one final blind inference with the frozen selected pipeline; no test-based tuning",
        "cache_policy": "hash-verified exact FP32 blind diffusion base for train/val only; fresh test base",
        "gpu": torch.cuda.get_device_name(0),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "torchvision", "numpy", "Pillow", "scikit-image", "lpips")},
        "code_sha256": {p: sha256(project / p) for p in (
            "cloud/finalize_sied_submission1000.py", "cloud/evaluate_sied1000.py",
            "src/dip_course/cli/infer_official_sied.py", "src/dip_course/enhancement/refiner.py",
            "src/dip_course/enhancement/adaptive_lut.py", "src/dip_course/enhancement/factorized_lut.py",
            "src/dip_course/io.py", "src/dip_course/metrics.py")},
    }
    binding_path = output / "binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("Submission binding changed; cannot mix or replace candidates after scoring")
    else:
        write_json(binding_path, binding)  # Freeze BEFORE reading any final test GT.
    binding_hash = sha256(binding_path)
    lut = None
    if chosen["checkpoint"]:
        checkpoint = torch.load(root / chosen["checkpoint"], map_location="cuda", weights_only=True)
        if (checkpoint["baseline_binding_sha256"] != receipt["binding_sha256"]
                or checkpoint["baseline_refiner_sha256"] != original_binding["refiner_sha256"]):
            raise ValueError("LUT was not trained against this fixed baseline")
        expected_type = chosen["name"]
        if checkpoint["model_type"] != expected_type:
            raise ValueError("Selected model architecture disagrees with checkpoint")
        model_class = PairwiseLowRankLUT if expected_type == "pairwise_rank4" else AdaptiveColorLUT
        lut = model_class(**checkpoint["model_config"]).cuda().eval()
        lut.load_state_dict(checkpoint["model"], strict=True)
    refiner, _ = load_refiner(refiner_path, torch.device("cuda"))
    model = None  # Diffusion is loaded only when the final test portion starts.
    lpips = LPIPSMetric("cuda")
    rows, new_count = [], 0
    expected = {s: [r.key for r in rs] for s, rs in records.items()}
    for split, subset in records.items():
        for index, record in enumerate(subset):
            name = f"{index:04d}_{record.key}"
            row_path = output / "rows" / split / f"{name}.json"
            image_path = output / "images" / split / f"{name}.png"
            seeds = [diffusion_seed(protocol["seed"], index, j) for j in range(protocol["ensemble_size"])]
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if (row["binding_sha256"] != binding_hash or row["key"] != record.key
                        or row["split"] != split or row["index"] != index
                        or row["seeds"] != ";".join(map(str, seeds))
                        or row["png_sha256"] != sha256(image_path)):
                    raise ValueError(f"Invalid submission resume row: {name}")
                summarize([row])
                rows.append(row)
                continue
            if shutil.disk_usage(output).free < 15 * 1024**3 + 128 * 1024**2:
                raise OSError("Submission output would violate the 15 GiB free-space reserve")
            started = time.perf_counter()
            raw = torch.from_numpy(load_packed_raw(record.low_raw, RawSpec())).unsqueeze(0).cuda()
            cache_hash = None
            if split in {"train", "val"}:
                cache = evaluation / "base_float32" / split / f"{name}.npy"
                reference = json.loads((evaluation / "rows" / split / f"{name}.json").read_text())
                if (reference["binding_sha256"] != receipt["binding_sha256"]
                        or reference["key"] != record.key or reference["index"] != index
                        or reference["split"] != split or reference["seeds"] != ";".join(map(str, seeds))):
                    raise ValueError("Cached base identity/protocol mismatch")
                cache_hash = sha256(cache)
                if cache_hash != reference["base_float32_sha256"]:
                    raise ValueError("Cached blind diffusion base hash mismatch")
                array = np.load(cache, allow_pickle=False)
                if array.shape != (2160, 3840, 3) or array.dtype != np.float32:
                    raise ValueError("Submission requires native FP32 base, not thumbnail/PNG")
                base_tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                del array
            else:
                if model is None:
                    model, _ = load_official_model(
                        root / "third_party/SIED-official", official,
                        sampling_steps=protocol["sampling_steps"], device=torch.device("cuda"))
                samples = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    samples.append(predict_official_sied(model, raw, precision=protocol["precision"]))
                base_tensor = average_prediction_samples(samples)
                del samples  # No test training cache is created.
            with torch.inference_mode(), _autocast_context(torch.device("cuda"), protocol["precision"]):
                prediction_tensor = apply_raw_guided_refiner_tiled(
                    refiner, base_tensor, raw, tile_size=protocol["refiner_tile_size"],
                    overlap=protocol["refiner_overlap"])
            if lut is not None:
                prediction_tensor = apply_color_lut_chunked(lut, prediction_tensor.float())
            torch.cuda.synchronize()
            prediction = prediction_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            del raw, base_tensor, prediction_tensor
            prediction_seconds = time.perf_counter() - started
            target = load_rgb(record.gt_rgb)  # Prediction is already fixed; GT is never a model input.
            row = {"split": split, "index": index, "key": record.key,
                   "binding_sha256": binding_hash, "seeds": ";".join(map(str, seeds)),
                   "evaluation_source": "same_frozen_selected_pipeline_float_native",
                   "base_source": "verified_blind_fp32_cache" if cache_hash else "fresh_blind_diffusion",
                   "base_float32_sha256": cache_hash,
                   "psnr": psnr_rgb(prediction, target), "ssim": ssim_rgb(prediction, target),
                   **color_fidelity_metrics(prediction, target), "lpips": lpips(prediction, target),
                   "prediction_seconds_with_cache_io": prediction_seconds}
            summarize([row])
            save_rgb(image_path, prediction)
            row["png_sha256"] = sha256(image_path)
            row["wall_seconds"] = time.perf_counter() - started
            write_json(row_path, row)
            rows.append(row)
            new_count += 1
            progress = {"state": "running", "completed": len(rows), "total": 1000,
                        "last_split": split, "last_index": index, "last_key": record.key,
                        "last_wall_seconds": row["wall_seconds"], "selected": chosen["name"]}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if args.limit_new and new_count >= args.limit_new:
                print("TRAIN_PILOT_ONLY: no partial 1000-image score", flush=True)
                return
    require_complete(rows, expected)
    if (sha256(evaluation / "metrics.csv") != receipt["metrics_sha256"]
            or sha256(root / "final/frozen_test/metrics.csv") != binding["sealed_test_csv_sha256"]):
        raise ValueError("Historical baseline/test changed during final submission evaluation")
    summary = {"purpose": binding["purpose"], "binding_sha256": binding_hash,
               "primary_metrics": binding["primary_metrics"], "overall": summarize(rows),
               "splits": {s: summarize([r for r in rows if r["split"] == s]) for s in records},
               "baseline_overall": baseline["overall"],
               "interpretation": binding["interpretation"],
               "selected": chosen["name"], "precision": binding["prediction_metrics"],
               "timing_caveat": "Cached train/val exclude diffusion generation; test includes it. "
                                "Do not label this mixed time as end-to-end single-image latency."}
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", {
        "binding_sha256": binding_hash, "metrics_sha256": sha256(output / "metrics.csv"),
        "summary_sha256": sha256(output / "summary.json"), "unique_count": 1000,
        "historical_baseline_csv_unchanged": True,
        "historical_test_csv_unchanged": True,
    })
    write_json(output / "progress.json", {"state": "complete", "count": 1000, "selected": chosen["name"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
