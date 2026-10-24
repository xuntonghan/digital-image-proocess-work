"""Fixed-model native evaluation: infer train/val, reuse sealed test results.

This is a descriptive 1000-image statistic, NOT a new generalization score.
No training, model selection, or repeated test inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

METRICS = {
    "psnr": "psnr_mean",
    "ssim": "ssim_mean",
    "delta_e2000_mean": "delta_e2000_mean",
    "delta_e2000_p95": "delta_e2000_p95_mean",
    "lightness_mae": "lightness_mae_mean",
    "saturation_mae": "saturation_mae_mean",
    "hue_error_degrees": "hue_error_degrees_mean",
    "lpips": "lpips_mean",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No rows; refusing to report an empty mean")
    identities = [(row["split"], row["key"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate row identities")
    result = {"count": len(rows)}
    for column, output in METRICS.items():
        values = [float(row[column]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite {column}; no silent filtering")
        result[output] = statistics.fmean(values)
    return result


def make_color_training_pair(prediction, target, *, size=(288, 512)):
    """Area resize the same FP32 RGB values used by metrics, including AMP outputs."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    thumbnails = []
    for array in (prediction, target):
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Color pairs require HWC RGB images")
        chw = np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
        thumbnail = F.interpolate(torch.from_numpy(chw).unsqueeze(0), size=size, mode="area")
        thumbnails.append(thumbnail.squeeze(0).permute(1, 2, 0).numpy())
    return np.stack(thumbnails)


def read_sealed_test(final: Path, expected_keys: list[str], protocol: dict) -> list[dict]:
    receipt = json.loads((final / "frozen_test/verification.json").read_text())
    for name, field in (("metrics.csv", "metrics_sha256"),
                        ("summary.json", "summary_sha256")):
        if sha256(final / "frozen_test" / name) != receipt[field]:
            raise ValueError(f"Sealed test {name} hash mismatch")
    if sha256(final / "selection.json") != receipt["selection_sha256"]:
        raise ValueError("Sealed test selection hash mismatch")
    if protocol["refiner_checkpoint_sha256"] != receipt["checkpoint_sha256"]:
        raise ValueError("Sealed test checkpoint mismatch")
    if protocol["manifest_sha256"] != receipt["manifest_sha256"]:
        raise ValueError("Sealed test manifest mismatch")
    with (final / "frozen_test/metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["key"] for row in rows] != expected_keys:
        raise ValueError("Sealed test keys/order mismatch")
    if receipt["count"] != len(rows) or protocol["count"] != len(rows):
        raise ValueError("Sealed test count mismatch")
    for index, row in enumerate(rows):
        if int(row["index"]) != index:
            raise ValueError("Sealed test index mismatch")
        expected_seeds = ";".join(str(protocol["seed"] + index + j * 1_000_003)
                                  for j in range(protocol["ensemble_size"]))
        if row["seeds"] != expected_seeds:
            raise ValueError("Sealed test seed mismatch")
        row["split"] = "test"
        row["evaluation_source"] = "reused_sealed_float_metrics"
    summary = summarize(rows)
    original = json.loads((final / "frozen_test/summary.json").read_text())
    for field in METRICS.values():
        if not math.isclose(summary[field], original[field], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Sealed test summary disagrees with rows: {field}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-new", type=int, default=0,
                        help="Pilot only; incomplete runs never get a full summary")
    parser.add_argument("--cache-base-fp32", action="store_true",
                        help="Keep exact blind ensemble tensors for later train/val research")
    parser.add_argument("--cache-color-pairs", action="store_true",
                        help="Keep native-prediction-derived train/val thumbnail pairs for LUT research")
    args = parser.parse_args()
    if args.limit_new < 0:
        parser.error("limit-new must be nonnegative")
    root = args.root.resolve()
    project = root / "project"
    data = root / "data/SIED_dataset"
    final = root / "final"
    output = root / "evaluation1000_native"
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

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
    from dip_course.enhancement.refiner import (
        RawGuidedNAFRefiner,
        apply_raw_guided_refiner_tiled,
    )
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
        raise RuntimeError("Evaluation requires the provisioned CUDA GPU")
    torch.set_num_threads(4)
    manifests = {s: project / f"manifests/enhancement/canon_{s}.txt"
                 for s in ("train", "val", "test")}
    records = {s: read_pair_manifest(data, path) for s, path in manifests.items()}
    all_manifest = project / "manifests/source/canon_all.txt"
    all_records = read_pair_manifest(data, all_manifest)
    pairs = [(r.low_raw, r.gt_rgb) for subset in records.values() for r in subset]
    if ({s: len(rs) for s, rs in records.items()} != {"train": 832, "val": 92, "test": 76}
            or len(set(pairs)) != 1000
            or len({r.key for subset in records.values() for r in subset}) != 1000
            or len(all_records) != 1000
            or set(pairs) != {(r.low_raw, r.gt_rgb) for r in all_records}):
        raise ValueError("Splits must form an exact disjoint partition of all 1000 pairs")

    selection = json.loads((final / "selection.json").read_text())
    protocol = selection["frozen_protocol"]
    weight = root / "checkpoints/official/canon_0.0001.pt"
    if sha256(weight) != protocol["official_checkpoint_sha256"]:
        raise ValueError("Official weight hash mismatch")
    if sha256(final / "best.pt") != protocol["refiner_checkpoint_sha256"]:
        raise ValueError("Refiner weight hash mismatch")
    if sha256(manifests["test"]) != protocol["manifest_sha256"]:
        raise ValueError("Frozen test manifest hash mismatch")
    test_rows = read_sealed_test(final, [r.key for r in records["test"]], protocol)
    import importlib.metadata
    binding = {
        "schema": 1, "purpose": "descriptive_full_dataset_not_generalization",
        "all_manifest_sha256": sha256(all_manifest),
        "split_manifests_sha256": {s: sha256(p) for s, p in manifests.items()},
        "selection_sha256": sha256(final / "selection.json"),
        "official_sha256": sha256(weight), "refiner_sha256": sha256(final / "best.pt"),
        "sealed_test_csv_sha256": sha256(final / "frozen_test/metrics.csv"),
        "protocol": protocol,
        "seed_indexing": "original split-local index; never global 1000 index",
        "prediction_metrics": "float native RGB; PNGs are visualization only",
        "training_cache": "native_float32_blind_base" if args.cache_base_fp32 else None,
        "color_training_cache": "float32_area512_rgb_pairs" if args.cache_color_pairs else None,
        "pretraining_exposure": "Official base saw 924 train+val images; refiner saw 832",
        "new_inference_gpu": torch.cuda.get_device_name(0),
        "test_hardware": "historical RTX 4080 SUPER vGPU; no test reinference",
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "torchvision", "numpy", "Pillow", "scikit-image", "lpips")},
        "code_sha256": {p: sha256(project / p) for p in (
            "cloud/evaluate_sied1000.py", "src/dip_course/cli/infer_official_sied.py",
            "src/dip_course/enhancement/refiner.py", "src/dip_course/io.py",
            "src/dip_course/metrics.py")},
    }
    binding_path = output / "binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("Resume binding changed; refusing to mix experiments")
    else:
        write_json(binding_path, binding)
    binding_hash = sha256(binding_path)
    model, _ = load_official_model(root / "third_party/SIED-official", weight,
                                   sampling_steps=protocol["sampling_steps"],
                                   device=torch.device("cuda"))
    refiner, _ = load_refiner(final / "best.pt", torch.device("cuda"))
    if not isinstance(refiner, RawGuidedNAFRefiner):
        raise TypeError("Expected the selected RAW-guided NAF")
    lpips = LPIPSMetric("cuda")
    spec = RawSpec()
    rows: list[dict] = []
    new_count = 0
    for split in ("train", "val"):
        for index, record in enumerate(records[split]):
            name = f"{index:04d}_{record.key}"
            row_path = output / "rows" / split / f"{name}.json"
            image_path = output / "images" / split / f"{name}.png"
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if (row["binding_sha256"] != binding_hash or row["split"] != split
                        or row["key"] != record.key or row["index"] != index
                        or row["png_sha256"] != sha256(image_path)):
                    raise ValueError(f"Invalid resume row: {row_path}")
                if args.cache_base_fp32:
                    cached = output / "base_float32" / split / f"{name}.npy"
                    if row.get("base_float32_sha256") != sha256(cached):
                        raise ValueError(f"Invalid resume base cache: {cached}")
                if args.cache_color_pairs:
                    cached = output / "color_pairs" / split / f"{name}.npy"
                    if row.get("color_pair_sha256") != sha256(cached):
                        raise ValueError(f"Invalid resume color cache: {cached}")
                summarize([row])
                rows.append(row)
                continue
            started = time.perf_counter()
            tensor = torch.from_numpy(load_packed_raw(record.low_raw, spec)).unsqueeze(0).cuda()
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            samples = []
            seeds = []
            for j in range(protocol["ensemble_size"]):
                seed = diffusion_seed(protocol["seed"], index, j)
                seeds.append(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                samples.append(predict_official_sied(model, tensor,
                                                      precision=protocol["precision"]))
            base_tensor = average_prediction_samples(samples)
            base_cache = output / "base_float32" / split / f"{name}.npy"
            if args.cache_base_fp32:
                import numpy as np

                # Cache only train/validation, never the sealed test. No GT is loaded.
                required = base_tensor.numel() * 4 + 15 * 1024**3
                if shutil.disk_usage(output).free < required:
                    raise OSError("Float cache would violate the 15 GiB free-space reserve")
                base_cache.parent.mkdir(parents=True, exist_ok=True)
                with base_cache.with_suffix(".npy.partial").open("wb") as stream:
                    np.save(stream, base_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                            allow_pickle=False)
                os.replace(base_cache.with_suffix(".npy.partial"), base_cache)
            with torch.inference_mode(), _autocast_context(torch.device("cuda"),
                                                            protocol["precision"]):
                prediction_tensor = apply_raw_guided_refiner_tiled(
                    refiner, base_tensor, tensor,
                    tile_size=protocol["refiner_tile_size"],
                    overlap=protocol["refiner_overlap"])
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_started
            prediction = prediction_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            raw_mean = float(tensor.mean().item())
            del prediction_tensor, base_tensor, samples, tensor
            # The GT is deliberately loaded only after the blind prediction.
            target = load_rgb(record.gt_rgb)
            row = {
                "split": split, "index": index, "key": record.key,
                "seeds": ";".join(map(str, seeds)),
                "evaluation_source": "fresh_blind_float_inference",
                "psnr": psnr_rgb(prediction, target), "ssim": ssim_rgb(prediction, target),
                **color_fidelity_metrics(prediction, target),
                "lpips": lpips(prediction, target),
                "inference_seconds": inference_seconds,
                "full_scene_raw_mean": raw_mean,
            }
            summarize([row])
            save_rgb(image_path, prediction)
            row["png_sha256"] = sha256(image_path)
            if args.cache_base_fp32:
                row["base_float32_sha256"] = sha256(base_cache)
            if args.cache_color_pairs:
                import numpy as np

                pair_path = output / "color_pairs" / split / f"{name}.npy"
                pair_path.parent.mkdir(parents=True, exist_ok=True)
                with pair_path.with_suffix(".npy.partial").open("wb") as stream:
                    np.save(stream, make_color_training_pair(prediction, target), allow_pickle=False)
                os.replace(pair_path.with_suffix(".npy.partial"), pair_path)
                row["color_pair_sha256"] = sha256(pair_path)
            row["binding_sha256"] = binding_hash
            row["wall_seconds"] = time.perf_counter() - started
            write_json(row_path, row)
            rows.append(row)
            new_count += 1
            progress = {"state": "running", "fresh_completed": len(rows),
                        "fresh_total": 924, "test_reused": 76, "last_split": split,
                        "last_index": index, "last_key": record.key,
                        "last_psnr": row["psnr"], "last_wall_seconds": row["wall_seconds"]}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if args.limit_new and new_count >= args.limit_new:
                print("PILOT_ONLY: full 1000 summary not yet available", flush=True)
                return
    rows.extend(test_rows)
    if len(rows) != 1000 or len({row["key"] for row in rows}) != 1000:
        raise ValueError("Exactly 1000 unique images required")
    result = {
        "purpose": binding["purpose"], "binding_sha256": binding_hash,
        "overall": summarize(rows),
        "splits": {s: summarize([r for r in rows if r["split"] == s]) for s in records},
        "interpretation": "Includes 832 refiner-training and 92 base-pretraining images. "
                          "NOT an independent test score. Reused 76 sealed test rows unchanged.",
        "precision": "Metrics use float RGB before PNG quantization at native 3840x2160",
    }
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", result)
    write_json(output / "verification.json", {
        "binding_sha256": binding_hash, "summary_sha256": sha256(output / "summary.json"),
        "metrics_sha256": sha256(output / "metrics.csv"), "unique_count": 1000,
    })
    write_json(output / "progress.json", {"state": "complete", "count": 1000,
                                          "fresh_completed": 924, "test_reused": 76})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
