"""Fresh blind inference with the frozen final model, without historical caches.

This report attachment is separate from the archived experiment. It never trains,
selects candidates, changes sealed outputs, or needs the deleted FP32 caches.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from evaluate_sied1000 import sha256, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--lut-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-new", type=int, choices=(0, 1, 2, 3), default=0,
                        help="A train-only smoke run; zero completes all 1000")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    frozen_path = project / "reports/assets/enhancement/submission1000_20261024/binding.json"
    frozen = json.loads(frozen_path.read_text())
    paths = {"official": args.official_checkpoint, "refiner": args.refiner_checkpoint,
             "lut": args.lut_checkpoint}
    expected = {"official": frozen["official_sha256"], "refiner": frozen["refiner_sha256"],
                "lut": frozen["selection"]["selected"]["checkpoint_sha256"]}
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Wrong frozen checkpoint: {name}")
    for name, digest in frozen["code_sha256"].items():
        if sha256(project / name) != digest:
            raise ValueError(f"Frozen inference source changed: {name}")

    import torch
    from dip_course.cli._common import write_json
    from dip_course.cli.infer_official_sied import (
        _autocast_context, average_prediction_samples, diffusion_seed,
        load_official_model, load_refiner, predict_official_sied,
    )
    from dip_course.enhancement.adaptive_lut import AdaptiveColorLUT, apply_color_lut_chunked
    from dip_course.enhancement.refiner import apply_raw_guided_refiner_tiled
    from dip_course.io import RawSpec, load_packed_raw, load_rgb, read_pair_manifest, save_rgb
    from dip_course.metrics import LPIPSMetric, color_fidelity_metrics, psnr_rgb, ssim_rgb

    if not torch.cuda.is_available():
        raise RuntimeError("The archived FP16 inference protocol requires CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    manifests = {s: project / f"manifests/enhancement/canon_{s}.txt"
                 for s in ("train", "val", "test")}
    for s, path in manifests.items():
        if sha256(path) != frozen["split_manifests_sha256"][s]:
            raise ValueError(f"Split manifest changed: {s}")
    records = {s: read_pair_manifest(args.data_root, p) for s, p in manifests.items()}
    if {s: len(v) for s, v in records.items()} != {"train": 832, "val": 92, "test": 76}:
        raise ValueError("Expected all 1000 pairs")
    if len({r.key for v in records.values() for r in v}) != 1000:
        raise ValueError("Image keys must be unique")
    output = args.output_dir.resolve()
    if output == (project / "outputs/sied1000_20261024/submission1000_native").resolve():
        raise ValueError("Choose a new reproduction output directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = frozen["protocol"]
    binding = {"purpose": "fresh_inference_reproduction_not_archived_experiment",
               "frozen_binding_sha256": sha256(frozen_path), "weights": expected,
               "protocol": protocol, "source_sha256": frozen["code_sha256"],
               "script_sha256": sha256(Path(__file__)),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    binding_file = output / "binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("Resume model, protocol, hardware or source changed")
    write_json(binding_file, binding)
    binding_hash = sha256(binding_file)
    base, _ = load_official_model(args.official_repo, args.official_checkpoint,
                                  sampling_steps=protocol["sampling_steps"], device=device)
    refiner, _ = load_refiner(args.refiner_checkpoint, device)
    state = torch.load(args.lut_checkpoint, map_location=device, weights_only=True)
    if state["model_type"] != "adaptive_color_lut":
        raise ValueError("The final submission uses the 1250-step adaptive LUT")
    lut = AdaptiveColorLUT(**state["model_config"]).to(device).eval()
    lut.load_state_dict(state["model"], strict=True)
    perceptual = LPIPSMetric("cuda")
    rows, new = [], 0
    for split, subset in records.items():
        for index, record in enumerate(subset):
            name = f"{index:04d}_{record.key}"
            row_file = output / "rows" / split / f"{name}.json"
            image_file = output / "images" / split / f"{name}.png"
            seeds = [diffusion_seed(protocol["seed"], index, j)
                     for j in range(protocol["ensemble_size"])]
            if row_file.exists():
                row = json.loads(row_file.read_text())
                if (row["binding_sha256"] != binding_hash or row["key"] != record.key
                        or row["split"] != split or row["index"] != index
                        or row["png_sha256"] != sha256(image_file)):
                    raise ValueError(f"Invalid resume row: {name}")
                summarize([row])
                rows.append(row)
                continue
            if args.limit_new and split != "train":
                raise ValueError("Smoke evaluation is restricted to training images")
            started = time.perf_counter()
            raw = torch.from_numpy(load_packed_raw(record.low_raw, RawSpec())).unsqueeze(0).cuda()
            samples = []
            for seed in seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                samples.append(predict_official_sied(base, raw, precision=protocol["precision"]))
            base_rgb = average_prediction_samples(samples)
            del samples
            with torch.inference_mode(), _autocast_context(device, protocol["precision"]):
                predicted = apply_raw_guided_refiner_tiled(
                    refiner, base_rgb, raw, tile_size=protocol["refiner_tile_size"],
                    overlap=protocol["refiner_overlap"])
            predicted = apply_color_lut_chunked(lut, predicted.float())
            torch.cuda.synchronize()
            image = predicted.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            del predicted, base_rgb, raw
            prediction_seconds = time.perf_counter() - started
            target = load_rgb(record.gt_rgb)
            if image.shape != (2160, 3840, 3) or target.shape != image.shape:
                raise ValueError("Native 3840x2160 RGB evaluation is required")
            row = {"key": record.key, "split": split, "index": index,
                   "binding_sha256": binding_hash, "seeds": ";".join(map(str, seeds)),
                   "psnr": psnr_rgb(image, target), "ssim": ssim_rgb(image, target),
                   **color_fidelity_metrics(image, target), "lpips": perceptual(image, target),
                   "prediction_seconds": prediction_seconds}
            summarize([row])
            save_rgb(image_file, image)
            row["png_sha256"] = sha256(image_file)
            write_json(row_file, row)
            rows.append(row)
            new += 1
            write_json(output / "progress.json", {"completed": len(rows), "total": 1000})
            print(json.dumps({"completed": len(rows), "key": record.key,
                              "psnr": row["psnr"], "ssim": row["ssim"]}), flush=True)
            if args.limit_new and new >= args.limit_new:
                return
    if len(rows) != 1000 or len({r["key"] for r in rows}) != 1000:
        raise ValueError("Partial runs cannot report an overall mean")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"count": 1000, "overall": summarize(rows),
              "splits": {s: summarize([r for r in rows if r["split"] == s]) for s in records},
              "interpretation": "Includes training images; not independent generalization.",
              "comparison_note": "Fresh FP16 GPU inference can differ numerically from archived caches."}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
