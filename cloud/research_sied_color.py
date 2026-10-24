"""Bounded validation-only TLC/LUT experiment after immutable baseline evaluation.

Never evaluates the 76 test images and never replaces yesterday's final weights.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_sied1000 import sha256, summarize
from torch.utils.data import DataLoader, Dataset

from dip_course.cli._common import write_json
from dip_course.cli.infer_official_sied import _autocast_context, load_refiner
from dip_course.cli.train_sied_restorer import _save_checkpoint
from dip_course.enhancement.adaptive_lut import (
    AdaptiveColorLUT,
    apply_color_lut_chunked,
)
from dip_course.enhancement.efficiency import (
    enable_local_channel_means,
    mean_image_psnr_loss,
)
from dip_course.enhancement.refiner import apply_raw_guided_refiner_tiled
from dip_course.io import (
    RawSpec,
    load_packed_raw,
    load_rgb,
    read_pair_manifest,
    save_rgb,
)
from dip_course.metrics import LPIPSMetric, color_fidelity_metrics, psnr_rgb, ssim_rgb


def safe_gain(candidate: dict, baseline: dict, gain: float = 0.02) -> bool:
    if candidate["count"] != baseline["count"]:
        return False
    if not all(math.isfinite(float(value)) for value in candidate.values()):
        return False
    if candidate["psnr_mean"] < baseline["psnr_mean"] + gain:
        return False
    if candidate["ssim_mean"] < baseline["ssim_mean"] - 0.002:
        return False
    for name in ("delta_e2000_mean", "delta_e2000_p95_mean", "lightness_mae_mean",
                 "saturation_mae_mean", "hue_error_degrees_mean", "lpips_mean"):
        if candidate[name] > baseline[name] * 1.02:
            return False
    return True


class ColorPairs(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pair = np.load(self.paths[index], allow_pickle=False)
        if pair.shape != (2, 288, 512, 3) or pair.dtype != np.float32:
            raise ValueError(f"Wrong color-pair shape/dtype: {self.paths[index]}")
        return tuple(torch.from_numpy(image.transpose(2, 0, 1).copy()) for image in pair)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1500:
        parser.error("Research budget is 1..1500 steps")
    root = args.root.resolve()
    output = root / "research/color_lut_tlc"
    output.mkdir(parents=True, exist_ok=True)
    lock = (root / "logs/gpu_work.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (output / "experiment_started.json").exists():
        raise RuntimeError("Research already started; inspect/resume explicitly, never overwrite")
    evaluation = root / "evaluation1000_native"
    full = json.loads((evaluation / "summary.json").read_text())
    receipt = json.loads((evaluation / "verification.json").read_text())
    binding = json.loads((evaluation / "binding.json").read_text())
    binding_hash = sha256(evaluation / "binding.json")
    if full["overall"]["count"] != 1000 or sha256(evaluation / "summary.json") != receipt["summary_sha256"]:
        raise ValueError("A verified, complete baseline is required first")
    if (sha256(evaluation / "metrics.csv") != receipt["metrics_sha256"]
            or binding_hash != receipt["binding_sha256"]
            or binding_hash != full["binding_sha256"]):
        raise ValueError("Baseline binding/CSV integrity failure")
    project = root / "project"
    data = root / "data/SIED_dataset"
    train = read_pair_manifest(data, project / "manifests/enhancement/canon_train.txt")
    val = read_pair_manifest(data, project / "manifests/enhancement/canon_val.txt")
    for split in ("train", "val"):
        if sha256(project / f"manifests/enhancement/canon_{split}.txt") != binding["split_manifests_sha256"][split]:
            raise ValueError("Research manifest changed after baseline evaluation")
    if len(train) != 832 or len(val) != 92 or {r.key for r in train} & {r.key for r in val}:
        raise ValueError("Invalid research splits")
    paths = []
    for split, records in (("train", train), ("val", val)):
        for index, record in enumerate(records):
            name = f"{index:04d}_{record.key}"
            row = json.loads((evaluation / "rows" / split / f"{name}.json").read_text())
            if (row["binding_sha256"] != binding_hash or row["key"] != record.key
                    or row["index"] != index or row["split"] != split):
                raise ValueError("Research cache row does not match its baseline binding")
            pair = evaluation / "color_pairs" / split / f"{name}.npy"
            if sha256(pair) != row["color_pair_sha256"]:
                raise ValueError(f"Color cache integrity failure: {name}")
            if split == "train":
                paths.append(pair)
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    random.seed(20261024)
    device = torch.device("cuda")
    base_model, _ = load_refiner(root / "final/best.pt", device)
    selection = json.loads((root / "final/selection.json").read_text())
    protocol = selection["frozen_protocol"]
    if sha256(root / "final/best.pt") != protocol["refiner_checkpoint_sha256"]:
        raise ValueError("Baseline weight changed")
    pilot_indices = sorted(random.Random(20261024).sample(range(92), 12))
    pilot = [val[i] for i in pilot_indices]
    val_index = {record.key: index for index, record in enumerate(val)}
    lpips = LPIPSMetric("cuda")
    spec = RawSpec()
    started = time.time()
    write_json(output / "experiment_started.json", {
        "started_unix": started, "max_steps": args.steps, "training_images": 832,
        "pilot_keys": [r.key for r in pilot], "seed": 20261024,
        "baseline_binding_sha256": sha256(evaluation / "binding.json"),
        "refiner_sha256": sha256(root / "final/best.pt"),
        "test_used": False, "max_training_wall_minutes": 45,
        "lut": "17 vertices, 3 bases, bounded 0.12 residual, area512 training",
        "code_sha256": {p: sha256(project / p) for p in (
            "cloud/research_sied_color.py", "src/dip_course/enhancement/adaptive_lut.py",
            "src/dip_course/enhancement/efficiency.py")},
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "precision": "baseline FP16 unchanged; LUT FP32; fusedAdamW",
        "caveat": "Internal val was seen by the official base pretraining; not unbiased generalization",
    })

    def base_prediction(record, structural_model):
        index = val_index[record.key]
        path = evaluation / "base_float32/val" / f"{index:04d}_{record.key}.npy"
        metadata = json.loads((evaluation / "rows/val" / f"{index:04d}_{record.key}.json").read_text())
        if sha256(path) != metadata["base_float32_sha256"]:
            raise ValueError("Native base cache integrity failure")
        array = np.load(path, allow_pickle=False)
        image = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
        raw = torch.from_numpy(load_packed_raw(record.low_raw, spec)).unsqueeze(0).cuda()
        with torch.inference_mode(), _autocast_context(device, protocol["precision"]):
            result = apply_raw_guided_refiner_tiled(
                structural_model, image, raw, tile_size=protocol["refiner_tile_size"],
                overlap=protocol["refiner_overlap"])
        return result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def evaluate(records, *, lut=None, structural_model=None, label):
        rows = []
        for record in records:
            cache = output / "pilot_baseline" / f"{record.key}.npy"
            if structural_model is None and cache.exists():
                array = np.load(cache, allow_pickle=False)
            else:
                array = base_prediction(record, structural_model or base_model)
                if structural_model is None and record.key in {r.key for r in pilot}:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, array, allow_pickle=False)
            baseline_array = array
            if lut is not None:
                tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                array = apply_color_lut_chunked(lut, tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
            target = load_rgb(record.gt_rgb)
            if structural_model is None:
                index = val_index[record.key]
                reference = json.loads((evaluation / "rows/val" /
                                        f"{index:04d}_{record.key}.json").read_text())
                difference = abs(psnr_rgb(baseline_array, target) - reference["psnr"])
                if difference > 0.001:
                    raise ValueError(f"Cache replay differs from frozen baseline: {record.key}, {difference} dB")
            row = {"split": "val", "key": record.key, "psnr": psnr_rgb(array, target),
                   "ssim": ssim_rgb(array, target), **color_fidelity_metrics(array, target),
                   "lpips": lpips(array, target)}
            rows.append(row)
            if len(records) == 92:
                save_rgb(output / label / "images" / f"{record.key}.png", array)
        summary = summarize(rows)
        write_json(output / label / "metrics.json", rows)
        write_json(output / label / "summary.json", summary)
        print(json.dumps({"evaluation": label, **summary}), flush=True)
        return summary

    baseline = evaluate(pilot, label="pilot_baseline")
    tlc = copy.deepcopy(base_model)
    enable_local_channel_means(tlc, rgb_window=512)
    tlc_summary = evaluate(pilot, structural_model=tlc, label="tlc512_pilot")
    tlc_pass = safe_gain(tlc_summary, baseline)
    del tlc

    lut = AdaptiveColorLUT().cuda()
    initial_state = copy.deepcopy(lut.state_dict())
    optimizer = torch.optim.AdamW(lut.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    loader = DataLoader(ColorPairs(paths), batch_size=16, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    best = baseline["psnr_mean"]
    best_step, stale, step = 0, 0, 0
    best_state = initial_state
    training_started = time.time()
    last_finished = time.perf_counter()
    with (output / "training.jsonl").open("a") as stream:
        while step < args.steps:
            for image, target in loader:
                compute_started = time.perf_counter()
                lut.train()
                image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = lut(image)
                loss = mean_image_psnr_loss(prediction, target) + 0.1 * lut.regularization()
                if not torch.isfinite(loss):
                    raise FloatingPointError("LUT loss became nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lut.parameters(), 1)
                optimizer.step()
                scheduler.step()
                step += 1
                row = {"step": step, "loss": float(loss.item()), "time_unix": time.time(),
                       "compute_step_seconds": time.perf_counter() - compute_started,
                       "data_and_step_seconds": time.perf_counter() - last_finished}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if step % 250 == 0 or step == args.steps:
                    lut.eval()
                    candidate = evaluate(pilot, lut=lut, label=f"lut_pilot_step{step}")
                    accepted = safe_gain(candidate, baseline) and candidate["psnr_mean"] >= best + 0.01
                    if accepted:
                        best, best_step, stale = candidate["psnr_mean"], step, 0
                        best_state = copy.deepcopy(lut.state_dict())
                        _save_checkpoint(output / "best_lut.pt", {
                            "model": best_state, "model_type": "adaptive_color_lut",
                            "model_config": lut.config, "step": step, "metrics": candidate,
                            "baseline_binding_sha256": binding_hash,
                            "baseline_refiner_sha256": protocol["refiner_checkpoint_sha256"],
                            "torch_version": str(torch.__version__), "seed": 20261024})
                    else:
                        stale += 1
                    write_json(output / "progress.json", {
                        "state": "training", "step": step, "budget_steps": args.steps,
                        "best_step": best_step, "pilot_psnr": candidate["psnr_mean"],
                        "accepted": accepted, "stale": stale})
                stop = (step >= args.steps or (step >= 750 and stale >= 3)
                        or time.time() - training_started > 45 * 60)
                if stop:
                    break
                last_finished = time.perf_counter()
            if stop:
                break
    full_baseline = full["splits"]["val"]
    outcomes = {"tlc512_pilot_passed": tlc_pass, "lut_best_step": best_step,
                "training_steps": step, "test_evaluated": False,
                "baseline_full_val": full_baseline}
    if best_step:
        outcomes["best_lut_sha256"] = sha256(output / "best_lut.pt")
    if tlc_pass:
        tlc = copy.deepcopy(base_model)
        enable_local_channel_means(tlc, rgb_window=512)
        metrics = evaluate(val, structural_model=tlc, label="tlc512_full_val")
        outcomes["tlc512_full_val"] = metrics
        outcomes["tlc512_full_passed"] = safe_gain(metrics, full_baseline)
        del tlc
    if best_step:
        lut.load_state_dict(best_state)
        lut.eval()
        metrics = evaluate(val, lut=lut, label="lut_full_val")
        outcomes["lut_full_val"] = metrics
        outcomes["lut_full_passed"] = safe_gain(metrics, full_baseline)
    write_json(output / "result.json", outcomes)
    write_json(output / "progress.json", {"state": "complete", **outcomes})
    print(json.dumps(outcomes, indent=2), flush=True)


if __name__ == "__main__":
    main()
