"""Bounded, matched global-vs-factorized LUT experiment; train/val only.

Run after research_sied_color.py, never alongside the frozen 1000 evaluation.
No test GT, repeated test inference, or replacement of the sealed final model.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_sied1000 import sha256, summarize
from research_sied_color import ColorPairs, safe_gain
from torch.utils.data import DataLoader

from dip_course.cli._common import write_json
from dip_course.cli.infer_official_sied import _autocast_context, load_refiner
from dip_course.cli.train_sied_restorer import _save_checkpoint
from dip_course.enhancement.adaptive_lut import (
    AdaptiveColorLUT,
    apply_color_lut_chunked,
)
from dip_course.enhancement.efficiency import mean_image_psnr_loss
from dip_course.enhancement.factorized_lut import PairwiseLowRankLUT
from dip_course.enhancement.refiner import apply_raw_guided_refiner_tiled
from dip_course.io import (
    RawSpec,
    load_packed_raw,
    load_rgb,
    read_pair_manifest,
    save_rgb,
)
from dip_course.metrics import LPIPSMetric, color_fidelity_metrics, psnr_rgb, ssim_rgb

SEED = 20261024


class TrainingBudgetExhausted(Exception):
    pass


class ResidentPairs:
    """FP32 train-only pairs resident once, without another quantization/resize."""

    def __init__(self, paths: list[Path], device="cuda"):
        if not paths:
            raise ValueError("No training pairs")
        device = torch.device(device)
        required = len(paths) * 2 * 3 * 288 * 512 * 4
        if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < required + 6 * 1024**3:
            raise RuntimeError("Insufficient GPU headroom for resident train pairs; no silent fallback")
        started = time.perf_counter()
        self.data = torch.empty((len(paths), 2, 3, 288, 512), dtype=torch.float32, device=device)
        dataset = ColorPairs(paths)
        for index in range(len(paths)):
            pair = torch.stack(dataset[index])
            if not torch.isfinite(pair).all() or pair.min() < 0 or pair.max() > 1:
                raise ValueError("Nonfinite/out-of-range color pair")
            self.data[index].copy_(pair)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.preload_seconds = time.perf_counter() - started
        self.bytes = required

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.data.device)
        pair = self.data.index_select(0, indices)
        return pair[:, 0].contiguous(), pair[:, 1].contiguous()


def training_batches(count, *, steps, batch=16, seed=SEED):
    """An identical, private RNG stream in both arms, including partial batches."""
    if count < 1 or steps < 1 or batch < 1:
        raise ValueError("Invalid batch schedule")
    generator = torch.Generator().manual_seed(seed)
    emitted = 0
    while emitted < steps:
        permutation = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count, batch):
            yield permutation[start:start + batch]
            emitted += 1
            if emitted >= steps:
                break


def benchmark(paths, resident, output, controller):
    """Same real train batches/precision/loss/optimizer, discard all updates."""
    indices = list(training_batches(len(paths), steps=20))
    reports = {}
    control_states = []
    for mode in ("global_pinned_loader", "global_resident", "factorized_resident"):
        torch.manual_seed(SEED)
        model = (PairwiseLowRankLUT() if mode.startswith("factorized") else AdaptiveColorLUT()).cuda()
        model.controller.load_state_dict(controller)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True, weight_decay=1e-4)
        if mode.endswith("loader"):
            loader = DataLoader(ColorPairs(paths), batch_sampler=indices, num_workers=2,
                                persistent_workers=False, pin_memory=True,
                                multiprocessing_context="spawn")
            iterator = iter(loader)
        else:
            iterator = (resident.batch(batch) for batch in indices)
        times = []
        first = None
        for step in range(20):
            torch.cuda.synchronize()
            started = time.perf_counter()
            image, target = next(iterator)
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            if step == 0:
                first = prediction.detach().cpu()
            loss = mean_image_psnr_loss(prediction, target) + 0.1 * model.regularization()
            if not torch.isfinite(loss):
                raise FloatingPointError("Benchmark loss became nonfinite")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if elapsed > 120:
                raise TimeoutError("Bounded real-data benchmark step exceeded 120 seconds")
            if step >= 4:
                times.append(elapsed)
        reports[mode] = {"mean_data_and_step_seconds": statistics.fmean(times),
                         "median_data_and_step_seconds": statistics.median(times),
                         "measured_steps": len(times), "warmup_steps": 4,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "finite_parameters": all(torch.isfinite(p).all().item() for p in model.parameters())}
        if mode.startswith("global"):
            control_states.append((first, copy.deepcopy(model.state_dict())))
        del iterator, optimizer, model
    first_a, state_a = control_states[0]
    first_b, state_b = control_states[1]
    reports["loader_vs_resident"] = {
        "first_prediction_max_abs_difference": float((first_a - first_b).abs().max()),
        "final_parameter_max_abs_difference": max(float((state_a[k] - state_b[k]).abs().max())
                                                   for k in state_a),
        "resident_preload_seconds": resident.preload_seconds, "resident_bytes": resident.bytes,
        "caveat": "Real train batches; short benchmark excludes native validation and preload amortization"}
    reports["same_model_measured_throughput_ratio"] = (
        reports["global_pinned_loader"]["mean_data_and_step_seconds"] /
        reports["global_resident"]["mean_data_and_step_seconds"])
    write_json(output / "real_data_benchmark.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("Each paired arm is limited to 1..1000 steps")
    root = args.root.resolve()
    output = root / "research/iccv_pairwise"
    output.mkdir(parents=True, exist_ok=True)
    lock = (root / "logs/gpu_work.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (output / "experiment_started.json").exists():
        raise RuntimeError("ICCV experiment already started; never overwrite/restart implicitly")
    if not (root / "research/color_lut_tlc/result.json").is_file():
        raise RuntimeError("The preceding bounded experiment must finish first")
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise RuntimeError("Insufficient disk reserve; do not expand or delete original data")
    evaluation = root / "evaluation1000_native"
    receipt = json.loads((evaluation / "verification.json").read_text())
    full = json.loads((evaluation / "summary.json").read_text())
    binding = json.loads((evaluation / "binding.json").read_text())
    binding_hash = sha256(evaluation / "binding.json")
    if (full["overall"]["count"] != 1000 or full["binding_sha256"] != binding_hash
            or receipt["binding_sha256"] != binding_hash
            or receipt["summary_sha256"] != sha256(evaluation / "summary.json")
            or receipt["metrics_sha256"] != sha256(evaluation / "metrics.csv")):
        raise ValueError("Incomplete or changed frozen evaluation")
    project, data = root / "project", root / "data/SIED_dataset"
    splits, paths = {}, []
    for split, count in (("train", 832), ("val", 92)):
        manifest = project / f"manifests/enhancement/canon_{split}.txt"
        if sha256(manifest) != binding["split_manifests_sha256"][split]:
            raise ValueError("Research manifest changed")
        splits[split] = read_pair_manifest(data, manifest)
        if len(splits[split]) != count:
            raise ValueError("Wrong research split size")
        for index, record in enumerate(splits[split]):
            name = f"{index:04d}_{record.key}"
            row = json.loads((evaluation / "rows" / split / f"{name}.json").read_text())
            if (row["binding_sha256"] != binding_hash or row["key"] != record.key
                    or row["split"] != split or row["index"] != index):
                raise ValueError("Research row integrity failure")
            if split == "train":
                path = evaluation / "color_pairs/train" / f"{name}.npy"
                if sha256(path) != row["color_pair_sha256"]:
                    raise ValueError("Train cache integrity failure")
                paths.append(path)
    if {r.key for r in splits["train"]} & {r.key for r in splits["val"]}:
        raise ValueError("Train/val overlap")
    if sha256(root / "final/best.pt") != binding["refiner_sha256"]:
        raise ValueError("Frozen refiner changed")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    controller = copy.deepcopy(AdaptiveColorLUT().controller.state_dict())
    val = splits["val"]
    pilot = [val[i] for i in sorted(random.Random(SEED).sample(range(92), 12))]
    val_index = {r.key: index for index, r in enumerate(val)}
    write_json(output / "experiment_started.json", {
        "started_unix": time.time(), "seed": SEED, "steps_per_arm": args.steps,
        "max_training_wall_minutes_per_arm": 15, "test_used": False,
        "training_keys": [r.key for r in splits["train"]],
        "pilot_keys": [r.key for r in pilot], "baseline_binding_sha256": binding_hash,
        "arms": ["global3d", "pairwise_rank4"],
        "controlled": "same controller initialization, batches, FP32, fusedAdamW, loss, native protocol",
        "scope": "Independent low-rank pairwise ablation, not full spatial SVDLUT reproduction",
        "code_sha256": {p: sha256(project / p) for p in (
            "cloud/research_sied_iccv.py", "cloud/research_sied_color.py",
            "cloud/evaluate_sied1000.py", "src/dip_course/enhancement/factorized_lut.py",
            "src/dip_course/enhancement/adaptive_lut.py", "src/dip_course/enhancement/efficiency.py")},
        "caveat": "Official base pretraining saw these validation images; not unbiased generalization"})
    resident = ResidentPairs(paths)
    speed = benchmark(paths, resident, output, controller)
    if (speed["loader_vs_resident"]["first_prediction_max_abs_difference"] > 1e-6
            or speed["loader_vs_resident"]["final_parameter_max_abs_difference"] > 1e-5):
        raise ValueError("Resident training differs from matched loader; do not claim equivalent speedup")
    base_model, _ = load_refiner(root / "final/best.pt", torch.device("cuda"))
    protocol = binding["protocol"]
    lpips = LPIPSMetric("cuda")

    def evaluate(records, model, label, deadline=None):
        rows, hashes = [], {}
        for record in records:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TrainingBudgetExhausted
            index = val_index[record.key]
            name = f"{index:04d}_{record.key}"
            reference = json.loads((evaluation / "rows/val" / f"{name}.json").read_text())
            cache = root / "research/color_lut_tlc/pilot_baseline" / f"{record.key}.npy"
            if record in pilot and cache.is_file():
                array = np.load(cache, allow_pickle=False)
                if array.shape != (2160, 3840, 3) or array.dtype != np.float32:
                    raise ValueError("Incorrect native pilot cache")
                hashes[record.key] = sha256(cache)
            else:
                path = evaluation / "base_float32/val" / f"{name}.npy"
                if sha256(path) != reference["base_float32_sha256"]:
                    raise ValueError("Incorrect native base hash")
                array = np.load(path, allow_pickle=False)
                image = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                raw = torch.from_numpy(load_packed_raw(record.low_raw, RawSpec())).unsqueeze(0).cuda()
                with torch.inference_mode(), _autocast_context(torch.device("cuda"), protocol["precision"]):
                    result = apply_raw_guided_refiner_tiled(
                        base_model, image, raw, tile_size=protocol["refiner_tile_size"],
                        overlap=protocol["refiner_overlap"])
                array = result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            target = load_rgb(record.gt_rgb)
            if abs(psnr_rgb(array, target) - reference["psnr"]) > 0.001:
                raise ValueError("Frozen native replay mismatch")
            if model is not None:
                image = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                array = apply_color_lut_chunked(model, image).squeeze(0).permute(1, 2, 0).cpu().numpy()
            rows.append({"split": "val", "key": record.key, "psnr": psnr_rgb(array, target),
                         "ssim": ssim_rgb(array, target), **color_fidelity_metrics(array, target),
                         "lpips": lpips(array, target)})
            if len(records) == 92:
                save_rgb(output / label / "images" / f"{record.key}.png", array)
        summary = summarize(rows)
        write_json(output / label / "metrics.json", rows)
        write_json(output / label / "summary.json", summary)
        write_json(output / label / "reused_pilot_cache_sha256.json", hashes)
        print(json.dumps({"evaluation": label, **summary}), flush=True)
        return summary

    baseline = evaluate(pilot, None, "pilot_baseline")
    outcomes = {"test_evaluated": False, "baseline_binding_sha256": binding_hash,
                "baseline_full_val": full["splits"]["val"], "arms": {}}
    for arm in ("global3d", "pairwise_rank4"):
        torch.manual_seed(SEED)
        model = (AdaptiveColorLUT() if arm == "global3d" else PairwiseLowRankLUT()).cuda()
        model.controller.load_state_dict(controller)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
        best, best_step, stale = baseline["psnr_mean"], 0, 0
        started = time.perf_counter()
        deadline = started + 15 * 60
        times = []
        checkpoint = output / arm / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with (output / arm / "training.jsonl").open("a") as stream:
            for step, indices in enumerate(training_batches(len(paths), steps=args.steps), 1):
                torch.cuda.synchronize()
                step_started = time.perf_counter()
                image, target = resident.batch(indices)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(image)
                loss = mean_image_psnr_loss(prediction, target) + 0.1 * model.regularization()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Paired LUT loss became nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - step_started
                times.append(elapsed)
                stream.write(json.dumps({"step": step, "loss": float(loss), "batch_size": len(indices),
                                         "data_and_step_seconds": elapsed, "time_unix": time.time()}) + "\n")
                stream.flush()
                if step % 250 == 0 or step == args.steps:
                    model.eval()
                    try:
                        candidate = evaluate(pilot, model, f"{arm}/pilot_step{step}", deadline=deadline)
                    except TrainingBudgetExhausted:
                        break
                    accepted = safe_gain(candidate, baseline) and candidate["psnr_mean"] >= best + 0.01
                    if accepted:
                        best, best_step, stale = candidate["psnr_mean"], step, 0
                        _save_checkpoint(checkpoint, {
                            "model": model.state_dict(), "model_type": arm, "model_config": model.config,
                            "step": step, "metrics": candidate, "seed": SEED,
                            "baseline_binding_sha256": binding_hash,
                            "baseline_refiner_sha256": binding["refiner_sha256"]})
                    else:
                        stale += 1
                    write_json(output / "progress.json", {"state": "training", "arm": arm,
                               "step": step, "best_step": best_step, "pilot": candidate,
                               "accepted": accepted, "stale": stale})
                if (step >= 750 and stale >= 3) or time.perf_counter() >= deadline:
                    break
        arm_result = {"steps": step, "best_step": best_step,
                      "training_including_validation_seconds": time.perf_counter() - started,
                      "mean_data_and_step_seconds": statistics.fmean(times),
                      "parameters": sum(p.numel() for p in model.parameters())}
        if best_step:
            saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
            model.load_state_dict(saved["model"])
            model.eval()
            metrics = evaluate(val, model, f"{arm}/full_val")
            arm_result.update({"full_val": metrics, "full_passed": safe_gain(metrics, full["splits"]["val"]),
                               "checkpoint_sha256": sha256(checkpoint)})
        outcomes["arms"][arm] = arm_result
        write_json(output / "partial_result.json", outcomes)
        del model, optimizer, scheduler
    accepted = [(name, result) for name, result in outcomes["arms"].items() if result.get("full_passed")]
    outcomes["best_paired_candidate"] = (max(accepted, key=lambda item: item[1]["full_val"]["psnr_mean"])[0]
                                          if accepted else None)
    outcomes["sealed_baseline_replaced"] = False
    write_json(output / "result.json", outcomes)
    write_json(output / "progress.json", {"state": "complete", **outcomes})
    print(json.dumps(outcomes, indent=2), flush=True)


if __name__ == "__main__":
    main()
