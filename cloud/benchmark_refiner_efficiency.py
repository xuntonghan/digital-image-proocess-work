"""Bounded synthetic kernel benchmark; no image-quality or training-gain claim."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dip_course.cli._common import write_json
from dip_course.cli.infer_official_sied import load_refiner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("steps must be within [1, 100]")
    root = args.root
    lock = (root / "logs/gpu_work.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = root / "research/efficiency"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cases = [("nchw_fp16_default_adamw", False, False, torch.float16),
             ("channels_last_fp16_default_adamw", True, False, torch.float16),
             ("channels_last_fp16_fused_adamw", True, True, torch.float16),
             ("channels_last_bf16_fused_adamw", True, True, torch.bfloat16)]
    rows = []
    reference = None
    for name, channels_last, fused, dtype in cases:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(20261024)
        model, _ = load_refiner(root / "final/best.pt", torch.device("cuda"))
        for parameter in model.photometric.parameters():
            parameter.requires_grad_(False)
        if channels_last:
            model.to(memory_format=torch.channels_last)
        corrected = torch.rand(1, 3, 512, 512, device="cuda")
        guidance = torch.rand(1, 4, 512, 512, device="cuda")
        target = (corrected * 0.9 + 0.03).clamp(0, 1)
        if channels_last:
            corrected = corrected.contiguous(memory_format=torch.channels_last)
            guidance = guidance.contiguous(memory_format=torch.channels_last)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, fused=fused)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16,
                                      init_scale=1024)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            initial = (corrected + model.restoration_residual(corrected, guidance)).clamp(0, 1)
            if reference is None:
                reference = initial.float().cpu()
            initial_difference = float((initial.float().cpu() - reference).abs().max())
        del initial
        times, scales = [], []
        torch.cuda.reset_peak_memory_stats()
        started_case = time.perf_counter()
        for step in range(5 + args.steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                prediction = (corrected + model.restoration_residual(corrected, guidance)).clamp(0, 1)
                loss = F.mse_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if step >= 5:
                times.append(elapsed)
                scales.append(scaler.get_scale() >= previous_scale)
            if time.perf_counter() - started_case > 240:
                raise TimeoutError(f"Benchmark exceeded 240 seconds for {name}")
        row = {
            "case": name, "batch_size": 1, "rgb_crop": 512,
            "measurement_steps": args.steps, "successful_updates": sum(scales),
            "seconds_per_step_median": statistics.median(times),
            "seconds_per_step_mean": statistics.fmean(times),
            "images_per_second": 1 / statistics.fmean(times),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "initial_max_abs_vs_reference": initial_difference,
            "synthetic_final_mse": float(loss.item()),
        }
        rows.append(row)
        write_json(output / "benchmark_partial.json", rows)
        print(json.dumps(row), flush=True)
        del model, optimizer, scaler, parameters, prediction, loss, corrected, guidance, target
    baseline = rows[0]["seconds_per_step_mean"]
    for row in rows:
        row["speedup_vs_baseline"] = baseline / row["seconds_per_step_mean"]
    write_json(output / "benchmark.json", {
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "purpose": "Synthetic structural forward/backward/optimizer only; excludes data/EMA/validation.",
        "quality_caveat": "No PSNR improvement claim; layout/precision need real-image validation.",
        "cases": rows,
    })


if __name__ == "__main__":
    main()
