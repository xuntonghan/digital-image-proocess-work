"""Native SIED refinement with identical full-scene photometry in train/test.

Cached PNG inputs are blind official SIED ensembles. Their full-scene photometric
correction is computed once, saved as float16 arrays, and cropped only afterwards.
The photometric network stays frozen. All checkpoint decisions use native tiled
validation; neither thumbnail PSNR nor a test image participates in selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dip_course.cli._common import (
    choose_torch_device,
    file_sha256,
    seed_everything,
    write_json,
)
from dip_course.cli.train_sied_restorer import (
    _loss_components,
    _save_checkpoint,
    _update_ema,
)
from dip_course.cli.train_sied_restorer_native import load_restorer_checkpoint
from dip_course.enhancement.refiner import (
    RawGuidedNAFRefiner,
    _blend_window,
    _tile_starts,
)
from dip_course.io import (
    PairRecord,
    RawSpec,
    load_packed_raw,
    load_rgb,
    pack_rggb,
    read_pair_manifest,
)
from dip_course.metrics import color_fidelity_metrics, psnr_rgb, ssim_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root", "train-manifest", "val-manifest", "prediction-dir",
        "cache-dir", "initial-checkpoint", "output-dir", "validation-output-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--val-prediction-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--min-steps", type=int, default=2500)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--min-psnr-improvement", type=float, default=0.01)
    parser.add_argument("--max-color-regression", type=float, default=0.02)
    parser.add_argument("--max-ssim-regression", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true")
    return parser


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def photometric_fingerprint(model: RawGuidedNAFRefiner) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.photometric.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr(model.photometric.context_side).encode())
    return digest.hexdigest()


def prediction_index(directory: str | Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(Path(directory).glob("*.png")):
        key = path.stem.split("_", 1)[-1]
        if key in indexed:
            raise ValueError(f"Duplicate cached prediction key: {key}")
        indexed[key] = path
    return indexed


def check_split_disjoint(
    train: list[PairRecord], validation: list[PairRecord]
) -> None:
    for attribute in ("low_raw", "gt_rgb", "key"):
        train_keys = [getattr(record, attribute) for record in train]
        val_keys = [getattr(record, attribute) for record in validation]
        if len(set(train_keys)) != len(train_keys) or len(set(val_keys)) != len(val_keys):
            raise ValueError(f"Duplicate {attribute} in a split")
        if set(train_keys) & set(val_keys):
            raise ValueError(f"Train/validation {attribute} overlap")


def normalized_raw_mean(path: Path, spec: RawSpec) -> float:
    if path.stat().st_size != spec.expected_bytes:
        raise ValueError(f"Invalid RAW byte count: {path}")
    raw = np.memmap(path, dtype=np.uint16, mode="r", shape=(spec.height, spec.width))
    total = 0.0
    for top in range(0, spec.height, 128):
        stripe = np.asarray(raw[top:top + 128], dtype=np.float32)
        stripe = np.clip(
            (stripe - spec.black_level) / (spec.white_level - spec.black_level), 0, 1
        )
        total += float(stripe.sum(dtype=np.float64))
    return total / spec.sample_count


@torch.inference_mode()
def prepare_corrected_cache(
    model: RawGuidedNAFRefiner,
    records: list[PairRecord],
    predictions: dict[str, Path],
    directory: Path,
    *,
    manifest_sha256: str,
    spec: RawSpec,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    """Build a provenance-bound, atomic, resumable cache without reading GT."""
    directory.mkdir(parents=True, exist_ok=True)
    missing = [record.key for record in records if record.key not in predictions]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} blind predictions: {missing[:5]}")
    source_signature = hashlib.sha256()
    for record in records:
        source = predictions[record.key]
        source_signature.update(
            f"{record.key}:{source.stat().st_size}:{source.stat().st_mtime_ns}".encode()
        )
    provenance = {
        "manifest_sha256": manifest_sha256,
        "photometric_sha256": photometric_fingerprint(model),
        "source_signature": source_signature.hexdigest(),
        "shape": [spec.height, spec.width, 3],
        "dtype": "float16",
        "photometric_context": "full_native_scene",
        "source_quantization": "blind_ensemble_rgb_png_uint8",
        "gt_used_in_cache": False,
    }
    metadata_path = directory / "provenance.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != provenance:
            raise ValueError(f"Corrected cache provenance mismatch: {directory}")
    else:
        write_json(metadata_path, provenance)
    pending = []
    expected_shape = (spec.height, spec.width, 3)
    for record in records:
        path = directory / f"{record.key}.npy"
        if path.is_file():
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            if cached.shape != expected_shape or cached.dtype != np.float16:
                raise ValueError(f"Invalid corrected cache entry: {path}")
        else:
            pending.append(record.key)
    bytes_required = len(pending) * spec.sample_count * 3 * 2
    reserve = 3 * 1024**3
    available = shutil.disk_usage(directory).free
    if bytes_required + reserve > available:
        raise OSError(
            f"Corrected cache needs {bytes_required / 1024**3:.2f} GiB plus "
            f"3 GiB reserve, but only {available / 1024**3:.2f} GiB is free"
        )
    means_path = directory / "raw_means.json"
    means = json.loads(means_path.read_text()) if means_path.exists() else {}
    model.eval()
    for index, record in enumerate(records):
        output = directory / f"{record.key}.npy"
        if not output.is_file():
            with Image.open(predictions[record.key]) as image:
                if image.size != (spec.width, spec.height):
                    raise ValueError(f"Prediction is not native resolution: {image.size}")
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            base = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            with autocast_context(device, precision):
                corrected = model.photometric(base)
            result = corrected.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
            if not np.isfinite(result).all():
                raise FloatingPointError(f"Non-finite full-scene correction: {record.key}")
            temporary = output.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
            temporary.replace(output)
        if record.key not in means:
            means[record.key] = normalized_raw_mean(record.low_raw, spec)
        if index % 25 == 0 or index + 1 == len(records):
            write_json(means_path, means)
            print(json.dumps({"stage": "full_scene_corrected_cache", "split": directory.name,
                              "completed": index + 1, "total": len(records)}), flush=True)
    return {str(key): float(value) for key, value in means.items()}


class CorrectedNativeDataset(Dataset):
    """Crop an already corrected full scene, keeping RAW and RGB aligned."""

    def __init__(
        self, records: list[PairRecord], cache: Path, raw_means: dict[str, float],
        spec: RawSpec, *, crop_size: int = 512, augment: bool = True,
    ) -> None:
        if crop_size <= 0 or crop_size % 2 or crop_size > min(spec.height, spec.width):
            raise ValueError("crop_size must be even and fit inside the native image")
        self.records, self.cache, self.raw_means = records, cache, raw_means
        self.spec, self.crop_size, self.augment = spec, crop_size, augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record, size, spec = self.records[index], self.crop_size, self.spec
        if self.augment:
            top = 2 * random.randint(0, (spec.height - size) // 2)
            left = 2 * random.randint(0, (spec.width - size) // 2)
        else:
            top, left = 2 * ((spec.height - size) // 4), 2 * ((spec.width - size) // 4)
        corrected = np.load(self.cache / f"{record.key}.npy", mmap_mode="r", allow_pickle=False)
        corrected = np.asarray(corrected[top:top + size, left:left + size], dtype=np.float32)
        with Image.open(record.gt_rgb) as image:
            if image.size != (spec.width, spec.height):
                raise ValueError(f"GT is not native resolution: {record.gt_rgb}")
            target = np.asarray(image.convert("RGB").crop((left, top, left + size, top + size)),
                                dtype=np.float32) / 255.0
        if record.low_raw.stat().st_size != spec.expected_bytes:
            raise ValueError(f"Invalid RAW byte count: {record.low_raw}")
        raw_map = np.memmap(record.low_raw, dtype=np.uint16, mode="r", shape=(spec.height, spec.width))
        raw = np.asarray(raw_map[top:top + size, left:left + size], dtype=np.float32)
        raw = pack_rggb(np.clip((raw - spec.black_level) / (spec.white_level - spec.black_level), 0, 1))
        corrected, target = corrected.transpose(2, 0, 1), target.transpose(2, 0, 1)
        if self.augment and random.random() < 0.5:
            corrected, target, raw = corrected[:, :, ::-1], target[:, :, ::-1], raw[:, :, ::-1]
        if self.augment and random.random() < 0.5:
            corrected, target, raw = corrected[:, ::-1, :], target[:, ::-1, :], raw[:, ::-1, :]
        return {
            "corrected": torch.from_numpy(np.ascontiguousarray(corrected)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "raw": torch.from_numpy(np.ascontiguousarray(raw)).float(),
            "raw_mean": torch.tensor(self.raw_means[record.key], dtype=torch.float32),
            "key": record.key,
        }


def restore_corrected(
    model: RawGuidedNAFRefiner, corrected: torch.Tensor, raw: torch.Tensor,
    raw_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Structural forward: deliberately never invokes model.photometric."""
    guidance = model.raw_guidance(raw, size=corrected.shape[-2:], raw_gain=raw_gain)
    return (corrected + model.restoration_residual(corrected, guidance)).clamp(0, 1)


@torch.inference_mode()
def restore_corrected_tiled(
    model: RawGuidedNAFRefiner, corrected: torch.Tensor, raw: torch.Tensor,
    *, tile_size: int = 512, overlap: int = 64,
) -> torch.Tensor:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid structural tile settings")
    guidance = model.raw_guidance(raw, size=corrected.shape[-2:])
    height, width = corrected.shape[-2:]
    if max(height, width) <= tile_size:
        return (corrected + model.restoration_residual(corrected, guidance)).clamp(0, 1)
    total = torch.zeros_like(corrected)
    weights = torch.zeros((1, 1, height, width), device=corrected.device, dtype=corrected.dtype)
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            residual = model.restoration_residual(
                corrected[:, :, top:bottom, left:right], guidance[:, :, top:bottom, left:right]
            )
            window = _blend_window(bottom - top, right - left,
                                   device=corrected.device, dtype=corrected.dtype)
            total[:, :, top:bottom, left:right] += residual * window
            weights[:, :, top:bottom, left:right] += window
    return (corrected + total / weights.clamp_min(1e-6)).clamp(0, 1)


@torch.inference_mode()
def validate_native_cache(
    model: RawGuidedNAFRefiner, records: list[PairRecord], cache: Path, spec: RawSpec,
    device: torch.device, *, precision: str, tile_size: int, overlap: int,
    include_ssim: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for record in records:
        corrected = np.load(cache / f"{record.key}.npy", allow_pickle=False).astype(np.float32)
        raw = torch.from_numpy(load_packed_raw(record.low_raw, spec)).unsqueeze(0).to(device)
        base = torch.from_numpy(corrected.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        with autocast_context(device, precision):
            refined = restore_corrected_tiled(model, base, raw, tile_size=tile_size, overlap=overlap)
        prediction = refined.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        target = load_rgb(record.gt_rgb)
        row: dict[str, object] = {"key": record.key, "psnr": psnr_rgb(prediction, target)}
        row.update(color_fidelity_metrics(prediction, target))
        if include_ssim:
            row["ssim"] = ssim_rgb(prediction, target)
        rows.append(row)
    if not rows:
        raise ValueError("Native validation split is empty")
    summary: dict[str, object] = {
        "method": "official_sied_plus_full_scene_context_naf",
        "count": len(rows), "evaluation_resolution": [spec.width, spec.height],
        "base_source": "blind_ensemble_rgb_png_uint8",
        "corrected_cache_dtype": "float16", "blind_inference": True,
        "refiner_tile_size": tile_size, "refiner_overlap": overlap,
    }
    metric_names = [name for name in rows[0] if name != "key"]
    for name in metric_names:
        # Match the established project's full-validation summary field names.
        output_name = name if name.endswith("_mean") else f"{name}_mean"
        summary[output_name] = float(np.mean([float(row[name]) for row in rows]))
    return summary, rows


def passes_metric_guardrails(
    candidate: dict[str, object], baseline: dict[str, object],
    *, color_tolerance: float = 0.02, ssim_tolerance: float = 0.002,
) -> bool:
    if float(candidate["ssim_mean"]) < float(baseline["ssim_mean"]) - ssim_tolerance:
        return False
    for metric in ("delta_e2000_mean", "delta_e2000_p95_mean", "lightness_mae_mean",
                   "saturation_mae_mean", "hue_error_degrees_mean"):
        limit = max(float(baseline[metric]), 1e-8) * (1 + color_tolerance)
        if not math.isfinite(float(candidate[metric])) or float(candidate[metric]) > limit:
            return False
    return math.isfinite(float(candidate["psnr_mean"]))


def checkpoint_payload(
    model: RawGuidedNAFRefiner, config: dict[str, object], step: int,
    metrics: dict[str, object], *, tile_size: int, overlap: int,
) -> dict[str, object]:
    return {
        "model": model.state_dict(), "model_type": model.model_type,
        "model_config": config, "global_step": step, "metrics": metrics,
        "torch_version": str(torch.__version__),
        "context_training": "full_scene_photometric_then_native_crop",
        "recommended_inference": {"tile_size": tile_size, "overlap": overlap},
    }


def main() -> None:
    args = build_parser().parse_args()
    for name in ("max_steps", "validation_interval", "val_limit", "batch_size",
                 "crop_size", "tile_size", "early_stop_patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.crop_size % 2 or args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise ValueError("Invalid crop/tile dimensions")
    if args.min_steps < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer/early stopping settings")
    if not 0 < args.ema_decay < 1 or args.num_workers < 0:
        raise ValueError("Invalid EMA/data loader settings")
    if min(args.min_psnr_improvement, args.max_color_regression, args.max_ssim_regression) < 0:
        raise ValueError("Metric tolerances cannot be negative")
    seed_everything(args.seed)
    device = choose_torch_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"CUDA required, got {device}")
    model, metadata = load_restorer_checkpoint(args.initial_checkpoint, device)
    for parameter in model.photometric.parameters():
        parameter.requires_grad_(False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_records = read_pair_manifest(args.data_root, args.train_manifest)
    val_records = read_pair_manifest(args.data_root, args.val_manifest)
    check_split_disjoint(train_records, val_records)
    spec = RawSpec()
    cache_root = Path(args.cache_dir) / photometric_fingerprint(model)[:16]
    train_cache, val_cache = cache_root / "train", cache_root / "validation"
    train_means = prepare_corrected_cache(
        model, train_records, prediction_index(args.prediction_dir), train_cache,
        manifest_sha256=file_sha256(args.train_manifest), spec=spec,
        device=device, precision=args.precision,
    )
    prepare_corrected_cache(
        model, val_records, prediction_index(args.val_prediction_dir or args.prediction_dir),
        val_cache, manifest_sha256=file_sha256(args.val_manifest), spec=spec,
        device=device, precision=args.precision,
    )
    selected_indices = sorted(random.Random(args.seed + 19).sample(
        range(len(val_records)), min(args.val_limit, len(val_records))))
    fast_val = [val_records[index] for index in selected_indices]
    write_json(output / "run_environment.json", {
        "method": "full_scene_context_naf", "settings": vars(args),
        "initial_checkpoint": metadata, "train_pairs": len(train_records),
        "val_pairs": len(val_records), "quick_validation_keys": [r.key for r in fast_val],
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "val_manifest_sha256": file_sha256(args.val_manifest),
        "corrected_cache_root": str(cache_root), "torch": str(torch.__version__),
        "started_unix": time.time(), "photometric_frozen": True,
        "validation_caveat": "Official pretrained SIED saw official train split including refiner validation scenes; final test is disjoint.",
        "precision_caveat": "Training/validation base uses uint8 PNG, corrected cache float16; final blind inference uses fresh floating-point SIED output.",
    })
    dataset = CorrectedNativeDataset(train_records, train_cache, train_means, spec,
                                     crop_size=args.crop_size, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    validation_options = {"precision": args.precision, "tile_size": args.tile_size,
                          "overlap": args.tile_overlap}
    baseline, _ = validate_native_cache(ema, fast_val, val_cache, spec, device, **validation_options)
    write_json(output / "initial_native_validation.json", baseline)
    best_metrics, best_psnr = baseline, float(baseline["psnr_mean"])
    config = metadata["model_config"]
    step, stale = 0, 0
    state_path = output / "resume_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["online_model"])
        ema.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, stale, best_psnr = state["step"], state["stale"], state["best_psnr"]
        baseline, best_metrics = state["baseline"], state["best_metrics"]
        print(json.dumps({"stage": "resumed", "step": step}), flush=True)
    else:
        _save_checkpoint(output / "best.pt", checkpoint_payload(
            ema, config, 0, baseline, tile_size=args.tile_size, overlap=args.tile_overlap))
    stopped_early = False
    while step < args.max_steps and not stopped_early:
        model.train()
        for batch in loader:
            corrected = batch["corrected"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True)
            means = batch["raw_mean"].to(device, non_blocking=True).reshape(-1, 1, 1, 1)
            gain = (model.raw_target_mean / means.clamp_min(1e-8)).clamp(1, model.raw_max_gain)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                prediction = restore_corrected(model, corrected, raw, gain)
                loss, components = _loss_components(
                    prediction, target, corrected,
                    {"frozen_photometric": corrected.new_zeros(())}, profile="psnr")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Phase 5 loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                continue
            scheduler.step()
            _update_ema(ema, model, args.ema_decay)
            step += 1
            row = {"step": step, "time_unix": time.time(), "train_loss": float(loss.item()),
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{name}": float(value.item()) for name, value in components.items()})
            if step % args.validation_interval == 0 or step == args.max_steps:
                metrics, _ = validate_native_cache(
                    ema, fast_val, val_cache, spec, device, **validation_options)
                safe = passes_metric_guardrails(
                    metrics, baseline, color_tolerance=args.max_color_regression,
                    ssim_tolerance=args.max_ssim_regression)
                improved = safe and float(metrics["psnr_mean"]) > best_psnr + args.min_psnr_improvement
                row.update({"native_validation": metrics, "metric_guardrails_passed": safe})
                if improved:
                    best_metrics, best_psnr, stale = metrics, float(metrics["psnr_mean"]), 0
                    _save_checkpoint(output / "best.pt", checkpoint_payload(
                        ema, config, step, metrics, tile_size=args.tile_size, overlap=args.tile_overlap))
                else:
                    stale += 1
                _save_checkpoint(output / f"step_{step:07d}.pt", checkpoint_payload(
                    ema, config, step, metrics, tile_size=args.tile_size, overlap=args.tile_overlap))
                _save_checkpoint(state_path, {
                    "online_model": model.state_dict(), "ema_model": ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "stale": stale,
                    "baseline": baseline, "best_metrics": best_metrics, "best_psnr": best_psnr,
                })
                stopped_early = step >= args.min_steps and stale >= args.early_stop_patience
                model.train()
            with (output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if step % 25 == 0 or "native_validation" in row:
                print(json.dumps(row), flush=True)
            if step >= args.max_steps or stopped_early:
                break
    best_model, _ = load_restorer_checkpoint(output / "best.pt", device)
    final_summary, rows = validate_native_cache(
        best_model, val_records, val_cache, spec, device, **validation_options)
    final_summary.update({"checkpoint": str((output / "best.pt").resolve()),
                          "checkpoint_sha256": file_sha256(output / "best.pt")})
    validation_output = Path(args.validation_output_dir)
    write_json(validation_output / "metrics.json", rows)
    write_json(validation_output / "summary.json", final_summary)
    write_json(output / "summary.json", {
        "global_step": step, "stopped_early": stopped_early,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "best_quick_native_validation": best_metrics,
        "full_native_validation": final_summary,
        "test_evaluated": False,
    })
    print(json.dumps(final_summary), flush=True)


if __name__ == "__main__":
    main()
