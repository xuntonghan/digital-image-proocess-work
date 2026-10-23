"""Validation-only SIED selection and immutable frozen-test provenance.

Only the standard library is used; no model inference or test image is loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import tempfile
from pathlib import Path

COLOR_METRICS = (
    "delta_e2000_mean", "delta_e2000_p95_mean", "lightness_mae_mean",
    "saturation_mae_mean", "hue_error_degrees_mean",
)
PHASE_PATHS = (
    (1, "canon_raw_guided_naf_phase1/selected_validation_summary.json", "canon_raw_guided_naf_phase1/selected.pt"),
    (2, "canon_raw_guided_naf_phase2_val_native/summary.json", "canon_raw_guided_naf_phase2_native/best.pt"),
    (3, "canon_raw_guided_naf_phase3_val_native/summary.json", "canon_raw_guided_naf_phase3_psnr/best.pt"),
    (4, "canon_raw_guided_naf_phase4_val_native/summary.json", "canon_raw_guided_naf_phase4_cached_native/best.pt"),
    (5, "canon_raw_guided_naf_phase5_val_native/summary.json", "canon_raw_guided_naf_phase5_context/best.pt"),
)


class FinalizationError(ValueError):
    """Fail closed without changing previously accepted artifacts."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FinalizationError(f"Cannot read valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise FinalizationError(f"Expected JSON object: {path}")
    return payload


def immutable_json(path: Path, payload: dict, *, allow_existing: bool = True) -> None:
    """Atomically create an artifact or verify identical contents; never replace."""
    if path.exists():
        if not allow_existing:
            raise FinalizationError(f"Artifact already exists; refusing another invocation: {path}")
        if read_json(path) != payload:
            raise FinalizationError(f"Immutable artifact mismatch; preserving {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not allow_existing:
                raise FinalizationError(f"Artifact already exists; refusing another invocation: {path}") from None
            if read_json(path) != payload:
                raise FinalizationError(f"Immutable artifact mismatch; preserving {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def finite_metric(summary: dict, name: str) -> float:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise FinalizationError(f"Missing/non-finite metric: {name}")
    return float(value)


def validation_candidate(root: Path, phase: int, summary_path: str, checkpoint_path: str) -> dict:
    summary_file, checkpoint = root / "runs" / summary_path, root / "runs" / checkpoint_path
    summary = read_json(summary_file)
    if summary.get("count") != 92:
        raise FinalizationError(f"Phase {phase}: expected complete 92-image validation")
    if not checkpoint.is_file():
        raise FinalizationError(f"Phase {phase}: missing checkpoint {checkpoint}")
    actual_hash = sha256(checkpoint)
    declared_hash = summary.get("checkpoint_sha256")
    if phase == 5 and not declared_hash:
        raise FinalizationError("Phase 5 validation must bind checkpoint_sha256")
    if declared_hash is not None and declared_hash != actual_hash:
        raise FinalizationError(f"Phase {phase}: validation/checkpoint SHA256 mismatch")
    metrics = {name: finite_metric(summary, name) for name in ("psnr_mean", "ssim_mean")}
    for name in COLOR_METRICS:
        if name in summary:
            metrics[name] = finite_metric(summary, name)
    for name, value in summary.items():
        if name.endswith("_mean") and value is not None:
            finite_metric(summary, name)
    tile = summary.get("refiner_tile_size", summary.get("tile_size", 512 if phase == 5 else 768))
    overlap = summary.get("refiner_overlap", summary.get("tile_overlap", 64 if phase == 5 else 96))
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise FinalizationError(f"Phase {phase}: invalid tile size")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < tile:
        raise FinalizationError(f"Phase {phase}: invalid overlap")
    return {
        "phase": phase, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_hash, "validation_summary": str(summary_file.resolve()),
        "validation_summary_sha256": sha256(summary_file), "count": 92,
        "refiner_tile_size": tile, "refiner_overlap": overlap, **metrics,
    }


def metric_rejection(candidate: dict, baseline: dict) -> list[str]:
    reasons = []
    if candidate["ssim_mean"] < baseline["ssim_mean"] - 0.002:
        reasons.append("SSIM decreases by more than 0.002 versus Phase 2")
    for name in COLOR_METRICS:
        if name not in baseline:
            continue
        if name not in candidate:
            reasons.append(f"Missing Phase 2 color metric: {name}")
        elif candidate[name] > baseline[name] * 1.02:
            reasons.append(f"{name} increases by more than 2% versus Phase 2")
    return reasons


def manifest_keys(manifest: Path) -> list[str]:
    keys = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = shlex.split(line)
        if len(fields) != 2:
            raise FinalizationError("Frozen manifest must contain LOW_RAW GT_RGB pairs")
        keys.append(Path(fields[0]).stem)
    if len(keys) != 76 or len(set(keys)) != 76:
        raise FinalizationError("Frozen manifest must contain exactly 76 unique images")
    return keys


def propose_selection(root: Path, manifest: Path, official_checkpoint: Path) -> dict:
    candidates, rejected = [], []
    baseline = validation_candidate(root, *PHASE_PATHS[1])
    phase5 = validation_candidate(root, *PHASE_PATHS[4])
    completion = read_json(root / "runs/canon_raw_guided_naf_phase5_context/summary.json")
    completed_validation = completion.get("full_native_validation")
    if not isinstance(completed_validation, dict) or completed_validation.get("count") != 92:
        raise FinalizationError("Phase 5 training completion summary is missing/incomplete")
    if completed_validation.get("checkpoint_sha256") != phase5["checkpoint_sha256"]:
        raise FinalizationError("Phase 5 completion and full validation checkpoint hashes differ")
    for phase, summary_path, checkpoint_path in PHASE_PATHS:
        try:
            candidate = baseline if phase == 2 else phase5 if phase == 5 else validation_candidate(
                root, phase, summary_path, checkpoint_path)
            reasons = metric_rejection(candidate, baseline)
            if reasons:
                rejected.append({"phase": phase, "reasons": reasons})
            else:
                candidates.append(candidate)
        except FinalizationError as error:
            if phase in (2, 5):
                raise
            rejected.append({"phase": phase, "reasons": [str(error)]})
    manifest_keys(manifest)
    if not official_checkpoint.is_file():
        raise FinalizationError("Missing official SIED checkpoint")
    selected = max(candidates, key=lambda item: (item["psnr_mean"], -item["phase"]))
    return {
        "schema_version": 2, "selection_metric": "validation_psnr_mean_after_full_validation_guardrails",
        "guardrails": {
            "reference_phase": 2, "validation_count": 92, "ssim_max_absolute_decrease": 0.002,
            "color_max_relative_increase": 0.02,
            "color_metrics": [name for name in COLOR_METRICS if name in baseline],
            "claim": "Bounded regression is allowed; not a claim that every metric improves.",
        },
        "reference": baseline, "selected": selected, "candidates": candidates, "rejected": rejected,
        "frozen_protocol": {
            "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest), "count": 76,
            "official_checkpoint": str(official_checkpoint.resolve()),
            "official_checkpoint_sha256": sha256(official_checkpoint),
            "refiner_checkpoint": str((root / "final/best.pt").resolve()),
            "refiner_checkpoint_sha256": selected["checkpoint_sha256"],
            "refiner_tile_size": selected["refiner_tile_size"], "refiner_overlap": selected["refiner_overlap"],
            "seed": 3407, "sampling_steps": 20, "ensemble_size": 3, "precision": "fp16", "lpips": True,
        },
    }


def verify_bound_inputs(selection: dict) -> None:
    protocol = selection["frozen_protocol"]
    checks = (
        (Path(selection["selected"]["checkpoint"]), selection["selected"]["checkpoint_sha256"]),
        (Path(protocol["manifest"]), protocol["manifest_sha256"]),
        (Path(protocol["official_checkpoint"]), protocol["official_checkpoint_sha256"]),
    )
    for path, expected in checks:
        if not path.is_file() or sha256(path) != expected:
            raise FinalizationError(f"Bound input changed; preserving frozen results: {path}")
    manifest_keys(Path(protocol["manifest"]))


def copy_snapshot(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise FinalizationError(f"Immutable checkpoint mismatch; preserving {destination}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".best.pt.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected_hash:
            raise FinalizationError("Selected checkpoint changed while snapshotting")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256(destination) != expected_hash:
                raise FinalizationError(f"Immutable checkpoint mismatch: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)


def result_receipt(root: Path, selection: dict) -> dict:
    return {
        "selection_sha256": sha256(root / "final/selection.json"),
        "checkpoint_sha256": selection["selected"]["checkpoint_sha256"],
        "summary_sha256": sha256(root / "final/frozen_test/summary.json"),
        "metrics_sha256": sha256(root / "final/frozen_test/metrics.csv"),
        "manifest_sha256": selection["frozen_protocol"]["manifest_sha256"], "count": 76,
    }


def verify_frozen_result(root: Path, selection: dict) -> dict:
    """Validate existing results read-only, including pre-test provenance."""
    frozen = root / "final/frozen_test"
    if read_json(frozen / "selection.json") != selection:
        raise FinalizationError("Frozen result selection does not match; refusing to overwrite or retest")
    expected_start = {"selection_sha256": sha256(root / "final/selection.json"),
                      "checkpoint_sha256": selection["selected"]["checkpoint_sha256"]}
    if read_json(frozen / "evaluation_started.json") != expected_start:
        raise FinalizationError("Frozen result has no matching pre-test checkpoint binding")
    protocol = selection["frozen_protocol"]
    snapshot = Path(protocol["refiner_checkpoint"])
    if not snapshot.is_file() or sha256(snapshot) != protocol["refiner_checkpoint_sha256"]:
        raise FinalizationError("Frozen checkpoint snapshot is missing or changed")
    summary = read_json(frozen / "summary.json")
    if summary.get("count") != 76 or not summary.get("blind_inference") or not summary.get("metrics_evaluated"):
        raise FinalizationError("Frozen result must contain 76 blind, metric-evaluated images")
    for name in ("seed", "sampling_steps", "ensemble_size", "precision", "refiner_tile_size", "refiner_overlap"):
        if summary.get(name) != protocol[name]:
            raise FinalizationError(f"Frozen inference parameter mismatch: {name}")
    if Path(summary.get("checkpoint", "")).resolve() != Path(protocol["official_checkpoint"]):
        raise FinalizationError("Frozen official checkpoint path mismatch")
    refiner = summary.get("refiner")
    if not isinstance(refiner, dict) or Path(refiner.get("checkpoint", "")).resolve() != snapshot:
        raise FinalizationError("Frozen refiner checkpoint does not match immutable snapshot")
    for name in ("psnr_mean", "ssim_mean", "lpips_mean", *COLOR_METRICS):
        finite_metric(summary, name)
    try:
        with (frozen / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise FinalizationError("Frozen per-image metrics are missing") from error
    if [row.get("key") for row in rows] != manifest_keys(Path(protocol["manifest"])):
        raise FinalizationError("Frozen per-image metrics do not match the 76-image manifest")
    receipt = frozen / "verification.json"
    if receipt.exists() and read_json(receipt) != result_receipt(root, selection):
        raise FinalizationError("Frozen verification receipt mismatch; preserving all existing files")
    return summary


def prepare_finalization(root: Path, manifest: Path, official_checkpoint: Path) -> str:
    selection_path = root / "final/selection.json"
    frozen = root / "final/frozen_test"
    proposed = propose_selection(root, manifest, official_checkpoint)
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection != proposed:
            raise FinalizationError("Immutable selection changed; refusing to overwrite or repeat frozen evaluation")
    else:
        selection = proposed
    verify_bound_inputs(selection)
    if (frozen / "summary.json").exists():
        verify_frozen_result(root, selection)
        return "reuse"
    if (frozen / "evaluation_started.json").exists():
        raise FinalizationError("Frozen evaluation already started without a completed summary; refusing automatic retest")
    if (root / "final/final_result.json").exists():
        raise FinalizationError("Final result exists without its frozen summary; refusing automatic retest")
    if (frozen / "metrics.csv").exists() or (frozen / "images").exists():
        raise FinalizationError("Unbound/partial frozen artifacts exist; preserving them and refusing retest")
    snapshot = Path(selection["frozen_protocol"]["refiner_checkpoint"])
    if snapshot.exists() and sha256(snapshot) != selection["selected"]["checkpoint_sha256"]:
        raise FinalizationError("Existing final checkpoint differs; preserving it")
    immutable_json(selection_path, selection)
    immutable_json(frozen / "selection.json", selection)
    copy_snapshot(Path(selection["selected"]["checkpoint"]), snapshot, selection["selected"]["checkpoint_sha256"])
    return "run"


def mark_evaluation_started(root: Path) -> None:
    selection = read_json(root / "final/selection.json")
    verify_bound_inputs(selection)
    frozen = root / "final/frozen_test"
    marker = frozen / "evaluation_started.json"
    if marker.exists() or (frozen / "summary.json").exists():
        raise FinalizationError("Frozen evaluation already started; refusing another invocation")
    if read_json(frozen / "selection.json") != selection:
        raise FinalizationError("Pre-test immutable selection mismatch")
    snapshot = Path(selection["frozen_protocol"]["refiner_checkpoint"])
    if sha256(snapshot) != selection["selected"]["checkpoint_sha256"]:
        raise FinalizationError("Pre-test snapshot SHA256 mismatch")
    immutable_json(marker, {"selection_sha256": sha256(root / "final/selection.json"),
                            "checkpoint_sha256": selection["selected"]["checkpoint_sha256"]},
                   allow_existing=False)


def seal_finalization(root: Path) -> dict:
    selection = read_json(root / "final/selection.json")
    verify_bound_inputs(selection)
    summary = verify_frozen_result(root, selection)
    receipt = result_receipt(root, selection)
    final_result = {"selection": selection, "frozen_test": summary, "verification": receipt}
    result_file = root / "final/final_result.json"
    if result_file.exists() and read_json(result_file) != final_result:
        raise FinalizationError("Existing final result differs; preserving it")
    immutable_json(root / "final/frozen_test/verification.json", receipt)
    immutable_json(result_file, final_result)
    return final_result
