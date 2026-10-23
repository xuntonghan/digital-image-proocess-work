"""Synthetic metadata tests only; never run a neural model or frozen dataset."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dip_course.enhancement.finalization import (
    COLOR_METRICS,
    PHASE_PATHS,
    FinalizationError,
    immutable_json,
    mark_evaluation_started,
    prepare_finalization,
    propose_selection,
    read_json,
    seal_finalization,
    sha256,
)


class SIEDFinalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sied-finalization-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "project/manifests/enhancement/canon_test.txt"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text("".join(f"low/{i}.raw gt/{i}.jpg\n" for i in range(76)))
        self.official = self.root / "checkpoints/official/canon_0.0001.pt"
        self.official.parent.mkdir(parents=True)
        self.official.write_bytes(b"synthetic-official-weight")
        self.write_phase(2, psnr=26.6)
        self.write_phase(5, psnr=26.8)

    def write_phase(self, phase: int, *, psnr=26.8, **changes) -> dict:
        _, summary_path, checkpoint_path = PHASE_PATHS[phase - 1]
        checkpoint = self.root / "runs" / checkpoint_path
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"synthetic-phase-{phase}".encode())
        summary = {
            "count": 92, "psnr_mean": psnr, "ssim_mean": 0.85,
            "delta_e2000_mean": 4.5, "delta_e2000_p95_mean": 9.0,
            "lightness_mae_mean": 4.0, "saturation_mae_mean": 0.03,
            "hue_error_degrees_mean": 10.0,
            "refiner_tile_size": 512 if phase == 5 else 768,
            "refiner_overlap": 64 if phase == 5 else 96,
        }
        if phase == 5:
            summary["checkpoint_sha256"] = sha256(checkpoint)
        summary.update(changes)
        summary_file = self.root / "runs" / summary_path
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary))
        if phase == 5:
            (checkpoint.parent / "summary.json").write_text(json.dumps({"full_native_validation": summary}))
        return summary

    def prepare(self) -> str:
        return prepare_finalization(self.root, self.manifest, self.official)

    def synthetic_completed_result(self) -> dict:
        self.assertEqual(self.prepare(), "run")
        mark_evaluation_started(self.root)
        selection = read_json(self.root / "final/selection.json")
        protocol = selection["frozen_protocol"]
        summary = {
            "count": 76, "blind_inference": True, "metrics_evaluated": True,
            "checkpoint": str(self.official),
            "refiner": {"checkpoint": protocol["refiner_checkpoint"]},
            "psnr_mean": 25.0, "ssim_mean": 0.82, "lpips_mean": 0.3,
            **{name: 0.2 for name in COLOR_METRICS},
            **{name: protocol[name] for name in (
                "seed", "sampling_steps", "ensemble_size", "precision",
                "refiner_tile_size", "refiner_overlap")},
        }
        frozen = self.root / "final/frozen_test"
        (frozen / "summary.json").write_text(json.dumps(summary))
        with (frozen / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["key"])
            writer.writeheader()
            writer.writerows({"key": str(i)} for i in range(76))
        return summary

    def tree_bytes(self) -> dict[str, bytes]:
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in (self.root / "final").rglob("*") if path.is_file()}

    def test_selects_highest_psnr_passing_full_validation_guardrails(self) -> None:
        self.write_phase(4, psnr=27.0)
        self.write_phase(5, psnr=30.0, delta_e2000_mean=6.0)
        result = propose_selection(self.root, self.manifest, self.official)
        self.assertEqual(result["selected"]["phase"], 4)
        self.assertEqual(result["guardrails"]["reference_phase"], 2)
        self.assertIn("delta_e2000_mean", result["rejected"][-1]["reasons"][0])

    def test_phase2_is_fallback_and_legacy_validation_needs_no_hash(self) -> None:
        self.write_phase(5, psnr=30.0, ssim_mean=0.84)
        result = propose_selection(self.root, self.manifest, self.official)
        self.assertEqual(result["selected"]["phase"], 2)
        self.assertEqual(len(result["selected"]["checkpoint_sha256"]), 64)

    def test_missing_color_metric_cannot_bypass_guardrail(self) -> None:
        summary = self.write_phase(4, psnr=31.0)
        del summary["hue_error_degrees_mean"]
        path = self.root / "runs" / PHASE_PATHS[3][1]
        path.write_text(json.dumps(summary))
        result = propose_selection(self.root, self.manifest, self.official)
        self.assertEqual(result["selected"]["phase"], 5)

    def test_invalid_optional_candidate_is_excluded(self) -> None:
        self.write_phase(4, psnr=31.0, count=91)
        result = propose_selection(self.root, self.manifest, self.official)
        self.assertEqual(result["selected"]["phase"], 5)
        self.write_phase(4, psnr=float("nan"))
        result = propose_selection(self.root, self.manifest, self.official)
        self.assertEqual(result["selected"]["phase"], 5)

    def test_phase5_completion_and_checkpoint_hash_are_required(self) -> None:
        for changes in ({"count": 12}, {"checkpoint_sha256": "wrong"}, {"psnr_mean": float("inf")},
                        {"ssim_mean": float("nan")}):
            with self.subTest(changes=changes):
                self.write_phase(5, **changes)
                with self.assertRaises(FinalizationError):
                    self.prepare()
        self.write_phase(5)
        completion = self.root / "runs/canon_raw_guided_naf_phase5_context/summary.json"
        completion.write_text(json.dumps({"full_native_validation": {"count": 92, "checkpoint_sha256": "old"}}))
        with self.assertRaisesRegex(FinalizationError, "hashes differ"):
            self.prepare()

    def test_declared_legacy_validation_hash_is_checked_when_present(self) -> None:
        self.write_phase(2, psnr=26.6, checkpoint_sha256="wrong")
        with self.assertRaisesRegex(FinalizationError, "SHA256 mismatch"):
            self.prepare()

    def test_protocol_binds_all_parameters_and_snapshot(self) -> None:
        self.assertEqual(self.prepare(), "run")
        selection = read_json(self.root / "final/selection.json")
        protocol = selection["frozen_protocol"]
        self.assertEqual(protocol["refiner_tile_size"], 512)
        self.assertEqual(protocol["refiner_overlap"], 64)
        self.assertEqual(protocol["seed"], 3407)
        self.assertEqual(protocol["manifest_sha256"], sha256(self.manifest))
        self.assertEqual(sha256(Path(protocol["refiner_checkpoint"])), selection["selected"]["checkpoint_sha256"])
        before = self.tree_bytes()
        self.assertEqual(self.prepare(), "run")
        self.assertEqual(self.tree_bytes(), before)

    def test_started_partial_evaluation_cannot_retest(self) -> None:
        self.prepare()
        mark_evaluation_started(self.root)
        before = self.tree_bytes()
        with self.assertRaisesRegex(FinalizationError, "refusing automatic retest"):
            self.prepare()
        with self.assertRaisesRegex(FinalizationError, "already started"):
            mark_evaluation_started(self.root)
        self.assertEqual(self.tree_bytes(), before)

    def test_start_marker_exclusive_creation_is_not_idempotent(self) -> None:
        marker = self.root / "marker.json"
        immutable_json(marker, {"checkpoint": "same"}, allow_existing=False)
        original = marker.read_bytes()
        with self.assertRaisesRegex(FinalizationError, "already exists"):
            immutable_json(marker, {"checkpoint": "same"}, allow_existing=False)
        self.assertEqual(marker.read_bytes(), original)

    def test_legacy_final_report_without_summary_cannot_start_new_test(self) -> None:
        final = self.root / "final"
        final.mkdir()
        (final / "final_result.json").write_text('{"old": true}')
        before = self.tree_bytes()
        with self.assertRaisesRegex(FinalizationError, "without its frozen summary"):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)

    def test_existing_legacy_test_is_not_rebound_or_overwritten(self) -> None:
        frozen = self.root / "final/frozen_test"
        frozen.mkdir(parents=True)
        (frozen / "summary.json").write_text(json.dumps({"count": 76, "psnr_mean": 22.9}))
        before = self.tree_bytes()
        with self.assertRaises(FinalizationError):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)
        self.assertFalse((self.root / "final/selection.json").exists())

    def test_completed_frozen_result_is_verified_and_reused_without_retest(self) -> None:
        self.synthetic_completed_result()
        result = seal_finalization(self.root)
        self.assertEqual(result["verification"]["count"], 76)
        before = self.tree_bytes()
        self.assertEqual(self.prepare(), "reuse")
        self.assertEqual(seal_finalization(self.root), result)
        self.assertEqual(self.tree_bytes(), before)

    def test_changed_selection_preserves_all_previous_artifacts(self) -> None:
        self.synthetic_completed_result()
        before = self.tree_bytes()
        self.write_phase(4, psnr=29.0)
        with self.assertRaisesRegex(FinalizationError, "Immutable selection changed"):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)

    def test_wrong_frozen_count_or_parameters_fail_without_writes(self) -> None:
        summary = self.synthetic_completed_result()
        path = self.root / "final/frozen_test/summary.json"
        for changed in ({"count": 75}, {"seed": 3}, {"refiner_tile_size": 768}, {"lpips_mean": None}):
            with self.subTest(changed=changed):
                path.write_text(json.dumps({**summary, **changed}))
                before = self.tree_bytes()
                with self.assertRaises(FinalizationError):
                    self.prepare()
                self.assertEqual(self.tree_bytes(), before)

    def test_tampered_sealed_metrics_or_checkpoint_are_rejected(self) -> None:
        self.synthetic_completed_result()
        seal_finalization(self.root)
        path = self.root / "final/frozen_test/summary.json"
        summary = read_json(path)
        summary["psnr_mean"] = 30.0
        path.write_text(json.dumps(summary))
        before = self.tree_bytes()
        with self.assertRaisesRegex(FinalizationError, "receipt mismatch"):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)
        (self.root / "final/best.pt").write_bytes(b"changed")
        before = self.tree_bytes()
        with self.assertRaisesRegex(FinalizationError, "snapshot"):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)

    def test_wrong_per_image_manifest_order_is_rejected(self) -> None:
        self.synthetic_completed_result()
        path = self.root / "final/frozen_test/metrics.csv"
        path.write_text("key\n" + "".join(f"{i}\n" for i in reversed(range(76))))
        before = self.tree_bytes()
        with self.assertRaisesRegex(FinalizationError, "per-image metrics"):
            self.prepare()
        self.assertEqual(self.tree_bytes(), before)


if __name__ == "__main__":
    unittest.main()
