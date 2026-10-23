"""Read-only experiment status; inspect real processes as well as status files."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def tail_lines(path: Path, *, count: int = 8, max_bytes: int = 200_000) -> list[str]:
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - max_bytes))
        return stream.read().decode("utf-8", errors="replace").splitlines()[-count:]


def command(arguments: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/sied30")
    args = parser.parse_args()
    root = Path(args.root)
    log_names = ("resume", "phase5_context", "geometry_upgrade", "compression_upgrade")
    logs = {}
    for name in log_names:
        path = root / "logs" / f"{name}.log"
        logs[name] = {"exists": path.is_file(), "tail": tail_lines(path),
                      "seconds_since_update": round(time.time() - path.stat().st_mtime, 1)
                      if path.is_file() else None}
    histories = {}
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        paths = [path for path in (directory / "training_log.jsonl",
                                   directory / "history.jsonl", directory / "summary.json")
                 if path.is_file()]
        if paths:
            histories[directory.name] = {path.name: tail_lines(path, count=2) for path in paths}
    ps = command(["ps", "-eo", "pid,etimes,pcpu,pmem,args"])
    if "stdout" in ps:
        ps["stdout"] = "\n".join(line for line in str(ps["stdout"]).splitlines()
                                    if "dip_course.cli." in line or "run_three_task_upgrade" in line)
    payload = {
        "checked_at_unix": time.time(),
        "finalization_held": (root / "HOLD_FINALIZE").exists(),
        "native_cache_png_count": len(list((root / "cache/official_sied/"
            "canon_train_val_native_ensemble3/images").glob("*.png"))),
        "phase_status": tail_lines(root / "logs/resume_status.txt"),
        "queue_status": tail_lines(root / "logs/three_task_status.txt"),
        "enhancement_queue_status": tail_lines(root / "logs/enhancement_upgrade_status.txt"),
        "processes": ps,
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                         "--format=csv,noheader"]),
        "disk": command(["df", "-h", str(root)]),
        "screens": command(["screen", "-ls"]),
        "logs": logs, "histories": histories,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
