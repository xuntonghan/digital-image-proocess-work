"""Prepare only the original Sony 0.001 course pairs in a fresh experiment root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

COUNTS = {"train": 822, "val": 91, "test": 87}
OFFICIAL_COMMIT = "fb7af94ac831e91c647415bf1ca0b1a75931833a"


def pairs(path: Path) -> list[tuple[str, str]]:
    result = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = shlex.split(line)
        if len(fields) != 2:
            raise ValueError(f"Invalid pair: {line}")
        for value in fields:
            p = PurePosixPath(value)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe manifest path")
        result.append(tuple(fields))
    return result


def validate_sony_manifests(project: Path) -> dict[str, list[tuple[str, str]]]:
    source = pairs(project / "manifests/source/sony_all.txt")
    splits = {s: pairs(project / f"manifests/enhancement/sony_{s}.txt") for s in COUNTS}
    combined = [pair for rows in splits.values() for pair in rows]
    if len(source) != 1000 or len(set(source)) != 1000 or set(source) != set(combined):
        raise ValueError("Sony source and split manifests do not form the same 1000 pairs")
    if {s: len(rows) for s, rows in splits.items()} != COUNTS or len(set(combined)) != 1000:
        raise ValueError("Sony requires disjoint 822/91/87 splits")
    if len({PurePosixPath(raw).stem for raw, _ in combined}) != 1000:
        raise ValueError("Duplicate Sony image keys")
    if any(not raw.startswith("0.001/") or not gt.startswith("gt 2/") for raw, gt in source):
        raise ValueError("Refusing a Canon or different-exposure manifest")
    return splits


def archive_members(names: list[str], expected: list[str]) -> dict[str, str]:
    """Resolve the Sony GT archive's folder name without changing pair identities."""
    by_name: dict[str, list[str]] = {}
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Unsafe path in official archive")
        if name.endswith("/") or "__MACOSX" in p.parts or p.name.startswith("."):
            continue
        by_name.setdefault(p.name, []).append(name)
    result = {}
    for target in expected:
        candidates = by_name.get(PurePosixPath(target).name, [])
        if target in candidates:
            result[target] = target
        elif len(candidates) == 1:
            result[target] = candidates[0]
        else:
            raise ValueError(f"Missing or ambiguous official archive member: {target}")
    return result


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "project"
    validate_sony_manifests(project)
    source = pairs(project / "manifests/source/sony_all.txt")
    import requests
    from download_official_sied_assets import (
        CHECKPOINT_SHARE,
        DATASET_SHARE,
        download_asset,
    )

    assets, data = root / "assets", root / "data/SIED_dataset"
    assets.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    required = {"Sony_0.001.zip": 11489809753, "Sony_gt.zip": 3540159732,
                "sony_0.001.pt": 2610974590}
    missing_assets = sum(max(0, size - (assets / name).stat().st_size)
                         if (assets / name).exists() else size for name, size in required.items())
    missing_data = sum(16588800 for raw, _ in source if not (data / raw).exists())
    # Include GT, native validation caches, final images and a 15 GiB reserve.
    needed = missing_assets + missing_data + 30 * 1024**3
    if shutil.disk_usage(root).free < needed:
        raise OSError(f"Insufficient free storage for Sony preparation: need {needed} bytes")
    def fetch(name):
        with requests.Session() as session:
            download_asset(session, DATASET_SHARE if name.endswith('.zip') else CHECKPOINT_SHARE,
                           name, assets, chunk_size=8 * 1024**2, attempts=8)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, required))
    receipt = {}
    for archive, column in [("Sony_0.001.zip", 0), ("Sony_gt.zip", 1)]:
        with zipfile.ZipFile(assets / archive) as z:
            mapping = archive_members(z.namelist(), [pair[column] for pair in source])
            for target, member in mapping.items():
                path = data / target
                info = z.getinfo(member)
                if column == 0 and info.file_size != 16588800:
                    raise ValueError(f"Wrong Sony RAW size: {member}")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    # Compare against the archive, never overwrite an existing dataset file.
                    digest = hashlib.sha256()
                    with z.open(member) as stream:
                        for block in iter(lambda: stream.read(4 * 1024**2), b""):
                            digest.update(block)
                    if path.stat().st_size != info.file_size or file_sha(path) != digest.hexdigest():
                        raise ValueError(f"Existing dataset file differs from official archive: {path}")
                else:
                    if shutil.disk_usage(root).free < 15 * 1024**3 + info.file_size:
                        raise OSError("15 GiB reserve reached")
                    partial = path.with_suffix(path.suffix + '.sony-part')
                    with z.open(member) as src, partial.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024**2)
                    os.replace(partial, path)
                receipt[target] = {"bytes": path.stat().st_size, "sha256": file_sha(path),
                                   "archive": archive, "member": member}
        print(json.dumps({"extracted": archive, "pairs": 1000}), flush=True)
    checkpoint = root / "checkpoints/official/sony_0.001.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        if file_sha(checkpoint) != file_sha(assets / checkpoint.name):
            raise ValueError("Existing official Sony checkpoint differs")
    else:
        # Hardlink the downloaded immutable file to avoid a second 2.6 GB copy.
        os.link(assets / checkpoint.name, checkpoint)
    metadata = {"camera": "sony", "exposure": "0.001", "count": 1000,
                "split_counts": COUNTS, "official_commit": OFFICIAL_COMMIT,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": file_sha(checkpoint), "files": receipt}
    marker = root / "logs/sony_data_verified.json"
    partial = marker.with_suffix('.json.part')
    partial.write_text(json.dumps(metadata, indent=2) + '\n')
    os.replace(partial, marker)
    print(json.dumps({k: v for k, v in metadata.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
