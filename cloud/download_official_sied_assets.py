#!/usr/bin/env python3
"""Download selected assets from the official public SIED OneDrive folders.

The official repository publishes folders rather than stable direct file URLs.
This helper resolves the current pre-authenticated URL, downloads to ``.part``
files with HTTP Range resume support, and verifies the byte count advertised by
OneDrive before atomically moving an asset into place.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

DATASET_SHARE = (
    "https://1drv.ms/f/c/e379fe7c770e3033/"
    "Ejd2sO7WNMlGiAVSqFtu1KkBF8UL_RU9unCds1Mu8z8IPw?e=gyBKoy"
)
CHECKPOINT_SHARE = (
    "https://1drv.ms/f/c/e379fe7c770e3033/"
    "Ekg7dQ-J7PxCuJ3_-FQ_uNsB0ZWvwL2HUK3dSFcogki0jA?e=XWToVn"
)
BADGER_TOKEN_URL = "https://api-badgerp.svc.ms/v1.0/token"
ONEDRIVE_API = "https://my.microsoftpersonalcontent.com/_api/v2.0"
BADGER_APP_ID = "5cbed6ac-a083-4e14-b191-b4ba07653de2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected official SIED data and checkpoints"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-assets",
        nargs="*",
        default=["Canon_0.0001.zip", "Canon_gt.zip"],
    )
    parser.add_argument(
        "--checkpoint-assets",
        nargs="*",
        default=["canon_0.0001.pt"],
    )
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=8)
    return parser


def _share_token(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = session.request(method, url, timeout=(30, 120), **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object from {url}")
    return payload


def get_badger_token(session: requests.Session) -> str:
    payload = _request_json(
        session,
        "POST",
        BADGER_TOKEN_URL,
        json={"appId": BADGER_APP_ID},
    )
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("OneDrive did not return a Badger access token")
    return token


def list_share(session: requests.Session, share_url: str) -> dict[str, dict[str, Any]]:
    token = get_badger_token(session)
    headers = {"Authorization": f"Badger {token}", "Prefer": "autoredeem"}
    encoded = _share_token(share_url)
    root = _request_json(
        session,
        "POST",
        f"{ONEDRIVE_API}/shares/u!{encoded}/driveitem?select=id,name,folder,size",
        headers=headers,
        data=b"",
    )
    item_id = root.get("id")
    if not isinstance(item_id, str) or "!" not in item_id:
        raise RuntimeError("The SIED OneDrive share did not resolve to a folder")
    drive_id = item_id.split("!", 1)[0]
    listing = _request_json(
        session,
        "GET",
        f"{ONEDRIVE_API}/drives/{drive_id}/items/{item_id}/children"
        "?select=id,name,size,file,@content.downloadUrl",
        headers=headers,
    )
    values = listing.get("value")
    if not isinstance(values, list):
        raise TypeError("The SIED OneDrive folder listing is malformed")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            result[value["name"]] = value
    return result


def _download_once(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    expected_size: int,
    chunk_size: int,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > expected_size:
        partial.unlink()
        downloaded = 0
    aria2 = shutil.which("aria2c")
    if aria2 is not None:
        subprocess.run(
            [
                aria2,
                "--continue=true",
                "--max-connection-per-server=16",
                "--split=16",
                "--min-split-size=4M",
                "--file-allocation=none",
                "--max-tries=5",
                "--retry-wait=2",
                f"--dir={partial.parent}",
                f"--out={partial.name}",
                url,
            ],
            check=True,
        )
        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            raise OSError(
                f"Incomplete download for {destination.name}: "
                f"{actual_size} of {expected_size} bytes"
            )
        os.replace(partial, destination)
        partial.with_suffix(partial.suffix + ".aria2").unlink(missing_ok=True)
        return
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with session.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 180),
    ) as response:
        response.raise_for_status()
        if downloaded and response.status_code != 206:
            partial.unlink(missing_ok=True)
            downloaded = 0
        mode = "ab" if downloaded else "wb"
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    stream.write(chunk)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise OSError(
            f"Incomplete download for {destination.name}: "
            f"{actual_size} of {expected_size} bytes"
        )
    os.replace(partial, destination)


def download_asset(
    session: requests.Session,
    share_url: str,
    name: str,
    output_dir: Path,
    *,
    chunk_size: int,
    attempts: int,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            assets = list_share(session, share_url)
            if name not in assets:
                raise FileNotFoundError(
                    f"{name!r} is absent; available: {', '.join(sorted(assets))}"
                )
            asset = assets[name]
            expected_size = int(asset["size"])
            destination = output_dir / name
            if destination.is_file() and destination.stat().st_size == expected_size:
                print(json.dumps({"asset": name, "status": "already_complete"}))
                return
            download_url = asset.get("@content.downloadUrl")
            if not isinstance(download_url, str) or not download_url:
                raise RuntimeError(f"OneDrive returned no download URL for {name}")
            print(
                json.dumps(
                    {
                        "asset": name,
                        "status": "downloading",
                        "bytes": expected_size,
                        "attempt": attempt,
                    }
                ),
                flush=True,
            )
            _download_once(
                session,
                download_url,
                destination,
                expected_size=expected_size,
                chunk_size=chunk_size,
            )
            print(json.dumps({"asset": name, "status": "complete"}), flush=True)
            return
        except (
            OSError,
            requests.RequestException,
            subprocess.CalledProcessError,
            KeyError,
            ValueError,
        ) as error:
            if attempt == attempts:
                raise
            print(
                json.dumps(
                    {
                        "asset": name,
                        "status": "retrying",
                        "attempt": attempt,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
            time.sleep(min(30, 2**attempt))


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_mib <= 0 or args.attempts <= 0:
        raise ValueError("chunk-mib and attempts must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "dip-course-sied-downloader/1.0"
    for name in args.dataset_assets:
        download_asset(
            session,
            DATASET_SHARE,
            name,
            output_dir,
            chunk_size=args.chunk_mib * 1024 * 1024,
            attempts=args.attempts,
        )
    for name in args.checkpoint_assets:
        download_asset(
            session,
            CHECKPOINT_SHARE,
            name,
            output_dir,
            chunk_size=args.chunk_mib * 1024 * 1024,
            attempts=args.attempts,
        )


if __name__ == "__main__":
    main()
