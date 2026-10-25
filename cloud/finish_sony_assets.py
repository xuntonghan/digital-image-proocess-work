"""Wait for complete Sony1000, check three fresh train inferences, export figures.

Never shuts the instance down. The Mac must first verify its independent backup.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from evaluate_sied1000 import METRICS, sha256
from sony1000_pipeline import write_json
from verify_sony1000 import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    log = root / 'logs/final_assets_status.txt'
    with (root / 'logs/final_assets.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if log.exists() and log.read_text().strip() == 'ASSETS_READY_PENDING_MAC_VERIFIED_BACKUP':
            return
        log.write_text('WAITING_FOR_FINAL_1000\n')
        wait_started = time.monotonic()
        while True:
            status = (root / 'logs/status.txt').read_text().strip()
            if status == 'COMPLETE_PENDING_INDEPENDENT_BACKUP_AND_REPORT':
                break
            if time.monotonic() - wait_started > 3 * 3600:
                raise TimeoutError('Sony final evaluation has not finished within bounded wait')
            time.sleep(5)
        verify(root / 'submission1000_native')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if active.strip():
            raise RuntimeError('GPU is still used; do not launch overlapping reproduction work')
        choice = json.loads((root / 'selection.json').read_text())['selected']
        command = [sys.executable, '-u', 'cloud/reproduce_sony_submission.py',
                   '--data-root', str(root / 'data/SIED_dataset'),
                   '--official-repo', str(root / 'third_party/SIED-official'),
                   '--official-checkpoint', str(root / 'checkpoints/official/sony_0.001.pt'),
                   '--sealed-directory', str(root / 'submission1000_native'),
                   '--output-dir', str(root / 'reproduction_smoke'),
                   '--gpu-lock', str(root / 'sony_gpu.lock'), '--limit-new', '3']
        if choice['checkpoint']:
            command += ['--lut-checkpoint', str(root / choice['checkpoint'])]
        log.write_text('THREE_TRAIN_RAW_REPRODUCTION\n')
        subprocess.run(command, cwd=root / 'project', check=True)
        rows = sorted((root / 'reproduction_smoke/rows/train').glob('*.json'))
        if len(rows) != 3 or list((root / 'reproduction_smoke/rows/test').glob('*.json')):
            raise ValueError('Only three original train images may be reproduced here')
        differences = {k: 0. for k in METRICS}
        png_equal = True
        keys = []
        for index, path in enumerate(rows):
            fresh = json.loads(path.read_text())
            old = json.loads((root / 'submission1000_native/rows/train' / path.name).read_text())
            if fresh['key'] != old['key'] or fresh['index'] != index or fresh['seeds'] != old['seeds']:
                raise ValueError('Reproduction image identity or seeds changed')
            image = root / 'reproduction_smoke/images/train' / (path.stem + '.png')
            if sha256(image) != fresh['png_sha256']:
                raise ValueError('Reproduction image checksum changed')
            png_equal = png_equal and fresh['png_sha256'] == old['png_sha256']
            keys.append(fresh['key'])
            for field in METRICS:
                delta = abs(float(fresh[field]) - float(old[field]))
                if not math.isfinite(delta):
                    raise ValueError('Nonfinite reproduction difference')
                differences[field] = max(differences[field], delta)
        proof = {'train_image_count':3,'keys':keys,'native_png_sha256_equal':png_equal,
                 'metric_max_absolute_difference':differences,
                 'all_eight_metrics_exactly_equal':all(v == 0 for v in differences.values()),
                 'same_camera_model_and_seeds':True,'held_out_test_rerun':False,
                 'sealed_binding_sha256':sha256(root / 'submission1000_native/binding.json'),
                 'reproduction_binding_sha256':sha256(root / 'reproduction_smoke/binding.json')}
        write_json(root / 'reproduction_smoke/verification_comparison.json', proof)
        # Preserve even a negative reproducibility result and require explicit review.
        if not png_equal or not all(v <= 1e-5 for v in differences.values()):
            raise ValueError('Fresh reproduction needs review; evidence saved, no shutdown')
        log.write_text('EXPORTING_SIX_FIXED_COMPARISONS\n')
        subprocess.run([sys.executable, 'cloud/export_sony_visualizations.py', '--root', str(root)],
                       cwd=root / 'project', check=True)
        log.write_text('ASSETS_READY_PENDING_MAC_VERIFIED_BACKUP\n')
        print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
