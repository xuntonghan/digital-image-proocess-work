"""Check native output equivalence and elapsed throughput on 12 fixed train images."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_sied1000 import METRICS, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    old = root / 'serial_native_pilot/prepared'
    new = root / 'prepared'
    old_binding = json.loads((old / 'binding.json').read_text())
    new_binding = json.loads((new / 'binding.json').read_text())
    for field in ['official_sha256', 'official_commit', 'camera', 'exposure', 'counts',
                  'protocol', 'data_receipt_sha256', 'all_manifest_sha256',
                  'split_manifests_sha256', 'gpu', 'packages']:
        if old_binding[field] != new_binding[field]:
            raise ValueError('Pilot changed ' + field)
    for file, digest in old_binding['code_sha256'].items():
        if file != 'cloud/sony1000_pipeline.py' and new_binding['code_sha256'][file] != digest:
            raise ValueError('Pilot changed a shared algorithm: ' + file)
    paths = sorted((new / 'rows/train').glob('*.json'))
    if len(paths) != 12:
        raise ValueError('The equivalence pilot requires exactly 12 complete training images')
    delta = {m: 0.0 for m in METRICS}
    serial_seconds = 0.0
    for index, path in enumerate(paths):
        before = json.loads((old / 'rows/train' / path.name).read_text())
        after = json.loads(path.read_text())
        if before['index'] != index or after['index'] != index:
            raise ValueError('Pilot is not the same fixed first 12 train images')
        for field in ['key', 'split', 'seeds', 'base_tensor_sha256', 'png_sha256', 'color_pair_sha256']:
            if before[field] != after[field]:
                raise ValueError('Pilot output changed: ' + field)
        for directory, row in [(old, before), (new, after)]:
            if sha256(directory / row['png']) != row['png_sha256']:
                raise ValueError('Native pilot PNG verification failed')
        for metric in METRICS:
            gap = abs(float(before[metric]) - float(after[metric]))
            tolerance = 1e-6 if metric == 'lpips' else 1e-10
            if not math.isfinite(gap) or gap > tolerance:
                raise ValueError('Pilot metric changed: ' + metric)
            delta[metric] = max(delta[metric], gap)
        serial_seconds += before['wall_seconds']
    progress = json.loads((root / 'logs/progress.json').read_text())
    if progress['completed'] != 12:
        raise ValueError('Pilot progress does not match completed rows')
    parallel_seconds = 12 / progress['new_images_per_second']
    result = {'count': 12, 'split': 'train', 'native_float_predictions_bit_identical': True,
              'native_png_bytes_identical': True, 'float32_color_pairs_identical': True,
              'maximum_metric_absolute_difference': delta,
              'serial_summed_row_wall_seconds': serial_seconds,
              'parallel_prepare_wall_seconds': parallel_seconds,
              'serial_seconds_per_image': serial_seconds / 12,
              'parallel_seconds_per_image_throughput': parallel_seconds / 12,
              'observed_preparation_throughput_ratio': serial_seconds / parallel_seconds,
              'caveat': 'Fixed 12-image preparation pilot, includes startup/cache conditions; '
                        'not a full-experiment or model-training speedup.',
              'new_source_sha256': new_binding['code_sha256']['cloud/sony1000_pipeline.py'],
              'old_binding_sha256': sha256(old / 'binding.json'),
              'new_binding_sha256': sha256(new / 'binding.json')}
    path = root / 'logs/parallel_equivalence.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['observed_preparation_throughput_ratio'] <= 1.05:
        raise ValueError('No demonstrated throughput benefit; keep the serial version')


if __name__ == '__main__':
    main()
