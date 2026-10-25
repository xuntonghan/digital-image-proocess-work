"""Independently recompute Sony's sealed 1000-image statistics without a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

FIELDS = {'psnr': 'psnr_mean', 'ssim': 'ssim_mean', 'lpips': 'lpips_mean',
          'delta_e2000_mean': 'delta_e2000_mean', 'delta_e2000_p95': 'delta_e2000_p95_mean',
          'lightness_mae': 'lightness_mae_mean', 'saturation_mae': 'saturation_mae_mean',
          'hue_error_degrees': 'hue_error_degrees_mean'}


def verify(directory: Path) -> dict:
    receipt = json.loads((directory / 'verification.json').read_text())
    for name, field in [('metrics.csv', 'metrics_sha256'), ('summary.json', 'summary_sha256'),
                        ('binding.json', 'binding_sha256'),
                        ('baseline_metrics.csv', 'baseline_metrics_sha256')]:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != receipt[field]:
            raise ValueError('Checksum mismatch: ' + name)
    with (directory / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000 or len({r['key'] for r in rows}) != 1000:
        raise ValueError('Exactly 1000 unique Sony images are required')
    if Counter(r['split'] for r in rows) != {'train': 822, 'val': 91, 'test': 87}:
        raise ValueError('The fixed Sony partition changed')
    if {r['binding_sha256'] for r in rows} != {receipt['binding_sha256']}:
        raise ValueError('Multiple model bindings')
    binding = json.loads((directory / 'binding.json').read_text())
    if binding['camera'] != 'sony' or binding['exposure'] != '0.001':
        raise ValueError('Wrong camera or exposure')
    summary = json.loads((directory / 'summary.json').read_text())
    answer = {'verified': True, 'count': 1000, 'splits': {}}

    def means(subset):
        result = {'count': len(subset)}
        for column, name in FIELDS.items():
            values = [float(r[column]) for r in subset]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite metric')
            result[name] = statistics.fmean(values)
        return result

    def compare(actual, expected):
        if any(not math.isclose(actual[k], expected[k], rel_tol=1e-12, abs_tol=1e-12) for k in actual):
            raise ValueError('Summary is not the arithmetic mean of all per-image values')

    answer['overall'] = means(rows)
    compare(answer['overall'], summary['overall'])
    for split in ['train', 'val', 'test']:
        subset = [r for r in rows if r['split'] == split]
        for index, row in enumerate(subset):
            if int(row['index']) != index or row['seeds'] != ';'.join(
                    str(3407 + index + j * 1000003) for j in range(3)):
                raise ValueError('Wrong Sony split-local index or seeds')
        answer['splits'][split] = means(subset)
        compare(answer['splits'][split], summary['splits'][split])
    with (directory / 'baseline_metrics.csv').open(newline='') as stream:
        baseline = list(csv.DictReader(stream))
    if [(r['split'], r['key']) for r in baseline] != [(r['split'], r['key']) for r in rows]:
        raise ValueError('Baseline and final images do not match')
    answer['official_baseline_overall'] = means(baseline)
    compare(answer['official_baseline_overall'], summary['official_baseline_overall'])
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] /
                        'reports/assets/enhancement/sony1000_20261025/submission1000_native')
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
