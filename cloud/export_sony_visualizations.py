"""Export six fixed Sony comparisons only after the full 1000-image verification.

The learned outputs are copied from frozen PNGs. Only RAW/ISP illustrations use
the classical processing functions. No inference, training or rescoring occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from verify_sony1000 import FIELDS, verify


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save_display(image: Image.Image, target: Path, *, crop=None, width=1440) -> dict:
    native = image.size
    image = image.convert('RGB')
    if crop is not None:
        left, top, right, bottom = crop
        if not 0 <= left < right <= native[0] or not 0 <= top < bottom <= native[1]:
            raise ValueError('Display crop must stay inside the native image')
        image = image.crop(crop)
    image.thumbnail((width, width), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=95, subsampling=0, optimize=True)
    return {'display_file': target.name, 'native_size': native, 'display_size': image.size,
            'crop_xyxy': crop, 'display_sha256': sha256(target),
            'processing': 'Equal-aspect display resize and JPEG95; no additional tone/color changes'}


def to_pil(array) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8))


def export(root: Path, output: Path | None = None) -> dict:
    final = root / 'submission1000_native'
    checked = verify(final)  # Mandatory before reading or exporting any held-out reference.
    from dip_course.enhancement.classical import (
        auto_tone_map_rgb,
        demosaic_rggb,
        tone_map_rgb,
    )
    from dip_course.io import (
        RawSpec,
        normalize_raw,
        read_pair_manifest,
        read_raw_uint16,
    )

    output = output or root / 'report_assets'
    records = read_pair_manifest(root / 'data/SIED_dataset',
                                 root / 'project/manifests/enhancement/sony_test.txt')
    summary = json.loads((final / 'summary.json').read_text())
    with (final / 'metrics.csv').open(newline='') as stream:
        rows = {r['key']: r for r in csv.DictReader(stream) if r['split'] == 'test'}
    with (final / 'baseline_metrics.csv').open(newline='') as stream:
        base = {r['key']: r for r in csv.DictReader(stream) if r['split'] == 'test'}
    data_receipt = json.loads((root / 'logs/sony_data_verified.json').read_text())
    examples = []
    crop = (1280, 720, 2560, 1440)
    for index, record in enumerate(records[:6]):
        key = record.key
        name = f'{index:04d}_{key}'
        selected_path = final / 'images/test' / f'{name}.png'
        official_path = final / 'comparison_inputs' / f'{name}_official.png'
        if sha256(selected_path) != rows[key]['png_sha256']:
            raise ValueError('Final Sony image does not match its sealed metrics row')
        for p in [record.low_raw, record.gt_rgb]:
            relative = p.relative_to(root / 'data/SIED_dataset').as_posix()
            if sha256(p) != data_receipt['files'][relative]['sha256']:
                raise ValueError('Source data changed before visualization export')
        with Image.open(official_path) as im:
            if im.size != (3840, 2160):
                raise ValueError('The official display must be native resolution')
        linear = demosaic_rggb(normalize_raw(read_raw_uint16(record.low_raw, RawSpec())))
        preview = tone_map_rgb(linear, exposure=1, gamma=2.2, white_balance=(1, 1, 1))
        isp, parameters = auto_tone_map_rgb(linear, gamma=1.8, percentile=99, target=0.9,
                                           white_balance_power=1)
        views = {}
        for label, array in [('raw', preview), ('isp', isp)]:
            views[label] = save_display(to_pil(array), output / f'sony_case_{key}_{label}.jpg')
        for label, path in [('official', official_path), ('final', selected_path), ('gt', record.gt_rgb)]:
            with Image.open(path) as image:
                if image.size != (3840, 2160):
                    raise ValueError('Source comparison images must share the native geometry')
                entry = save_display(image, output / f'sony_case_{key}_{label}.jpg')
                entry.update(source=str(path.relative_to(root)), source_sha256=sha256(path))
                views[label] = entry
                entry = save_display(image, output / f'sony_case_{key}_{label}_crop.jpg', crop=crop)
                entry.update(source=str(path.relative_to(root)), source_sha256=sha256(path))
                views[label + '_crop'] = entry
        examples.append({'index': index, 'key': key, 'views': views,
                         'raw_source_sha256': sha256(record.low_raw),
                         'raw_preview': {'gamma': 2.2, 'exposure': 1, 'white_balance': [1, 1, 1]},
                         'classical_display_parameters': parameters,
                         'final_metrics': {k: float(rows[key][k]) for k in FIELDS},
                         'official_metrics': {k: float(base[key][k]) for k in FIELDS}})
    manifest = {'camera': 'sony', 'count': 6, 'selection': 'fixed first six official test images',
                'overall_1000': checked['overall'], 'splits': checked['splits'],
                'model': summary['selected'], 'examples': examples,
                'source_binding_sha256': sha256(final / 'binding.json'),
                'source_metrics_sha256': sha256(final / 'metrics.csv'),
                'script_sha256': sha256(Path(__file__)),
                'note': 'All numerical scores come from sealed native-float metrics; JPEG files are display copies.'}
    (output / 'sony_visualizations.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root.resolve())
    print(json.dumps({'camera': result['camera'], 'examples': result['count']}, indent=2))


if __name__ == '__main__':
    main()
