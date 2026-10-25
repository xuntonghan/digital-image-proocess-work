"""Fresh Sony RAW inference with the sealed selected model, without training caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from evaluate_sied1000 import sha256, summarize
from prepare_sony1000 import COUNTS, validate_sony_manifests
from sony1000_pipeline import require_complete, write_json


def validate_binding(binding: dict) -> None:
    if (binding.get('camera') != 'sony' or binding.get('exposure') != '0.001'
            or binding.get('counts') != COUNTS):
        raise ValueError('The frozen model must be Sony 0.001 with the original 822/91/87 split')
    protocol = binding['protocol']
    expected = {'precision': 'fp16', 'sampling_steps': 20, 'ensemble_size': 3, 'seed': 3407,
                'native_rgb': [2160, 3840, 3]}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('The archived Sony inference protocol changed')


def validate_official_source(directory: Path, manifest: dict, commit: str) -> None:
    if manifest['commit'] != commit:
        raise ValueError('Wrong official source commit')
    for name, expected in manifest['files'].items():
        content = (directory / name).read_bytes()
        actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if actual != expected:
            raise ValueError('Official inference source changed: ' + name)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--official-checkpoint', type=Path, required=True)
    parser.add_argument('--source-verification', type=Path, default=project /
                        'reports/assets/enhancement/sony1000_20261025/official_source_verification.json')
    parser.add_argument('--lut-checkpoint', type=Path)
    parser.add_argument('--sealed-directory', type=Path, default=project /
                        'reports/assets/enhancement/sony1000_20261025/submission1000_native')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-new', type=int, choices=[0, 3], default=0)
    parser.add_argument('--gpu-lock', type=Path)
    args = parser.parse_args()
    sealed = args.sealed_directory.resolve()
    frozen = json.loads((sealed / 'binding.json').read_text())
    receipt = json.loads((sealed / 'verification.json').read_text())
    if sha256(sealed / 'binding.json') != receipt['binding_sha256']:
        raise ValueError('Sealed Sony binding checksum failed')
    validate_binding(frozen)
    validate_official_source(args.official_repo,
                            json.loads(args.source_verification.read_text()), frozen['official_commit'])
    if sha256(args.official_checkpoint) != frozen['official_sha256']:
        raise ValueError('Wrong official camera checkpoint')
    selected = frozen['selection']['selected']
    if bool(selected['checkpoint']) != bool(args.lut_checkpoint):
        raise ValueError('Supply the LUT only when the sealed Sony model actually selected it')
    if args.lut_checkpoint and sha256(args.lut_checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Wrong Sony LUT checkpoint')
    for name, expected_hash in frozen['code_sha256'].items():
        if sha256(project / name) != expected_hash:
            raise ValueError('Frozen Sony inference source changed: ' + name)
    validate_sony_manifests(project)
    for split in COUNTS:
        if sha256(project / f'manifests/enhancement/sony_{split}.txt') != frozen['split_manifests_sha256'][split]:
            raise ValueError('Sony manifest changed')
    output = args.output_dir.resolve()
    if output == sealed:
        raise ValueError('Choose a separate reproduction output directory')
    if output.exists() and any(output.iterdir()) and not (output / 'binding.json').exists():
        raise ValueError('Use an empty output directory or resume this exact reproduction')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'reproduction.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared_lock = None
    if args.gpu_lock:
        shared_lock = args.gpu_lock.open('a')
        fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch

    from dip_course.cli.infer_official_sied import (
        average_prediction_samples,
        diffusion_seed,
        load_official_model,
        predict_official_sied,
    )
    from dip_course.enhancement.adaptive_lut import (
        AdaptiveColorLUT,
        apply_color_lut_chunked,
    )
    from dip_course.io import (
        RawSpec,
        load_packed_raw,
        load_rgb,
        read_pair_manifest,
        save_rgb,
    )
    from dip_course.metrics import (
        LPIPSMetric,
        color_fidelity_metrics,
        psnr_rgb,
        ssim_rgb,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('The archived FP16 Sony protocol requires a CUDA GPU')
    torch.set_num_threads(4)
    binding = {'purpose': 'fresh_sony_reproduction_without_training_caches',
               'sealed_binding_sha256': receipt['binding_sha256'], 'camera': 'sony',
               'protocol': frozen['protocol'], 'official_sha256': frozen['official_sha256'],
               'selected_model': selected, 'script_sha256': sha256(Path(__file__)),
               'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__)}
    binding_path = output / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Cannot mix a different model, source, GPU or protocol when resuming')
    if not binding_path.exists():
        write_json(binding_path, binding)
    binding_hash = sha256(binding_path)
    records = {s: read_pair_manifest(args.data_root, project /
               f'manifests/enhancement/sony_{s}.txt') for s in COUNTS}
    diffusion, _ = load_official_model(args.official_repo, args.official_checkpoint,
                                      sampling_steps=20, device=torch.device('cuda'))
    lut = None
    if args.lut_checkpoint:
        checkpoint = torch.load(args.lut_checkpoint, map_location='cuda', weights_only=True)
        if (checkpoint['model_type'] != 'adaptive_color_lut' or
                checkpoint['baseline_binding_sha256'] != frozen['selection']['baseline_binding_sha256']):
            raise ValueError('Sony LUT provenance changed')
        lut = AdaptiveColorLUT(**checkpoint['model_config']).cuda().eval()
        lut.load_state_dict(checkpoint['model'])
    perceptual = LPIPSMetric('cuda')
    if args.limit_new:
        records = {'train': records['train'][:3]}  # A resumed smoke run never advances into validation/test.
    rows = []
    for split, subset in records.items():
        for index, record in enumerate(subset):
            name = f'{index:04d}_{record.key}'
            row_path = output / f'rows/{split}/{name}.json'
            image_path = output / f'images/{split}/{name}.png'
            seeds = [diffusion_seed(3407, index, j) for j in range(3)]
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if (row['binding_sha256'] != binding_hash or row['key'] != record.key or
                        row['index'] != index or row['split'] != split or
                        row['seeds'] != ';'.join(map(str, seeds)) or sha256(image_path) != row['png_sha256']):
                    raise ValueError('Invalid reproduction resume row')
                summarize([row])
                rows.append(row)
                continue
            started = time.perf_counter()
            raw = torch.from_numpy(load_packed_raw(record.low_raw, RawSpec())).unsqueeze(0).cuda()
            with torch.inference_mode():
                samples = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    samples.append(predict_official_sied(diffusion, raw, precision='fp16'))
                prediction = average_prediction_samples(samples)
                if lut is not None:
                    prediction = apply_color_lut_chunked(lut, prediction)
            prediction = prediction[0].permute(1, 2, 0).float().cpu().numpy()
            target = load_rgb(record.gt_rgb)  # Used only after blind prediction.
            row = {'split': split, 'index': index, 'key': record.key,
                   'binding_sha256': binding_hash, 'seeds': ';'.join(map(str, seeds)),
                   'psnr': psnr_rgb(prediction, target), 'ssim': ssim_rgb(prediction, target),
                   **color_fidelity_metrics(prediction, target), 'lpips': perceptual(prediction, target)}
            summarize([row])
            save_rgb(image_path, prediction)
            row.update(png_sha256=sha256(image_path), wall_seconds=time.perf_counter() - started)
            write_json(row_path, row)
            rows.append(row)
            print(json.dumps({'completed': len(rows), 'split': split, 'key': record.key}), flush=True)
    if args.limit_new:
        print('TRAIN_ONLY_SMOKE_COMPLETE: no partial overall score', flush=True)
        return
    require_complete(rows, {s: [r.key for r in rs] for s, rs in records.items()})
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'purpose': binding['purpose'], 'count': 1000, 'overall': summarize(rows),
               'splits': {s: summarize([r for r in rows if r['split'] == s]) for s in COUNTS},
               'note': frozen['interpretation']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'verification.json', {'unique_count': 1000, 'split_counts': COUNTS,
               'binding_sha256': binding_hash, 'metrics_sha256': sha256(output / 'metrics.csv'),
               'summary_sha256': sha256(output / 'summary.json')})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
