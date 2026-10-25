"""Sony-only SIED adaptation and one frozen, native-resolution 1000-image submission.

Prepare 822 train and 91 validation images, run one bounded LUT experiment, freeze
the choice on all 91 validation images, then evaluate the same model on all 1000.
The 87 held-out references are not opened until the final choice has been saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_sied1000 import METRICS, make_color_training_pair, sha256, summarize
from prepare_sony1000 import COUNTS, OFFICIAL_COMMIT, validate_sony_manifests


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(partial, path)


def require_complete(rows: list[dict], expected: dict[str, list[str]]) -> None:
    if len(rows) != 1000 or len({r['key'] for r in rows}) != 1000:
        raise ValueError('A submission needs all 1000 unique Sony images')
    if {s: len(k) for s, k in expected.items()} != COUNTS:
        raise ValueError('Wrong Sony partition')
    for split, keys in expected.items():
        subset = [r for r in rows if r['split'] == split]
        if [r['key'] for r in subset] != keys or [r['index'] for r in subset] != list(range(len(keys))):
            raise ValueError('Wrong submission split, order or image identity')
    if len({r['binding_sha256'] for r in rows}) != 1:
        raise ValueError('Cannot mix models in a submission')
    for row in rows:
        expected_seeds = ';'.join(str(3407 + row['index'] + j * 1000003) for j in range(3))
        if row['seeds'] != expected_seeds:
            raise ValueError('Incorrect split-local Sony diffusion seeds')
    summarize(rows)


def qualifies(candidate: dict, baseline: dict) -> bool:
    if candidate.get('count') != baseline.get('count'):
        return False
    fields = list(METRICS.values())
    if not all(math.isfinite(float(candidate.get(f, float('nan')))) for f in fields):
        return False
    if candidate['psnr_mean'] < baseline['psnr_mean'] + 0.02:
        return False
    if candidate['ssim_mean'] < baseline['ssim_mean'] - 0.002:
        return False
    return all(candidate[f] <= baseline[f] * 1.02 for f in fields
               if f not in {'psnr_mean', 'ssim_mean'})


def tensor_digest(array) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def array_save(path: Path, array) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    with partial.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(partial, path)


def compressed_array_save(path: Path, array) -> None:
    """Lossless native FP32 cache, with a fast standard-library compression level."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    with (
        zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z,
        z.open('rgb.npy', 'w', force_zip64=True) as stream,
    ):
        np.lib.format.write_array(stream, array, allow_pickle=False)
    os.replace(partial, path)


class Experiment:
    def __init__(self, root: Path, limit_new: int = 0):
        import torch

        from dip_course.io import RawSpec, read_pair_manifest
        from dip_course.metrics import LPIPSMetric
        self.torch = torch
        self.cache_lock = threading.Lock()
        self.lpips_lock = threading.Lock()
        self.limit_new = limit_new
        self.root = root.resolve()
        self.project = self.root / 'project'
        validate_sony_manifests(self.project)
        self.data = self.root / 'data/SIED_dataset'
        self.records = {s: read_pair_manifest(self.data, self.project /
                        f'manifests/enhancement/sony_{s}.txt') for s in COUNTS}
        self.spec = RawSpec()
        self.prepared = self.root / 'prepared'
        self.research = self.root / 'research/sony_lut'
        self.output = self.root / 'submission1000_native'
        self.native_cache_bytes = sum(p.stat().st_size for p in
                                      (self.prepared / 'native_train').glob('*.npz'))
        self.checkpoint = self.root / 'checkpoints/official/sony_0.001.pt'
        self.data_receipt = json.loads((self.root / 'logs/sony_data_verified.json').read_text())
        if self.data_receipt['camera'] != 'sony' or self.data_receipt['count'] != 1000:
            raise ValueError('Only the audited Sony dataset is allowed')
        if not torch.cuda.is_available():
            raise RuntimeError('Native Sony inference requires the allocated CUDA GPU')
        torch.set_num_threads(4)
        self.device = torch.device('cuda')
        self.protocol = {'sampling_steps': 20, 'ensemble_size': 3, 'precision': 'fp16',
                         'seed': 3407, 'seed_indexing': 'original split-local index',
                         'native_rgb': [2160, 3840, 3], 'metric_input': 'float32 before PNG encoding'}
        code_names = ['cloud/sony1000_pipeline.py', 'cloud/prepare_sony1000.py',
                      'cloud/evaluate_sied1000.py', 'src/dip_course/cli/infer_official_sied.py',
                      'src/dip_course/enhancement/adaptive_lut.py',
                      'src/dip_course/io.py', 'src/dip_course/metrics.py']
        import importlib.metadata
        self.binding = {
            'camera': 'sony', 'exposure': '0.001', 'counts': COUNTS,
            'official_commit': OFFICIAL_COMMIT, 'official_sha256': sha256(self.checkpoint),
            'protocol': self.protocol,
            'train_cache': 'lossless FP32 deflate1, at most 40 GiB; uncached bases are regenerated',
            'preparation_execution': 'one seeded diffusion producer, two bounded metric/I/O workers',
            'all_manifest_sha256': sha256(self.project / 'manifests/source/sony_all.txt'),
            'split_manifests_sha256': {s: sha256(self.project / f'manifests/enhancement/sony_{s}.txt')
                                     for s in COUNTS},
            'data_receipt_sha256': sha256(self.root / 'logs/sony_data_verified.json'),
            'code_sha256': {n: sha256(self.project / n) for n in code_names},
            'gpu': torch.cuda.get_device_name(0),
            'packages': {p: importlib.metadata.version(p) for p in
                         ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-image', 'lpips']},
            'interpretation': 'Overall includes 822 train and 91 official-pretraining-exposed val images; '
                              'not independent generalization. Report the 87 test images separately.'}
        if self.binding['official_sha256'] != self.data_receipt['checkpoint_sha256']:
            raise ValueError('Official Sony checkpoint changed after data preparation')
        self.prepared.mkdir(parents=True, exist_ok=True)
        binding_path = self.prepared / 'binding.json'
        if binding_path.exists() and json.loads(binding_path.read_text()) != self.binding:
            raise ValueError('Cannot change source, protocol, data or GPU in an existing Sony run')
        if not binding_path.exists():
            write_json(binding_path, self.binding)
        self.binding_hash = sha256(binding_path)
        self.diffusion = None
        self.lpips = LPIPSMetric('cuda')

    def check_space(self):
        if shutil.disk_usage(self.root).free < 15 * 1024**3 + 256 * 1024**2:
            raise OSError('Stopping before the 15 GiB free-space reserve')

    def infer(self, split, index, record):
        from dip_course.cli.infer_official_sied import (
            average_prediction_samples,
            diffusion_seed,
            load_official_model,
            predict_official_sied,
        )
        from dip_course.io import load_packed_raw
        torch = self.torch
        if self.diffusion is None:
            self.diffusion, _ = load_official_model(
                self.root / 'third_party/SIED-official', self.checkpoint,
                sampling_steps=20, device=self.device)
        relative = record.low_raw.relative_to(self.data).as_posix()
        if sha256(record.low_raw) != self.data_receipt['files'][relative]['sha256']:
            raise ValueError(f'Sony input RAW changed: {record.key}')
        raw = torch.from_numpy(load_packed_raw(record.low_raw, self.spec)).unsqueeze(0).cuda()
        seeds = [diffusion_seed(3407, index, j) for j in range(3)]
        with torch.inference_mode():
            samples = []
            for seed in seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                samples.append(predict_official_sied(self.diffusion, raw, precision='fp16'))
            image = average_prediction_samples(samples)
        prediction = image[0].permute(1, 2, 0).float().cpu().numpy()
        del raw, samples, image
        if prediction.shape != (2160, 3840, 3):
            raise ValueError('Only native 3840 by 2160 predictions may be scored')
        return prediction, ';'.join(map(str, seeds))

    def metrics(self, prediction, target):
        from dip_course.metrics import color_fidelity_metrics, psnr_rgb, ssim_rgb
        result = {'psnr': psnr_rgb(prediction, target), 'ssim': ssim_rgb(prediction, target),
                  **color_fidelity_metrics(prediction, target)}
        # LPIPS is shared across CPU workers; only its GPU forward is serialized.
        with self.lpips_lock:
            result['lpips'] = self.lpips(prediction, target)
        if not all(math.isfinite(float(v)) for v in result.values()):
            raise ValueError('Nonfinite image metric; refusing to drop the image')
        return result

    def target(self, record):
        from dip_course.io import load_rgb
        relative = record.gt_rgb.relative_to(self.data).as_posix()
        if sha256(record.gt_rgb) != self.data_receipt['files'][relative]['sha256']:
            raise ValueError(f'Sony reference changed: {record.key}')
        return load_rgb(record.gt_rgb)

    def load_prepared(self, split, index, record):
        path = self.prepared / f'rows/{split}/{index:04d}_{record.key}.json'
        row = json.loads(path.read_text())
        if (row['binding_sha256'] != self.binding_hash or row['split'] != split or
                row['key'] != record.key or row['index'] != index):
            raise ValueError('Prepared Sony row identity or binding mismatch')
        for field in ['png', 'color_pair', 'native_val', 'native_train']:
            if row.get(field) and sha256(self.prepared / row[field]) != row[field + '_sha256']:
                raise ValueError(f'Prepared Sony artifact changed: {field}/{record.key}')
        summarize([row])
        return row

    def prepare_row(self, split, index, record, prediction, seeds, started, inference_seconds):
        from dip_course.io import save_rgb

        name = f'{index:04d}_{record.key}'
        target = self.target(record)  # Already-predicted RGB only; no GT-conditioned inference.
        row = {'split': split, 'index': index, 'key': record.key,
               'binding_sha256': self.binding_hash, 'seeds': seeds,
               'inference_seconds': inference_seconds,
               'base_tensor_sha256': tensor_digest(prediction),
               **self.metrics(prediction, target)}
        artifacts = {'color_pair': (f'color_pairs/{split}/{name}.npy',
                                   make_color_training_pair(prediction, target))}
        if split == 'val':
            artifacts['native_val'] = (f'native_val/{name}.npy', prediction)
        for field, (relative, array) in artifacts.items():
            array_save(self.prepared / relative, array)
            row[field] = relative
            row[field + '_sha256'] = sha256(self.prepared / relative)
        reserve = 0
        with self.cache_lock:
            if split == 'train' and self.native_cache_bytes + prediction.nbytes < 40 * 1024**3:
                reserve = prediction.nbytes
                self.native_cache_bytes += reserve
        if reserve:
            cache_started = time.perf_counter()
            relative = f'native_train/{name}.npz'
            compressed_array_save(self.prepared / relative, prediction)
            row['native_train'] = relative
            row['native_train_sha256'] = sha256(self.prepared / relative)
            cache_size = (self.prepared / relative).stat().st_size
            with self.cache_lock:
                self.native_cache_bytes += cache_size - reserve
            row['native_cache_bytes'] = cache_size
            row['native_cache_seconds'] = time.perf_counter() - cache_started
        row['png'] = f'images/{split}/{name}.png'
        save_rgb(self.prepared / row['png'], prediction)
        row['png_sha256'] = sha256(self.prepared / row['png'])
        row['wall_seconds'] = time.perf_counter() - started
        write_json(self.prepared / f'rows/{split}/{name}.json', row)
        return row

    def prepare(self):
        completed, new_count, scheduled = 0, 0, 0
        pending = deque()
        run_started = time.perf_counter()

        def commit(row):
            nonlocal completed, new_count
            completed += 1
            new_count += 1
            progress = {'state': 'preparing_train_val', 'completed': completed, 'total': 913,
                        'last_split': row['split'], 'last_key': row['key'],
                        'wall_seconds': row['wall_seconds'],
                        'new_images_per_second': new_count / (time.perf_counter() - run_started)}
            write_json(self.root / 'logs/progress.json', progress)
            print(json.dumps(progress), flush=True)

        # Bounded futures keep at most two metric/I/O jobs and one producer image in RAM.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for split in ['train', 'val']:
                for index, record in enumerate(self.records[split]):
                    path = self.prepared / f'rows/{split}/{index:04d}_{record.key}.json'
                    if path.exists():
                        self.load_prepared(split, index, record)
                        completed += 1
                        continue
                    if self.limit_new and scheduled >= self.limit_new:
                        while pending:
                            commit(pending.popleft().result())
                        print('TRAIN_PILOT_ONLY: no partial overall score', flush=True)
                        return
                    self.check_space()
                    started = time.perf_counter()
                    prediction, seeds = self.infer(split, index, record)
                    inference_seconds = time.perf_counter() - started
                    pending.append(pool.submit(self.prepare_row, split, index, record,
                                               prediction, seeds, started, inference_seconds))
                    scheduled += 1
                    if len(pending) >= 2:
                        commit(pending.popleft().result())
            while pending:
                commit(pending.popleft().result())
        write_json(self.prepared / 'complete.json', {'count': completed, 'binding_sha256': self.binding_hash})

    def validate(self, model, indices, out):
        import numpy as np

        from dip_course.enhancement.adaptive_lut import apply_color_lut_chunked
        from dip_course.io import save_rgb
        rows = []
        for index in indices:
            record = self.records['val'][index]
            source = self.load_prepared('val', index, record)
            prediction = np.load(self.prepared / source['native_val'], allow_pickle=False)
            if prediction.shape != (2160, 3840, 3) or prediction.dtype != np.float32:
                raise ValueError('Validation needs exact native FP32 predictions')
            tensor = self.torch.from_numpy(prediction.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
            with self.torch.inference_mode():
                tensor = apply_color_lut_chunked(model, tensor)
            candidate = tensor[0].permute(1, 2, 0).cpu().numpy()
            row = {'split': 'val', 'index': index, 'key': record.key,
                   **self.metrics(candidate, self.target(record))}
            if out is not None:
                name = f'{index:04d}_{record.key}'
                save_rgb(out / 'images' / f'{name}.png', candidate)
                write_json(out / 'rows' / f'{name}.json', row)
            rows.append(row)
        summary = summarize(rows)
        if out is not None:
            write_json(out / 'metrics.json', rows)
            write_json(out / 'summary.json', summary)
        return summary

    def research_color(self):
        import numpy as np

        from dip_course.enhancement.adaptive_lut import AdaptiveColorLUT
        from dip_course.enhancement.efficiency import mean_image_psnr_loss
        torch = self.torch
        selected = self.root / 'selection.json'
        if selected.exists():
            self.selection()
            return
        if not (self.prepared / 'complete.json').exists():
            raise ValueError('Complete train/validation preparation first')
        started_file = self.research / 'started.json'
        if started_file.exists():
            raise RuntimeError('Interrupted color experiment: inspect its evidence before explicitly resuming')
        self.research.mkdir(parents=True, exist_ok=True)
        write_json(started_file, {'max_steps': 1000, 'max_training_wall_minutes': 45,
                                 'train_count': 822, 'val_count': 91, 'test_used': False,
                                 'baseline_binding_sha256': self.binding_hash})
        torch.manual_seed(20261025)
        generator = torch.Generator(device='cpu').manual_seed(20261025)
        pilot = sorted(random.Random(20261025).sample(range(91), 12))
        baseline_rows = [self.load_prepared('val', i, self.records['val'][i]) for i in range(91)]
        pilot_baseline = summarize([baseline_rows[i] for i in pilot])
        full_baseline = summarize(baseline_rows)
        pairs = []
        for i, record in enumerate(self.records['train']):
            row = self.load_prepared('train', i, record)
            pair = np.load(self.prepared / row['color_pair'], allow_pickle=False)
            if pair.shape != (2, 288, 512, 3) or pair.dtype != np.float32:
                raise ValueError('Invalid Sony training pair')
            pairs.append(torch.from_numpy(pair.transpose(0, 3, 1, 2).copy()))
        resident = torch.stack(pairs).cuda()
        del pairs
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise OSError('Less than 6 GiB GPU headroom after loading training thumbnails')
        model = AdaptiveColorLUT().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, fused=True)
        timer = time.monotonic()
        best_psnr, best_step, bad, best_path = pilot_baseline['psnr_mean'], 0, 0, None
        log = (self.research / 'training.jsonl').open('a')
        for step in range(1, 1001):
            if time.monotonic() - timer > 45 * 60:
                break
            index = torch.randint(822, (16,), generator=generator).cuda()
            batch = resident[index]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = mean_image_psnr_loss(model(batch[:, 0]), batch[:, 1]) + 0.1 * model.regularization()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Sony training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 50 == 0:
                log.write(json.dumps({'step': step, 'loss': float(loss.detach()),
                                      'elapsed_seconds': time.monotonic() - timer}) + '\n')
                log.flush()
            if step % 250 == 0:
                model.eval()
                score = self.validate(model, pilot, None)
                accepted = qualifies(score, pilot_baseline) and score['psnr_mean'] > best_psnr
                bad = 0 if accepted else bad + 1
                if accepted:
                    best_psnr, best_step = score['psnr_mean'], step
                    best_path = self.research / 'best.pt'
                    torch.save({'model_type': 'adaptive_color_lut', 'model_config': model.config,
                                'model': model.state_dict(), 'step': step, 'pilot': score,
                                'baseline_binding_sha256': self.binding_hash}, best_path)
                progress = {'state': 'training_sony_lut', 'step': step, 'max_steps': 1000,
                            'pilot': score, 'best_step': best_step, 'test_used': False}
                write_json(self.root / 'logs/progress.json', progress)
                write_json(self.research / f'validation_{step}.json', score)
                print(json.dumps(progress), flush=True)
                if step >= 750 and bad >= 3:
                    break
        log.close()
        del resident, optimizer
        torch.cuda.empty_cache()
        choice = {'name': 'official_sony_sied', 'checkpoint': None, 'checkpoint_sha256': None}
        full_score = None
        if best_path is not None:
            checkpoint = torch.load(best_path, map_location='cuda', weights_only=True)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            full_score = self.validate(model, range(91), self.research / 'full_val')
            if qualifies(full_score, full_baseline):
                choice = {'name': 'official_sony_sied_plus_sony_lut',
                          'checkpoint': best_path.relative_to(self.root).as_posix(),
                          'checkpoint_sha256': sha256(best_path)}
        decision = {'baseline_binding_sha256': self.binding_hash, 'selected': choice,
                    'pilot_indices': pilot, 'full_val_baseline': full_baseline,
                    'full_val_candidate': full_score, 'best_step': best_step,
                    'training_seconds': time.monotonic() - timer,
                    'test_used_for_selection': False, 'selection_uses_all_91_native_validation_images': True}
        write_json(self.research / 'result.json', decision)
        write_json(selected, decision)  # Freeze before any held-out reference is loaded.

    def selection(self):
        path = self.root / 'selection.json'
        choice = json.loads(path.read_text())
        if choice['baseline_binding_sha256'] != self.binding_hash or choice['test_used_for_selection']:
            raise ValueError('Sony final model is not frozen to this experiment')
        selected = choice['selected']
        if selected['checkpoint'] and sha256(self.root / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise ValueError('Selected Sony LUT checkpoint changed')
        return choice

    def finalize(self):
        import numpy as np

        from dip_course.enhancement.adaptive_lut import (
            AdaptiveColorLUT,
            apply_color_lut_chunked,
        )
        from dip_course.io import save_rgb
        choice = self.selection()
        binding = {**self.binding, 'selection': choice,
                   'selection_sha256': sha256(self.root / 'selection.json'),
                   'aggregation': 'Arithmetic mean of 1000 per-image PSNR and SSIM values'}
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / 'binding.json'
        if path.exists() and json.loads(path.read_text()) != binding:
            raise ValueError('Cannot change Sony selection after final scoring has started')
        if not path.exists():
            write_json(path, binding)
        final_hash = sha256(path)
        model = None
        if choice['selected']['checkpoint']:
            checkpoint = self.torch.load(self.root / choice['selected']['checkpoint'],
                                         map_location='cuda', weights_only=True)
            model = AdaptiveColorLUT(**checkpoint['model_config']).cuda().eval()
            model.load_state_dict(checkpoint['model'])
        rows, base_rows = [], []
        for split, records in self.records.items():
            for index, record in enumerate(records):
                name = f'{index:04d}_{record.key}'
                row_path = self.output / f'rows/{split}/{name}.json'
                image_path = self.output / f'images/{split}/{name}.png'
                if row_path.exists():
                    row = json.loads(row_path.read_text())
                    if (row['binding_sha256'] != final_hash or row['key'] != record.key or
                            row['split'] != split or row['index'] != index or
                            row['png_sha256'] != sha256(image_path)):
                        raise ValueError('Final Sony resume identity or model mismatch')
                    summarize([row])
                    rows.append(row)
                    base_rows.append({'split': split, 'key': record.key, **row['baseline_metrics']})
                    continue
                self.check_space()
                started = time.perf_counter()
                prepared = self.load_prepared(split, index, record) if split != 'test' else None
                if model is None and prepared is not None:
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(self.prepared / prepared['png'], image_path) if not image_path.exists() else None
                    measured = {k: prepared[k] for k in METRICS}
                    baseline = measured
                    seeds = prepared['seeds']
                else:
                    if split == 'val':
                        prediction = np.load(self.prepared / prepared['native_val'], allow_pickle=False)
                        seeds = prepared['seeds']
                    elif prepared is not None and prepared.get('native_train'):
                        with np.load(self.prepared / prepared['native_train'], allow_pickle=False) as cache:
                            prediction = cache['rgb']
                        if tensor_digest(prediction) != prepared['base_tensor_sha256']:
                            raise ValueError('Lossless Sony cache does not match the original prediction')
                        seeds = prepared['seeds']
                    else:
                        prediction, seeds = self.infer(split, index, record)
                        if prepared and tensor_digest(prediction) != prepared['base_tensor_sha256']:
                            raise ValueError('Repeated blind Sony base is not numerically identical')
                    if model is not None:
                        tensor = self.torch.from_numpy(prediction.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                        with self.torch.inference_mode():
                            enhanced = apply_color_lut_chunked(model, tensor)[0].permute(1, 2, 0).cpu().numpy()
                    else:
                        enhanced = prediction
                    target = self.target(record)
                    baseline = ({k: prepared[k] for k in METRICS} if prepared else
                                self.metrics(prediction, target))
                    measured = self.metrics(enhanced, target) if model is not None else baseline
                    save_rgb(image_path, enhanced)
                    if split == 'test' and index < 6:
                        # Fixed first-six display samples; never used by model selection.
                        displays = self.output / 'comparison_inputs'
                        save_rgb(displays / f'{name}_official.png', prediction)
                        (displays / 'references').mkdir(parents=True, exist_ok=True)
                        shutil.copy2(record.gt_rgb, displays / 'references' / f'{name}.jpg')
                row = {'split': split, 'index': index, 'key': record.key, 'binding_sha256': final_hash,
                       'seeds': seeds, **measured, 'baseline_metrics': baseline,
                       'png_sha256': sha256(image_path), 'wall_seconds': time.perf_counter() - started}
                write_json(row_path, row)
                rows.append(row)
                base_rows.append({'split': split, 'key': record.key, **baseline})
                progress = {'state': 'final_sony_1000', 'completed': len(rows), 'total': 1000,
                            'last_split': split, 'last_key': record.key, 'wall_seconds': row['wall_seconds']}
                write_json(self.root / 'logs/progress.json', progress)
                print(json.dumps(progress), flush=True)
        require_complete(rows, {s: [r.key for r in rs] for s, rs in self.records.items()})
        summary = {'camera': 'sony', 'count': 1000, 'binding_sha256': final_hash,
                   'primary_metrics': ['psnr_mean', 'ssim_mean'], 'overall': summarize(rows),
                   'splits': {s: summarize([r for r in rows if r['split'] == s]) for s in COUNTS},
                   'official_baseline_overall': summarize(base_rows),
                   'selected': choice['selected'], 'interpretation': self.binding['interpretation']}
        with (self.output / 'metrics.csv').open('w', newline='') as stream:
            fields = [k for k in rows[0] if k != 'baseline_metrics']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        with (self.output / 'baseline_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(base_rows[0]))
            writer.writeheader()
            writer.writerows(base_rows)
        write_json(self.output / 'summary.json', summary)
        write_json(self.output / 'verification.json', {
            'unique_count': 1000, 'split_counts': COUNTS, 'binding_sha256': final_hash,
            'metrics_sha256': sha256(self.output / 'metrics.csv'),
            'baseline_metrics_sha256': sha256(self.output / 'baseline_metrics.csv'),
            'summary_sha256': sha256(self.output / 'summary.json'),
            'selection_sha256': sha256(self.root / 'selection.json')})
        write_json(self.root / 'logs/progress.json', {'state': 'complete_pending_verified_backup', 'count': 1000})
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'research', 'finalize'], required=True)
    parser.add_argument('--limit-new', type=int, choices=[0, 3, 12], default=0)
    args = parser.parse_args()
    if args.limit_new and args.phase != 'prepare':
        parser.error('--limit-new is a train-only preparation smoke run')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'sony_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiment = Experiment(args.root, args.limit_new)
    {'prepare': experiment.prepare, 'research': experiment.research_color,
     'finalize': experiment.finalize}[args.phase]()


if __name__ == '__main__':
    main()
