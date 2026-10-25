"""Bounded Sony follow-up and equivalent parallel scoring for the user's deadline.

The original frozen inference/metric sources stay byte-identical. This executor
has its own signed receipt. It changes scheduling, never the scoring definition.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_sied1000 import METRICS, sha256, summarize
from prepare_sony1000 import COUNTS
from sony1000_pipeline import (
    Experiment,
    qualifies,
    require_complete,
    tensor_digest,
    write_json,
)


def brightness_training_loss(prediction, target):
    """Independent formulation of the ICCV2025 GT-Mean Gaussian/L1 mechanism.

    sigma=.1; detach only the mixture weight. Reference mean is used in training
    only. The deployed AdaptiveColorLUT has no reference/mean input.
    """
    import torch
    coefficient = prediction.new_tensor([.2989, .5870, .1140])[None, :, None, None]
    x = (prediction * coefficient).sum(1).mean((1, 2)).clamp_min(1e-6)
    y = (target * coefficient).sum(1).mean((1, 2)).clamp_min(1e-6)
    variance = .005 * (x.square() + y.square())
    divergence = .5 * torch.log(variance / (.01 * x * y)) + (x - y).square() / (8 * variance)
    mixing = divergence.clamp(0, 1).detach()
    normalized = (prediction * (y / x)[:, None, None, None]).clamp(0, 1)
    raw_error = (prediction - target).abs().mean((1, 2, 3))
    scaled_error = (normalized - target).abs().mean((1, 2, 3))
    return (mixing * raw_error + (1 - mixing) * scaled_error).mean()


class DeadlineExperiment(Experiment):
    def __init__(self, root, workers=2):
        super().__init__(root)
        self.workers = workers
        self.executor_hash = sha256(Path(__file__))

    def model(self, checkpoint):
        from dip_course.enhancement.adaptive_lut import AdaptiveColorLUT
        state = self.torch.load(checkpoint, map_location='cuda', weights_only=True)
        assert state['baseline_binding_sha256'] == self.binding_hash
        model = AdaptiveColorLUT(**state['model_config']).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        return model

    def base(self, split, index, record):
        import numpy as np
        started = time.perf_counter()
        source = self.load_prepared(split, index, record) if split != 'test' else None
        if split == 'val':
            prediction = np.load(self.prepared / source['native_val'], allow_pickle=False)
            seeds = source['seeds']
        elif source and source.get('native_train'):
            with np.load(self.prepared / source['native_train'], allow_pickle=False) as z:
                prediction = z['rgb']
            seeds = source['seeds']
        else:
            prediction, seeds = self.infer(split, index, record)
        if source and tensor_digest(prediction) != source['base_tensor_sha256']:
            raise ValueError('Native FP32 base no longer matches the frozen blind prediction')
        if prediction.shape != (2160, 3840, 3) or prediction.dtype != np.float32:
            raise ValueError('Native FP32 scoring required')
        return prediction, seeds, source, time.perf_counter() - started

    def score(self, model, jobs, directory=None, binding_hash=None, resume=False, deadline=None):
        from dip_course.enhancement.adaptive_lut import apply_color_lut_chunked
        from dip_course.io import save_rgb
        torch = self.torch
        pending, rows = deque(), []
        started_all = time.perf_counter()

        def worker(split, index, record, enhanced, prediction, seeds, source, started, load_seconds):
            metric_start = time.perf_counter()
            target = self.target(record)  # Blind GPU prediction is already complete.
            measured = self.metrics(enhanced, target)
            baseline = ({k: source[k] for k in METRICS} if source else self.metrics(prediction, target))
            metric_seconds = time.perf_counter() - metric_start
            row = {'split': split, 'index': index, 'key': record.key, 'seeds': seeds, **measured}
            if binding_hash:
                row.update(binding_sha256=binding_hash, baseline_metrics=baseline)
            if directory:
                name = f'{index:04d}_{record.key}'
                image_path = directory / ('images' if not binding_hash else f'images/{split}') / f'{name}.png'
                write_start = time.perf_counter()
                save_rgb(image_path, enhanced)
                row.update(png_sha256=sha256(image_path), png_seconds=time.perf_counter() - write_start)
                if binding_hash and split == 'test' and index < 6:
                    displays = directory / 'comparison_inputs'
                    save_rgb(displays / f'{name}_official.png', prediction)
                    (displays / 'references').mkdir(parents=True, exist_ok=True)
                    shutil.copy2(record.gt_rgb, displays / 'references' / f'{name}.jpg')
                row.update(execution_sha256=self.executor_hash, load_seconds=load_seconds,
                           metric_seconds=metric_seconds, wall_seconds=time.perf_counter() - started)
                row_path = directory / ('rows' if not binding_hash else f'rows/{split}') / f'{name}.json'
                write_json(row_path, row)
            return row

        def commit(row):
            rows.append(row)
            if binding_hash and resume:
                progress = {'state': 'parallel_final_sony_1000', 'completed': len(rows), 'total': 1000,
                            'last_split': row['split'], 'last_key': row['key'],
                            'execution_sha256': self.executor_hash,
                            'elapsed_seconds': time.perf_counter() - started_all}
                write_json(self.root / 'logs/progress.json', progress)
                print(json.dumps(progress), flush=True)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for split, index, record in jobs:
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError('Bounded follow-up validation budget reached')
                if resume:
                    name = f'{index:04d}_{record.key}'
                    path = directory / f'rows/{split}/{name}.json'
                    if path.exists():
                        row = json.loads(path.read_text())
                        if (row['binding_sha256'] != binding_hash or row['index'] != index or
                                row['key'] != record.key or row['split'] != split or
                                sha256(directory / f'images/{split}/{name}.png') != row['png_sha256']):
                            raise ValueError('Cannot mix prior rows from another frozen model')
                        summarize([row])
                        commit(row)
                        continue
                self.check_space()
                started = time.perf_counter()
                prediction, seeds, source, load_seconds = self.base(split, index, record)
                tensor = torch.from_numpy(prediction.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                with torch.inference_mode():
                    enhanced = apply_color_lut_chunked(model, tensor)[0].permute(1, 2, 0).cpu().numpy()
                pending.append(pool.submit(worker, split, index, record, enhanced, prediction,
                                           seeds, source, started, load_seconds))
                del tensor
                if len(pending) >= self.workers:
                    commit(pending.popleft().result())
            while pending:
                commit(pending.popleft().result())
        return rows, time.perf_counter() - started_all

    def pilot(self):
        choice = self.selection()
        model = self.model(self.root / choice['selected']['checkpoint'])
        jobs = [('train', i, r) for i, r in enumerate(self.records['train'][:4])]
        out = self.root / 'execution_pilot'
        rows, elapsed = self.score(model, jobs, out)
        serial_seconds, maximum = 0., {k: 0. for k in METRICS}
        for row in rows:
            name = f"{row['index']:04d}_{row['key']}"
            previous = json.loads((self.output / f'rows/train/{name}.json').read_text())
            if row['png_sha256'] != previous['png_sha256']:
                raise ValueError('Equivalent scheduler changed a native PNG')
            for field in METRICS:
                maximum[field] = max(maximum[field], abs(row[field] - previous[field]))
            serial_seconds += previous['wall_seconds']
        if any(v != 0 for v in maximum.values()):
            raise ValueError('Parallel scoring changed a metric')
        proof = {'equivalent': True, 'train_only_count': 4, 'all_8_metric_max_difference': maximum,
                 'png_sha256_equal': True, 'serial_four_seconds': serial_seconds,
                 'parallel_four_seconds': elapsed, 'short_run_ratio': serial_seconds / elapsed,
                 'workers': self.workers, 'executor_sha256': self.executor_hash,
                 'note': 'Four train images only. Not an end-to-end speed guarantee.'}
        write_json(self.root / 'logs/final_execution_equivalence.json', proof)
        print(json.dumps(proof, indent=2), flush=True)

    def research_followup(self):
        import numpy as np

        from dip_course.enhancement.efficiency import mean_image_psnr_loss
        torch = self.torch
        destination = self.root / 'research/gtmean_deadline'
        if (destination / 'result.json').exists():
            return
        if (destination / 'started.json').exists():
            raise RuntimeError('Follow-up already started; no automatic extra training')
        original = self.selection()
        if list((self.output / 'rows/test').glob('*.json')):
            raise ValueError('Follow-up selection is prohibited after held-out scoring starts')
        source_checkpoint = self.root / original['selected']['checkpoint']
        declaration = {'max_steps_per_arm': 250, 'arms': ['control', 'gtmean'], 'seed': 202610251840,
                       'learning_rate': 3e-5, 'batch_size': 16, 'precision': 'float32',
                       'gtmean_coefficient': 5., 'sigma': .1, 'regularization': .1,
                       'training_and_all_validations_max_minutes': 20, 'train_count': 822,
                       'warm_start_sha256': sha256(source_checkpoint), 'test_used': False,
                       'reference_paper': 'https://openaccess.thecvf.com/content/ICCV2025/html/Liao_GT-Mean_Loss_A_Simple_Yet_Effective_Solution_for_Brightness_Mismatch_ICCV_2025_paper.html',
                       'reference_code_commit': '1a09e2e077613c195ae7890005ffdfc8ce01dd4d',
                       'method': 'Warm-start LUT; log-MSE plus independently implemented GT-Mean L1. Not a full paper reproduction.',
                       'executor_sha256': self.executor_hash}
        write_json(destination / 'started.json', declaration)
        deadline = time.monotonic() + 20 * 60
        pairs = []
        for i, record in enumerate(self.records['train']):
            source = self.load_prepared('train', i, record)
            pair = np.load(self.prepared / source['color_pair'], allow_pickle=False)
            pairs.append(torch.from_numpy(pair.transpose(0, 3, 1, 2).copy()))
        resident = torch.stack(pairs).cuda()
        del pairs
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise MemoryError('Need 6 GiB GPU reserve beyond unchanged FP32 training pairs')
        pilot = original['pilot_indices']
        baseline = original['full_val_candidate']
        pilot_reference = self.torch.load(source_checkpoint, map_location='cpu', weights_only=True)['pilot']
        results, eligible = {}, []
        for arm in declaration['arms']:
            model = self.model(source_checkpoint).train()
            generator = torch.Generator(device='cpu').manual_seed(declaration['seed'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0, fused=True)
            training_started = time.perf_counter()
            with (destination / f'{arm}_training.jsonl').open('w') as log:
                for step in range(1, 251):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Follow-up training budget reached')
                    indices = torch.randint(822, (16,), generator=generator).cuda()
                    batch = resident[indices]
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model(batch[:, 0])
                    loss = mean_image_psnr_loss(predicted, batch[:, 1]) + .1 * model.regularization()
                    if arm == 'gtmean':
                        loss = loss + 5 * brightness_training_loss(predicted, batch[:, 1])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite follow-up loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    if step % 50 == 0:
                        log.write(json.dumps({'step': step, 'loss': float(loss.detach())}) + '\n')
                        log.flush()
            training_seconds = time.perf_counter() - training_started
            model.eval()
            checkpoint = destination / f'{arm}.pt'
            torch.save({'model_type': 'adaptive_color_lut', 'model_config': model.config,
                        'model': model.state_dict(), 'step': 1250, 'additional_steps': 250,
                        'baseline_binding_sha256': self.binding_hash, 'followup_arm': arm}, checkpoint)
            del optimizer
            jobs = [('val', i, self.records['val'][i]) for i in pilot]
            rows, _ = self.score(model, jobs, deadline=deadline)
            score = summarize(rows)
            results[arm] = {'pilot': score, 'checkpoint': str(checkpoint.relative_to(self.root)),
                            'checkpoint_sha256': sha256(checkpoint),
                            'training_seconds': training_seconds,
                            'pilot_qualified': qualifies(score, pilot_reference)}
            if results[arm]['pilot_qualified']:
                jobs = [('val', i, r) for i, r in enumerate(self.records['val'])]
                rows, wall = self.score(model, jobs, destination / f'{arm}_full_val', deadline=deadline)
                full = summarize(rows)
                write_json(destination / f'{arm}_full_val/metrics.json', rows)
                write_json(destination / f'{arm}_full_val/summary.json', full)
                results[arm].update(full_val=full, validation_seconds=wall,
                                    full_val_qualified=qualifies(full, baseline))
                if qualifies(full, baseline):
                    eligible.append((full['psnr_mean'], arm))
            write_json(destination / 'partial_result.json', results)
            del model
            torch.cuda.empty_cache()
        del resident
        winner = max(eligible)[1] if eligible else None
        result = {'arms': results, 'selected_followup': winner, 'test_used': False,
                  'previous_full_val': baseline, 'declaration': declaration,
                  'complete': True, 'selection_uses_all_91_native_validation_images': True}
        write_json(destination / 'result.json', result)
        if winner:
            # No held-out reference has been scored. Preserve the old partial run whole.
            archive = self.root / 'historical_sony_partial_before_followup'
            if archive.exists():
                raise RuntimeError('Historical partial run already exists')
            shutil.copy2(self.root / 'selection.json', destination / 'previous_selection.json')
            self.output.rename(archive)
            chosen = results[winner]
            selected = {**original, 'selected': {
                'name': f'official_sony_sied_plus_sony_lut_{winner}_refinement',
                'checkpoint': chosen['checkpoint'], 'checkpoint_sha256': chosen['checkpoint_sha256']},
                'full_val_candidate': chosen['full_val'], 'best_step': 1250,
                'previous_model_validation': baseline,
                'followup_result_sha256': sha256(destination / 'result.json')}
            write_json(self.root / 'selection.json', selected)
        print(json.dumps(result, indent=2), flush=True)

    def finish(self):
        proof = json.loads((self.root / 'logs/final_execution_equivalence.json').read_text())
        if not proof['equivalent'] or proof['executor_sha256'] != self.executor_hash:
            raise ValueError('The exact executor must pass the train-only equivalence pilot')
        choice = self.selection()
        model = self.model(self.root / choice['selected']['checkpoint'])
        binding = {**self.binding, 'selection': choice,
                   'selection_sha256': sha256(self.root / 'selection.json'),
                   'aggregation': 'Arithmetic mean of 1000 per-image PSNR and SSIM values'}
        path = self.output / 'binding.json'
        if path.exists() and json.loads(path.read_text()) != binding:
            raise ValueError('Cannot resume rows with another selected model')
        if not path.exists():
            write_json(path, binding)
        binding_hash = sha256(path)
        write_json(self.output / 'execution_receipt.json', {
            'executor_sha256': self.executor_hash, 'base_inference_code_sha256': self.binding['code_sha256'],
            'equivalence_proof_sha256': sha256(self.root / 'logs/final_execution_equivalence.json'),
            'model_binding_sha256': binding_hash, 'workers': self.workers,
            'note': 'Bounded CPU scoring/I/O workers. Native prediction, model, seeds, metrics and PNG encoding unchanged.'})
        jobs = [(s, i, r) for s, records in self.records.items() for i, r in enumerate(records)]
        rows, elapsed = self.score(model, jobs, self.output, binding_hash=binding_hash, resume=True)
        require_complete(rows, {s: [r.key for r in records] for s, records in self.records.items()})
        base_rows = [{'split': r['split'], 'key': r['key'], **r['baseline_metrics']} for r in rows]
        summary = {'camera': 'sony', 'count': 1000, 'binding_sha256': binding_hash,
                   'primary_metrics': ['psnr_mean', 'ssim_mean'], 'overall': summarize(rows),
                   'splits': {s: summarize([r for r in rows if r['split'] == s]) for s in COUNTS},
                   'official_baseline_overall': summarize(base_rows), 'selected': choice['selected'],
                   'interpretation': self.binding['interpretation'], 'execution_wall_seconds': elapsed}
        for name, data in [('metrics.csv', rows), ('baseline_metrics.csv', base_rows)]:
            fields = list(dict.fromkeys(k for row in data for k in row if k != 'baseline_metrics'))
            with (self.output / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(data)
        write_json(self.output / 'summary.json', summary)
        write_json(self.output / 'verification.json', {
            'unique_count': 1000, 'split_counts': COUNTS, 'binding_sha256': binding_hash,
            'metrics_sha256': sha256(self.output / 'metrics.csv'),
            'baseline_metrics_sha256': sha256(self.output / 'baseline_metrics.csv'),
            'summary_sha256': sha256(self.output / 'summary.json'),
            'selection_sha256': sha256(self.root / 'selection.json'),
            'execution_receipt_sha256': sha256(self.output / 'execution_receipt.json')})
        write_json(self.root / 'logs/progress.json', {'state': 'complete_pending_verified_backup', 'count': 1000})
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['pilot', 'research', 'finalize'], required=True)
    args = parser.parse_args()
    with (args.root / 'sony_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment = DeadlineExperiment(args.root)
        {'pilot': experiment.pilot, 'research': experiment.research_followup, 'finalize': experiment.finish}[args.phase]()


if __name__ == '__main__':
    main()
