from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def reproduction(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / 'cloud'))
    spec = importlib.util.spec_from_file_location('sony_reproduction', root / 'cloud/reproduce_sony_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding():
    return {'camera': 'sony', 'exposure': '0.001', 'counts': {'train': 822, 'val': 91, 'test': 87},
            'protocol': {'precision': 'fp16', 'sampling_steps': 20, 'ensemble_size': 3,
                         'seed': 3407, 'native_rgb': [2160, 3840, 3]}}


def test_correct_sony_reproduction_binding(reproduction):
    reproduction.validate_binding(binding())


@pytest.mark.parametrize('field,value', [('camera', 'canon'), ('exposure', '0.0001'),
                         ('counts', {'train': 832, 'val': 92, 'test': 76})])
def test_reproduction_rejects_other_camera_settings(reproduction, field, value):
    wrong = binding()
    wrong[field] = value
    with pytest.raises(ValueError, match='Sony 0.001'):
        reproduction.validate_binding(wrong)


def test_reproduction_rejects_changed_sampling_protocol(reproduction):
    wrong = binding()
    wrong['protocol']['ensemble_size'] = 1
    with pytest.raises(ValueError, match='protocol'):
        reproduction.validate_binding(wrong)


def test_reproduction_rejects_changed_official_source(reproduction, tmp_path):
    content = b'original inference source\n'
    source = tmp_path / 'inference.py'
    source.write_bytes(content)
    expected = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    manifest = {'commit': 'fixed-commit', 'files': {'inference.py': expected}}
    reproduction.validate_official_source(tmp_path, manifest, 'fixed-commit')
    source.write_bytes(b'changed inference source\n')
    with pytest.raises(ValueError, match='source changed'):
        reproduction.validate_official_source(tmp_path, manifest, 'fixed-commit')
