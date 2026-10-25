from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def exporter(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / 'cloud'))
    spec = importlib.util.spec_from_file_location('sony_visualizations', root / 'cloud/export_sony_visualizations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_requires_completed_and_verified_1000(exporter, tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter.export(tmp_path)
    assert not (tmp_path / 'report_assets').exists()


def test_display_crop_keeps_geometry_without_upscaling(exporter, tmp_path):
    image = Image.new('RGB', (20, 12), (20, 100, 160))
    path = tmp_path / 'crop.jpg'
    receipt = exporter.save_display(image, path, crop=(4, 3, 16, 9))
    with Image.open(path) as cropped:
        assert cropped.size == (12, 6)
    assert receipt['native_size'] == (20, 12)
    assert receipt['display_size'] == (12, 6)
    assert receipt['display_sha256'] == exporter.sha256(path)


def test_display_rejects_crop_outside_native_frame(exporter, tmp_path):
    image = Image.new('RGB', (20, 12))
    path = tmp_path / 'invalid.jpg'
    with pytest.raises(ValueError, match='native image'):
        exporter.save_display(image, path, crop=(0, 0, 21, 12))
    assert not path.exists()


def test_real_raw_preview_calls_use_the_explicit_sensor_spec(exporter, tmp_path):
    import numpy as np

    from dip_course.io import RawSpec
    spec = RawSpec(width=8, height=8)
    raw = np.linspace(512, 6000, 64).reshape(8, 8).astype('<u2')
    path = tmp_path / 'sensor.raw'
    path.write_bytes(raw.tobytes())
    preview, isp, parameters = exporter.raw_display_views(path, spec)
    assert preview.shape == isp.shape == (8, 8, 3)
    assert np.isfinite(preview).all() and np.isfinite(isp).all()
    assert float(preview.max()) > 0 and parameters['gamma'] == 1.8
