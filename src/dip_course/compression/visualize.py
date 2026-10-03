from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    panel.paste(contained, (left, top))
    return panel


def build_compression_grid(
    *,
    originals: Mapping[str, Path],
    decoded_by_quality: Mapping[int, Mapping[str, Path]],
    keys: Sequence[str],
    output: str | Path,
    panel_size: tuple[int, int] = (300, 220),
) -> None:
    if not keys:
        raise ValueError("At least one visualization key is required")
    if not decoded_by_quality:
        raise ValueError("At least one JPEG quality is required")
    if panel_size[0] <= 0 or panel_size[1] <= 0:
        raise ValueError("panel dimensions must be positive")

    qualities = sorted(decoded_by_quality)
    headers = ["Original"] + [f"JPEG Q{quality}" for quality in qualities]
    header_height = 38
    label_width = 58
    canvas = Image.new(
        "RGB",
        (
            label_width + panel_size[0] * len(headers),
            header_height + panel_size[1] * len(keys),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text(
            (label_width + column * panel_size[0] + 10, 12),
            header,
            fill="black",
        )

    for row, key in enumerate(keys):
        if key not in originals:
            raise FileNotFoundError(f"Missing original image for visualization key {key}")
        paths = [originals[key]]
        for quality in qualities:
            quality_paths = decoded_by_quality[quality]
            if key not in quality_paths:
                raise FileNotFoundError(
                    f"Missing decoded Q{quality} image for visualization key {key}"
                )
            paths.append(quality_paths[key])

        y = header_height + row * panel_size[1]
        draw.text((10, y + 12), key, fill="black")
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                panel = _fit_panel(image, panel_size)
            canvas.paste(panel, (label_width + column * panel_size[0], y))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
