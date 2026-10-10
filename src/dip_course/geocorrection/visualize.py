from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _panel(image: Image.Image, size: tuple[int, int], *, binary: bool = False) -> Image.Image:
    source = image.convert("L" if binary else "RGB")
    contained = ImageOps.contain(source, size, Image.Resampling.NEAREST if binary else Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained.convert("RGB"), (left, top))
    return canvas


def build_geometry_grid(
    *,
    originals: Mapping[str, Path],
    masks: Mapping[str, Path],
    homographies: Mapping[str, Path],
    corrected: Mapping[str, Path],
    keys: Sequence[str],
    output: str | Path,
    panel_size: tuple[int, int] = (330, 235),
) -> None:
    if not keys:
        raise ValueError("At least one visualization key is required")
    headers = ("Input", "Dataset mask", "Homography", "Boundary dewarp")
    label_width, header_height = 170, 38
    canvas = Image.new(
        "RGB",
        (
            label_width + len(headers) * panel_size[0],
            header_height + len(keys) * panel_size[1],
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((label_width + column * panel_size[0] + 8, 12), header, fill="black")
    for row, key in enumerate(keys):
        sources = (
            (originals, False),
            (masks, True),
            (homographies, False),
            (corrected, False),
        )
        y = header_height + row * panel_size[1]
        draw.text((6, y + 12), key[:26], fill="black")
        for column, (mapping, binary) in enumerate(sources):
            if key not in mapping:
                raise FileNotFoundError(f"Missing visualization image for {key}")
            with Image.open(mapping[key]) as image:
                panel = _panel(image, panel_size, binary=binary)
            canvas.paste(panel, (label_width + column * panel_size[0], y))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
