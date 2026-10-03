from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

_SERIES_COLORS = ("#2468B4", "#D0502D", "#378A4A", "#8B5FBF")


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
    codec_label: str = "JPEG",
) -> None:
    if not keys:
        raise ValueError("At least one visualization key is required")
    if not decoded_by_quality:
        raise ValueError("At least one JPEG quality is required")
    if panel_size[0] <= 0 or panel_size[1] <= 0:
        raise ValueError("panel dimensions must be positive")

    qualities = sorted(decoded_by_quality)
    headers = ["Original"] + [f"{codec_label} Q{quality}" for quality in qualities]
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


def build_rate_distortion_plot(
    *,
    series: Mapping[str, Sequence[Mapping[str, float]]],
    output: str | Path,
    size: tuple[int, int] = (1000, 460),
) -> None:
    """Draw PSNR/bpp and SSIM/bpp curves without a plotting dependency."""
    if not series or any(not points for points in series.values()):
        raise ValueError("Rate-distortion plot requires non-empty series")
    if size[0] < 600 or size[1] < 300:
        raise ValueError("Rate-distortion plot is too small")
    required = {"quality", "bits_per_pixel_mean", "psnr_mean", "ssim_mean"}
    for points in series.values():
        for point in points:
            if not required.issubset(point):
                raise ValueError("Rate-distortion point is missing required metrics")

    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    outer_margin = 54
    gap = 64
    panel_width = (size[0] - 2 * outer_margin - gap) // 2
    panel_height = size[1] - 2 * outer_margin
    all_points = [point for points in series.values() for point in points]
    x_max = max(float(point["bits_per_pixel_mean"]) for point in all_points) * 1.08
    panels = (
        ("psnr_mean", "PSNR (dB)", 1.0),
        ("ssim_mean", "SSIM", 0.02),
    )

    for panel_index, (metric, title, padding) in enumerate(panels):
        left = outer_margin + panel_index * (panel_width + gap)
        top = outer_margin
        right = left + panel_width
        bottom = top + panel_height
        values = [float(point[metric]) for point in all_points]
        y_min = min(values) - padding
        y_max = max(values) + padding
        draw.rectangle((left, top, right, bottom), outline="#333333", width=1)
        draw.text((left, 18), f"{title} vs bits per pixel", fill="black")
        draw.text((left + panel_width // 2 - 24, bottom + 18), "bits/pixel", fill="black")
        draw.text((left - 42, top - 18), f"{y_max:.3f}", fill="#555555")
        draw.text((left - 42, bottom - 7), f"{y_min:.3f}", fill="#555555")
        draw.text((left - 3, bottom + 5), "0", fill="#555555")
        draw.text((right - 32, bottom + 5), f"{x_max:.2f}", fill="#555555")

        for series_index, (label, points) in enumerate(series.items()):
            color = _SERIES_COLORS[series_index % len(_SERIES_COLORS)]
            ordered = sorted(points, key=lambda point: point["bits_per_pixel_mean"])
            coordinates: list[tuple[int, int]] = []
            for point in ordered:
                x = left + round(
                    float(point["bits_per_pixel_mean"]) / x_max * panel_width
                )
                y = bottom - round(
                    (float(point[metric]) - y_min) / (y_max - y_min) * panel_height
                )
                coordinates.append((x, y))
            if len(coordinates) > 1:
                draw.line(coordinates, fill=color, width=3)
            for coordinate, point in zip(coordinates, ordered, strict=True):
                x, y = coordinate
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
                draw.text((x + 6, y - 13), f"Q{int(point['quality'])}", fill=color)
            legend_y = top + 14 + series_index * 20
            draw.line((right - 115, legend_y, right - 91, legend_y), fill=color, width=3)
            draw.text((right - 84, legend_y - 7), label, fill=color)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
