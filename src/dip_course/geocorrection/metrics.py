from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, resize

from dip_course.geocorrection.rectify import find_page_corners


def _point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length == 0.0:
        return np.linalg.norm(points - start, axis=1)
    projection = np.clip(((points - start) @ segment) / squared_length, 0.0, 1.0)
    nearest = start + projection[:, None] * segment
    return np.linalg.norm(points - nearest, axis=1)


def boundary_linearity(mask: np.ndarray) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {binary.shape}")
    boundary = binary ^ binary_erosion(binary)
    y, x = np.nonzero(boundary)
    if x.size < 16:
        raise ValueError("Mask boundary contains too few pixels")
    points = np.column_stack((x, y)).astype(np.float64)
    if points.shape[0] > 20_000:
        stride = math.ceil(points.shape[0] / 20_000)
        points = points[::stride]
    corners = find_page_corners(binary)
    distances = np.stack(
        [
            _point_segment_distance(points, corners[index], corners[(index + 1) % 4])
            for index in range(4)
        ],
        axis=1,
    ).min(axis=1)
    diagonal = math.hypot(binary.shape[0], binary.shape[1])
    return {
        "boundary_rmse_normalized": float(
            np.sqrt(np.mean(distances**2, dtype=np.float64)) / diagonal
        ),
        "boundary_p95_normalized": float(np.percentile(distances, 95.0) / diagonal),
        "page_fill_ratio": float(binary.mean(dtype=np.float64)),
    }


def text_line_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int = 900,
) -> dict[str, float | int | None]:
    rgb = np.asarray(image, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != binary.shape:
        raise ValueError("Text-line metric requires aligned HxWx3 image and HxW mask")
    scale = min(1.0, max_side / max(rgb.shape[:2]))
    if scale < 1.0:
        shape = (
            max(1, round(rgb.shape[0] * scale)),
            max(1, round(rgb.shape[1] * scale)),
        )
        rgb = resize(
            rgb,
            shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        binary = resize(
            binary.astype(np.uint8),
            shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    eroded = binary_erosion(binary, iterations=max(1, round(min(binary.shape) / 150)))
    edges = canny(rgb2gray(rgb), sigma=1.2) & eroded
    edge_count = int(edges.sum())
    if edge_count < 32:
        return {
            "text_line_skew_degrees": None,
            "hough_line_strength": 0.0,
            "hough_peak_count": 0,
            "edge_pixel_count": edge_count,
        }
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 361, endpoint=False)
    hough_space, angles, distances = hough_line(edges, theta=tested_angles)
    accumulators, peak_angles, _ = hough_line_peaks(
        hough_space,
        angles,
        distances,
        min_distance=8,
        min_angle=4,
        threshold=0.2 * float(hough_space.max()),
        num_peaks=40,
    )
    if accumulators.size == 0:
        return {
            "text_line_skew_degrees": None,
            "hough_line_strength": 0.0,
            "hough_peak_count": 0,
            "edge_pixel_count": edge_count,
        }
    line_angles = (np.rad2deg(peak_angles) + 90.0) % 180.0
    horizontal_deviation = np.minimum(line_angles, 180.0 - line_angles)
    keep = horizontal_deviation <= 45.0
    if not np.any(keep):
        return {
            "text_line_skew_degrees": None,
            "hough_line_strength": 0.0,
            "hough_peak_count": 0,
            "edge_pixel_count": edge_count,
        }
    weights = accumulators[keep].astype(np.float64)
    deviation = horizontal_deviation[keep]
    return {
        "text_line_skew_degrees": float(np.average(deviation, weights=weights)),
        "hough_line_strength": float(weights.sum() / edge_count),
        "hough_peak_count": int(keep.sum()),
        "edge_pixel_count": edge_count,
    }
