from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter1d
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import (
    ProjectiveTransform,
    hough_line,
    hough_line_peaks,
    resize,
    warp,
)


@dataclass(frozen=True)
class RectificationResult:
    input_image: np.ndarray
    input_mask: np.ndarray
    homography_image: np.ndarray
    homography_mask: np.ndarray
    corrected_image: np.ndarray
    corrected_mask: np.ndarray
    corners_xy: np.ndarray
    orientation_quarter_turns: int


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return array


def _as_binary_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected HxW document mask, got {array.shape}")
    binary = array > 127 if array.dtype != np.bool_ else array.copy()
    if int(binary.sum()) < 64:
        raise ValueError("Document mask contains too few foreground pixels")
    return binary


def _resize_pair(
    image: np.ndarray, mask: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image, mask
    shape = (
        max(1, round(image.shape[0] * scale)),
        max(1, round(image.shape[1] * scale)),
    )
    resized_image = resize(
        image,
        shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    resized_mask = resize(
        mask.astype(np.uint8),
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    return resized_image, resized_mask


def find_page_corners(mask: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL corners from a clean binary page mask."""
    binary = _as_binary_mask(mask)
    y, x = np.nonzero(binary)
    xy = np.column_stack((x, y)).astype(np.float64)
    coordinate_sum = xy[:, 0] + xy[:, 1]
    coordinate_difference = xy[:, 0] - xy[:, 1]
    corners = np.stack(
        (
            xy[np.argmin(coordinate_sum)],
            xy[np.argmax(coordinate_difference)],
            xy[np.argmax(coordinate_sum)],
            xy[np.argmin(coordinate_difference)],
        )
    )
    if len(np.unique(corners, axis=0)) != 4:
        raise ValueError("Could not find four distinct page corners")
    x_values, y_values = corners[:, 0], corners[:, 1]
    polygon_area = 0.5 * abs(
        np.dot(x_values, np.roll(y_values, 1))
        - np.dot(y_values, np.roll(x_values, 1))
    )
    if polygon_area < 64:
        raise ValueError("Document-mask quadrilateral is degenerate")
    return corners


def _destination_size(corners: np.ndarray) -> tuple[int, int]:
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    right = np.linalg.norm(corners[2] - corners[1])
    left = np.linalg.norm(corners[3] - corners[0])
    return max(8, round(max(top, bottom))), max(8, round(max(left, right)))


def _projective_rectification(
    image: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = _destination_size(corners)
    destination = np.array(
        ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
        dtype=np.float64,
    )
    transform = ProjectiveTransform.from_estimate(corners, destination)
    warped_image = warp(
        image,
        inverse_map=transform.inverse,
        output_shape=(height, width),
        order=3,
        mode="constant",
        cval=255,
        preserve_range=True,
    )
    warped_mask = warp(
        mask.astype(np.uint8),
        inverse_map=transform.inverse,
        output_shape=(height, width),
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    return np.clip(np.rint(warped_image), 0, 255).astype(np.uint8), warped_mask > 0.5


def _warp_rgb_with_coordinates(
    image: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    channels = [
        warp(
            image[..., channel],
            coordinates,
            order=3,
            mode="edge",
            preserve_range=True,
        )
        for channel in range(3)
    ]
    return np.clip(np.rint(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)


def _interpolate_limits(
    mask: np.ndarray, *, along_columns: bool
) -> tuple[np.ndarray, np.ndarray]:
    length = mask.shape[1] if along_columns else mask.shape[0]
    lower = np.full(length, np.nan, dtype=np.float64)
    upper = np.full(length, np.nan, dtype=np.float64)
    for index in range(length):
        occupied = np.flatnonzero(mask[:, index] if along_columns else mask[index, :])
        if occupied.size:
            lower[index] = occupied[0]
            upper[index] = occupied[-1]
    valid = np.flatnonzero(np.isfinite(lower))
    if valid.size < 4:
        raise ValueError("Too few valid mask slices for boundary dewarping")
    coordinates = np.arange(length)
    lower = np.interp(coordinates, valid, lower[valid])
    upper = np.interp(coordinates, valid, upper[valid])
    sigma = max(1.0, length / 250.0)
    lower = gaussian_filter1d(lower, sigma=sigma, mode="nearest")
    upper = gaussian_filter1d(upper, sigma=sigma, mode="nearest")
    upper = np.maximum(upper, lower + 1.0)
    return lower, upper


def _straighten_top_bottom(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    top, bottom = _interpolate_limits(mask, along_columns=True)
    normalized_y = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
    source_y = top[None, :] + normalized_y * (bottom - top)[None, :]
    source_x = np.broadcast_to(np.arange(width, dtype=np.float64), (height, width))
    coordinates = np.stack((source_y, source_x))
    corrected_image = _warp_rgb_with_coordinates(image, coordinates)
    corrected_mask = warp(
        mask.astype(np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    return (
        corrected_image,
        corrected_mask > 0.5,
    )


def _straighten_left_right(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    left, right = _interpolate_limits(mask, along_columns=False)
    normalized_x = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
    source_x = left[:, None] + normalized_x * (right - left)[:, None]
    source_y = np.broadcast_to(
        np.arange(height, dtype=np.float64)[:, None], (height, width)
    )
    coordinates = np.stack((source_y, source_x))
    corrected_image = _warp_rgb_with_coordinates(image, coordinates)
    corrected_mask = warp(
        mask.astype(np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    corrected_mask = corrected_mask > 0.5
    corrected_image[~corrected_mask] = 255
    return corrected_image, corrected_mask


def _needs_quarter_turn(image: np.ndarray, mask: np.ndarray) -> bool:
    scale = min(1.0, 600 / max(image.shape[:2]))
    if scale < 1.0:
        shape = (
            max(1, round(image.shape[0] * scale)),
            max(1, round(image.shape[1] * scale)),
        )
        image = resize(
            image,
            shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        mask = resize(
            mask.astype(np.uint8),
            shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    interior = binary_erosion(mask, iterations=max(2, round(min(mask.shape) / 100)))
    edges = canny(rgb2gray(image), sigma=1.2) & interior
    if int(edges.sum()) < 32:
        return False
    hough_space, angles, distances = hough_line(edges)
    accumulators, peak_angles, _ = hough_line_peaks(
        hough_space,
        angles,
        distances,
        threshold=0.2 * float(hough_space.max()),
        num_peaks=50,
    )
    if accumulators.size == 0:
        return False
    line_angles = (np.rad2deg(peak_angles) + 90.0) % 180.0
    horizontal_distance = np.minimum(line_angles, 180.0 - line_angles)
    vertical_distance = np.abs(line_angles - 90.0)
    weights = accumulators.astype(np.float64)
    horizontal_score = float(np.sum(weights * np.exp(-(horizontal_distance / 12.0) ** 2)))
    vertical_score = float(np.sum(weights * np.exp(-(vertical_distance / 12.0) ** 2)))
    return vertical_score > horizontal_score * 1.25


def needs_quarter_turn(image: np.ndarray, mask: np.ndarray) -> bool:
    """Return whether document text is predominantly vertical in the image frame."""
    return _needs_quarter_turn(image, mask)


def rectify_document(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    *,
    max_side: int = 1400,
    boundary_dewarp: bool = True,
    auto_orient: bool = True,
) -> RectificationResult:
    rgb = _as_rgb_array(image)
    binary = _as_binary_mask(mask)
    if rgb.shape[:2] != binary.shape:
        raise ValueError(
            f"Image/mask dimensions differ: {rgb.shape[:2]} vs {binary.shape}"
        )
    rgb, binary = _resize_pair(rgb, binary, max_side)
    corners = find_page_corners(binary)
    homography_image, homography_mask = _projective_rectification(
        rgb, binary, corners
    )
    if boundary_dewarp:
        vertical_image, vertical_mask = _straighten_top_bottom(
            homography_image, homography_mask
        )
        corrected_image, corrected_mask = _straighten_left_right(
            vertical_image, vertical_mask
        )
    else:
        corrected_image, corrected_mask = homography_image, homography_mask
    orientation_quarter_turns = 0
    if auto_orient and _needs_quarter_turn(corrected_image, corrected_mask):
        corrected_image = np.rot90(corrected_image, k=1).copy()
        corrected_mask = np.rot90(corrected_mask, k=1).copy()
        orientation_quarter_turns = 1
    return RectificationResult(
        input_image=rgb,
        input_mask=binary,
        homography_image=homography_image,
        homography_mask=homography_mask,
        corrected_image=corrected_image,
        corrected_mask=corrected_mask,
        corners_xy=corners,
        orientation_quarter_turns=orientation_quarter_turns,
    )
