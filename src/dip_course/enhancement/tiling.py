from __future__ import annotations

import torch
from torch.nn import functional as F


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


@torch.inference_mode()
def predict_tiled(
    model: torch.nn.Module,
    packed_raw: torch.Tensor,
    *,
    tile_size: int = 256,
    overlap: int = 32,
) -> torch.Tensor:
    if packed_raw.ndim != 4 or packed_raw.shape[0] != 1:
        raise ValueError("Tiled inference expects one NCHW image")
    if tile_size % getattr(model, "input_multiple", 1):
        raise ValueError("tile_size is incompatible with the model downsampling factor")

    original_height, original_width = packed_raw.shape[-2:]
    pad_height = max(0, tile_size - original_height)
    pad_width = max(0, tile_size - original_width)
    if pad_height or pad_width:
        mode = "reflect" if original_height > 1 and original_width > 1 else "replicate"
        packed_raw = F.pad(packed_raw, (0, pad_width, 0, pad_height), mode=mode)

    height, width = packed_raw.shape[-2:]
    scene_context = None
    context_function = getattr(model, "compute_scene_context", None)
    if callable(context_function):
        scene_context = context_function(packed_raw)
    output = torch.zeros(
        (1, 3, height * 2, width * 2), device=packed_raw.device, dtype=torch.float32
    )
    weights = torch.zeros_like(output[:, :1])
    feather_1d = torch.hann_window(
        tile_size * 2,
        periodic=False,
        device=packed_raw.device,
        dtype=torch.float32,
    ).clamp_min(1e-3)
    feather = torch.outer(feather_1d, feather_1d)[None, None]

    for top in tile_starts(height, tile_size, overlap):
        for left in tile_starts(width, tile_size, overlap):
            tile = packed_raw[:, :, top : top + tile_size, left : left + tile_size]
            if scene_context is None:
                prediction = model(tile).float()
            else:
                prediction = model(tile, scene_context=scene_context).float()
            out_top, out_left = top * 2, left * 2
            out_bottom, out_right = out_top + tile_size * 2, out_left + tile_size * 2
            output[:, :, out_top:out_bottom, out_left:out_right] += prediction * feather
            weights[:, :, out_top:out_bottom, out_left:out_right] += feather

    output = output / weights.clamp_min(1e-8)
    return output[:, :, : original_height * 2, : original_width * 2]
