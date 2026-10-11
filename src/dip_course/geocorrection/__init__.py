"""Document-image geometric correction and evaluation."""

from dip_course.geocorrection.dataset import GeoCorrectionPair, discover_pairs
from dip_course.geocorrection.rectify import (
    RectificationResult,
    needs_quarter_turn,
    rectify_document,
)
from dip_course.geocorrection.uvdoc import (
    MaskCrop,
    crop_to_foreground,
    crop_to_page_mask,
    uvdoc_result_to_rgb,
)

__all__ = [
    "GeoCorrectionPair",
    "MaskCrop",
    "RectificationResult",
    "crop_to_foreground",
    "crop_to_page_mask",
    "discover_pairs",
    "needs_quarter_turn",
    "rectify_document",
    "uvdoc_result_to_rgb",
]
