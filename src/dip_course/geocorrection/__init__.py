"""Document-image geometric correction and evaluation."""

from dip_course.geocorrection.dataset import GeoCorrectionPair, discover_pairs
from dip_course.geocorrection.rectify import RectificationResult, rectify_document

__all__ = [
    "GeoCorrectionPair",
    "RectificationResult",
    "discover_pairs",
    "rectify_document",
]
