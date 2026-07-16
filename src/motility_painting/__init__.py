"""motility_painting: deck.gl + anywidget tools for aligning live-cell motility
tracking to Cell Painting morphology, plus the data-prep pipeline behind it.

Subpackages
-----------
viz : interactive deck.gl/anywidget widgets (aligner, trajectory scrubber).
pre : data-prep utilities (centroids, trajectories, CP stitching).
"""

from .io import image_file_to_data_url
from .transforms import (
    apply_similarity_transform,
    fit_similarity_transform,
    invert_similarity_transform,
    serialize_transform,
)
from .viz import MotilityPaintingAligner, TrajectoryScrubber

__all__ = [
    "MotilityPaintingAligner",
    "TrajectoryScrubber",
    "apply_similarity_transform",
    "fit_similarity_transform",
    "image_file_to_data_url",
    "invert_similarity_transform",
    "serialize_transform",
]
