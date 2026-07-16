"""Data-prep utilities for the motility_painting workflow.

Modules
-------
centroids : final-frame LCI centroid extraction from the MASTER parquet.
trajectories : polygon-intersection cell tracking across LCI frames.
stitching : Cell Painting XML + CellProfiler merge into global mosaic coords.
"""

from . import centroids, images, linking, stitching, trajectories
from .centroids import final_frame_centroids, frame_centroids, to_widget_points
from .linking import (
    apply_fov_transforms,
    cp_fov_bitmaps_in_lci,
    link_cp_to_trajectories,
    trajectory_motility_metrics,
    transform_orientation_deg,
)
from .stitching import (
    add_global_coords,
    cp_global_centroids,
    combine_sc_with_positions,
    parse_stage_positions,
)
from .trajectories import build_trajectories, track_from_dir, trajectories_reaching

__all__ = [
    "centroids",
    "images",
    "linking",
    "stitching",
    "trajectories",
    "apply_fov_transforms",
    "cp_fov_bitmaps_in_lci",
    "link_cp_to_trajectories",
    "trajectory_motility_metrics",
    "transform_orientation_deg",
    "final_frame_centroids",
    "frame_centroids",
    "to_widget_points",
    "add_global_coords",
    "combine_sc_with_positions",
    "cp_global_centroids",
    "parse_stage_positions",
    "build_trajectories",
    "track_from_dir",
    "trajectories_reaching",
]
