"""Data-prep utilities for the motility_painting workflow.

Modules
-------
centroids : final-frame LCI centroid extraction from the MASTER parquet.
trajectories : polygon-intersection cell tracking across LCI frames.
stitching : Cell Painting XML + CellProfiler merge into global mosaic coords.
morphology : per-frame shape features derived from segmentation polygons.
spatial : per-frame local density/crowding and collective-motion metrics.
nullmodel : direction-randomized random-walk null model for trajectories.
"""

from . import centroids, images, linking, morphology, nullmodel, spatial, stitching, trajectories
from .centroids import final_frame_centroids, frame_centroids, to_widget_points
from .linking import (
    apply_fov_transforms,
    cp_fov_bitmaps_in_lci,
    link_cp_to_trajectories,
    trajectory_motility_metrics,
    transform_orientation_deg,
)
from .morphology import (
    all_frames_shape_features,
    frame_shape_features,
    join_shape_to_trajectories,
    shape_variability,
)
from .nullmodel import (
    permutation_displacement_test,
    population_displacement_test,
    randomize_step_directions,
    stepwise_turning_angles,
    turning_angle_autocorrelation,
)
from .spatial import (
    all_frames_collisions,
    all_frames_local_density,
    frame_collisions,
    frame_local_density,
    trajectory_step_speeds,
    velocity_correlation_vs_distance,
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
    "morphology",
    "nullmodel",
    "spatial",
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
    "all_frames_shape_features",
    "frame_shape_features",
    "join_shape_to_trajectories",
    "shape_variability",
    "permutation_displacement_test",
    "population_displacement_test",
    "randomize_step_directions",
    "stepwise_turning_angles",
    "turning_angle_autocorrelation",
    "all_frames_collisions",
    "all_frames_local_density",
    "frame_collisions",
    "frame_local_density",
    "trajectory_step_speeds",
    "velocity_correlation_vs_distance",
    "add_global_coords",
    "combine_sc_with_positions",
    "cp_global_centroids",
    "parse_stage_positions",
    "build_trajectories",
    "track_from_dir",
    "trajectories_reaching",
]
