"""Per-frame LCI cell morphology, derived from the segmentation polygons.

Only centroid x/y is persisted anywhere upstream (:mod:`.centroids`,
:mod:`.trajectories`) -- shape is derived here directly from each cell's
polygon in the per-frame GeoDataFrames (``1.2GeoDataFrames/Time*.parquet``),
giving morphology *over time* once joined onto a trajectory table. Features
are shapely-derived (area/perimeter/solidity/elongation from the polygon and
its convex hull / minimum rotated rectangle) rather than image-moment based
(``regionprops`` on the raw masks would additionally give eccentricity and
orientation, but needs re-opening the raw mask arrays -- out of scope here).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .centroids import dedupe_split_cells, parse_frame_index

SHAPE_COLS = [
    "area",
    "perimeter",
    "solidity",
    "elongation",
    "equivalent_diameter",
    "form_factor",
]


def frame_shape_features(gdf: gpd.GeoDataFrame, *, dedupe: bool = True) -> pd.DataFrame:
    """Shapely-derived shape features for every cell in one frame's GeoDataFrame.

    Adds, per polygon:
    ``area``, ``perimeter`` (``geometry.length``), ``solidity``
    (``area / convex_hull.area``), ``elongation`` (long/short side ratio of the
    minimum rotated rectangle), ``equivalent_diameter`` (diameter of a circle
    with the same area), ``form_factor`` (``4*pi*area / perimeter**2``, 1.0 for
    a perfect circle, lower for elongated/irregular shapes).

    Returns ``cell_id, label, frame_index, centroid_x, centroid_y`` plus
    :data:`SHAPE_COLS`.
    """
    sub = dedupe_split_cells(gdf) if dedupe else gdf

    area = sub.geometry.area.to_numpy()
    perimeter = sub.geometry.length.to_numpy()
    hull_area = sub.geometry.convex_hull.area.to_numpy()
    solidity = np.divide(area, hull_area, out=np.full_like(area, np.nan), where=hull_area > 0)

    rects = sub.geometry.minimum_rotated_rectangle()
    elongation = np.empty(len(sub))
    for i, rect in enumerate(rects):
        coords = np.asarray(rect.exterior.coords)
        sides = np.hypot(*np.diff(coords, axis=0).T)
        long, short = sides.max(), sides.min()
        elongation[i] = (long / short) if short > 0 else np.nan

    equivalent_diameter = np.sqrt(4.0 * area / np.pi)
    form_factor = np.divide(
        4.0 * np.pi * area, perimeter**2, out=np.full_like(area, np.nan), where=perimeter > 0
    )

    out = pd.DataFrame(
        {
            "cell_id": sub["cell_id"].to_numpy(),
            "label": sub["label"].to_numpy(),
            "frame_index": sub["frame_index"].to_numpy(),
            "centroid_x": sub["centroid_x"].to_numpy(),
            "centroid_y": sub["centroid_y"].to_numpy(),
            "area": area,
            "perimeter": perimeter,
            "solidity": solidity,
            "elongation": elongation,
            "equivalent_diameter": equivalent_diameter,
            "form_factor": form_factor,
        }
    )
    return out.reset_index(drop=True)


def all_frames_shape_features(gdf_dir: str | Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatenate :func:`frame_shape_features` over every per-frame GeoDataFrame in ``gdf_dir``."""
    files = sorted(glob(str(Path(gdf_dir) / pattern)), key=lambda f: parse_frame_index(Path(f).name))
    parts = []
    for path in files:
        gdf = gpd.read_parquet(path)
        if "frame_index" not in gdf.columns:
            gdf["frame_index"] = parse_frame_index(Path(path).name)
        parts.append(frame_shape_features(gdf))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SHAPE_COLS)


def join_shape_to_trajectories(
    shape_df: pd.DataFrame,
    traj: pd.DataFrame,
    *,
    on: tuple[str, str] = ("cell_id", "frame_index"),
) -> pd.DataFrame:
    """Attach shape-over-time to a trajectory table.

    ``traj`` is a long (one row per trajectory per frame) table like
    :func:`motility_painting.pre.trajectories.build_trajectories`'s output --
    joins on ``(cell_id, frame_index)`` by default, which both tables share.
    Rows in ``traj`` without a shape match (dropped during dedupe, or a frame
    not covered by ``shape_df``) are kept with NaN shape columns.
    """
    cols = list(on) + SHAPE_COLS
    return traj.merge(shape_df[cols], on=list(on), how="left")


def shape_variability(
    shape_traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
) -> pd.DataFrame:
    """Per-trajectory shape summary: mean and coefficient-of-variation of each shape feature.

    Input is the output of :func:`join_shape_to_trajectories`. Returns one row
    per trajectory with ``{feature}_mean`` and ``{feature}_cv`` for every
    column in :data:`SHAPE_COLS`, plus ``n_frames``.
    """

    def _cv(s: pd.Series) -> float:
        m = s.mean()
        return float(s.std() / m) if m else np.nan

    grouped = shape_traj.groupby(traj_col)
    out = grouped.size().rename("n_frames").to_frame()
    for col in SHAPE_COLS:
        out[f"{col}_mean"] = grouped[col].mean()
        out[f"{col}_cv"] = grouped[col].apply(_cv)
    return out.reset_index()
