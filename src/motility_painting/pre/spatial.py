"""Per-frame spatial context: local crowding and collective motion.

Everything here operates on **all** detected cells in a frame (straight from
the per-frame GeoDataFrames), not just cells that were successfully tracked or
that reach the final frame -- crowding is a property of a cell's neighbours at
that instant, independent of whether it (or they) were later linked into a
trajectory.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from .centroids import dedupe_split_cells, parse_frame_index


def frame_local_density(
    gdf: gpd.GeoDataFrame,
    *,
    radius_um: float = 150.0,
    um_per_px: float,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Neighbour count within ``radius_um`` for every cell in one frame.

    Returns ``cell_id, label, frame_index, centroid_x, centroid_y,
    neighbor_count`` (self excluded). Coordinates are read in pixels and the
    radius query is done in the same units after scaling by ``um_per_px``, so
    the returned ``neighbor_count`` is comparable across frames/datasets with
    different pixel calibration.
    """
    sub = dedupe_split_cells(gdf) if dedupe else gdf
    coords_um = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=float) * um_per_px
    tree = cKDTree(coords_um)
    counts = tree.query_ball_point(coords_um, r=radius_um, return_length=True) - 1

    return pd.DataFrame(
        {
            "cell_id": sub["cell_id"].to_numpy(),
            "label": sub["label"].to_numpy(),
            "frame_index": sub["frame_index"].to_numpy(),
            "centroid_x": sub["centroid_x"].to_numpy(),
            "centroid_y": sub["centroid_y"].to_numpy(),
            "neighbor_count": counts.astype(int),
        }
    )


def all_frames_local_density(
    gdf_dir: str | Path,
    *,
    radius_um: float = 150.0,
    um_per_px: float,
    pattern: str = "*.parquet",
) -> pd.DataFrame:
    """Concatenate :func:`frame_local_density` over every per-frame GeoDataFrame in ``gdf_dir``."""
    files = sorted(glob(str(Path(gdf_dir) / pattern)), key=lambda f: parse_frame_index(Path(f).name))
    parts = []
    for path in files:
        gdf = gpd.read_parquet(path)
        if "frame_index" not in gdf.columns:
            gdf["frame_index"] = parse_frame_index(Path(path).name)
        parts.append(frame_local_density(gdf, radius_um=radius_um, um_per_px=um_per_px))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def frame_collisions(gdf: gpd.GeoDataFrame, *, dedupe: bool = True) -> pd.DataFrame:
    """Pairs of distinct cells whose segmentation polygons touch/overlap in one frame.

    Cellpose masks partition the pixel grid, so adjacent cells' polygons share
    a boundary (``touches``, zero-area intersection) rather than truly
    overlapping -- this is the literal "bumper cars" contact event, found via
    ``intersects`` (which covers both touching and any incidental overlap)
    rather than requiring positive intersection area.

    Returns one row per unordered colliding pair: ``frame_index, cell_id_a,
    cell_id_b, label_a, label_b``.
    """
    sub = dedupe_split_cells(gdf) if dedupe else gdf
    sub = sub.reset_index(drop=True)
    sindex = sub.sindex

    rows = []
    geoms = sub.geometry.values
    cell_ids = sub["cell_id"].to_numpy()
    labels = sub["label"].to_numpy()
    frame_index = sub["frame_index"].iloc[0] if len(sub) else None
    for i, geom in enumerate(geoms):
        for j in sindex.query(geom, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            rows.append(
                {
                    "frame_index": frame_index,
                    "cell_id_a": cell_ids[i],
                    "cell_id_b": cell_ids[j],
                    "label_a": labels[i],
                    "label_b": labels[j],
                }
            )
    return pd.DataFrame(rows, columns=["frame_index", "cell_id_a", "cell_id_b", "label_a", "label_b"])


def all_frames_collisions(gdf_dir: str | Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatenate :func:`frame_collisions` over every per-frame GeoDataFrame in ``gdf_dir``."""
    files = sorted(glob(str(Path(gdf_dir) / pattern)), key=lambda f: parse_frame_index(Path(f).name))
    parts = []
    for path in files:
        gdf = gpd.read_parquet(path)
        if "frame_index" not in gdf.columns:
            gdf["frame_index"] = parse_frame_index(Path(path).name)
        parts.append(frame_collisions(gdf))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["frame_index", "cell_id_a", "cell_id_b", "label_a", "label_b"]
    )


def trajectory_step_speeds(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
    um_per_px: float | None = None,
) -> pd.DataFrame:
    """Per-step instantaneous speed for every tracked cell (all trajectories, every frame-to-frame step).

    Unlike :func:`motility_painting.pre.linking.trajectory_motility_metrics`
    (one aggregate row per trajectory), this keeps one row per step so it can
    be joined against per-frame local density for a within-trajectory,
    all-cells speed-vs-crowding analysis.

    Returns ``trajectory_id, frame_index`` (the *later* frame of the step),
    ``cell_id, label, step_speed`` (``_um`` variant if ``um_per_px`` given).
    """
    rows = []
    for tid, g in traj.sort_values(frame_col).groupby(traj_col):
        g = g.reset_index(drop=True)
        xs = g[x_col].to_numpy(dtype=float)
        ys = g[y_col].to_numpy(dtype=float)
        speed = np.hypot(np.diff(xs), np.diff(ys))
        for i in range(1, len(g)):
            rows.append(
                {
                    traj_col: tid,
                    frame_col: int(g[frame_col].iloc[i]),
                    "cell_id": g["cell_id"].iloc[i] if "cell_id" in g.columns else None,
                    "label": g["label"].iloc[i] if "label" in g.columns else None,
                    "step_speed": float(speed[i - 1]),
                }
            )
    out = pd.DataFrame(rows)
    if um_per_px is not None and len(out):
        out["step_speed_um"] = out["step_speed"] * um_per_px
    return out


def velocity_correlation_vs_distance(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
    um_per_px: float,
    distance_bins_um: tuple[float, ...] = (0, 50, 100, 150, 200, 300, 400, 600),
) -> pd.DataFrame:
    """Mean heading-cosine-similarity between all pairs of cells in the same frame, by distance bin.

    For each frame, computes every tracked cell's instantaneous velocity
    vector (previous->current step), then for every pair of cells present in
    that frame, bins their separation and the cosine similarity of their
    velocity directions. Averaged across all frames, this is the classic
    "velocity correlation function" used to detect collective/coordinated
    migration (values near 0 = independent movement; consistently positive at
    short range = local coordination/flocking).

    Returns one row per distance bin: ``bin_left_um, bin_right_um,
    mean_cos_similarity, n_pairs``.
    """
    n_bins = len(distance_bins_um) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    # Build per-(trajectory, frame) velocity vectors once, then do the pairwise binning per frame.
    vel_rows = []
    for tid, tg in traj.sort_values(frame_col).groupby(traj_col):
        tg = tg.reset_index(drop=True)
        xs = tg[x_col].to_numpy(dtype=float)
        ys = tg[y_col].to_numpy(dtype=float)
        for i in range(1, len(tg)):
            vel_rows.append(
                {
                    traj_col: tid,
                    frame_col: int(tg[frame_col].iloc[i]),
                    "vx": xs[i] - xs[i - 1],
                    "vy": ys[i] - ys[i - 1],
                    "x": xs[i],
                    "y": ys[i],
                }
            )
    vel = pd.DataFrame(vel_rows)

    for _, g in vel.groupby(frame_col):
        if len(g) < 2:
            continue
        xy = g[["x", "y"]].to_numpy(dtype=float) * um_per_px
        v = g[["vx", "vy"]].to_numpy(dtype=float)
        speed = np.hypot(v[:, 0], v[:, 1])
        unit = np.divide(v, speed[:, None], out=np.zeros_like(v), where=speed[:, None] > 0)

        tree = cKDTree(xy)
        pairs = tree.query_pairs(r=distance_bins_um[-1], output_type="ndarray")
        if len(pairs) == 0:
            continue
        d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
        cos_sim = np.sum(unit[pairs[:, 0]] * unit[pairs[:, 1]], axis=1)
        valid = (speed[pairs[:, 0]] > 0) & (speed[pairs[:, 1]] > 0)

        bin_idx = np.digitize(d[valid], distance_bins_um[1:-1])
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                sums[b] += cos_sim[valid][mask].sum()
                counts[b] += int(mask.sum())

    mean_cos = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
    return pd.DataFrame(
        {
            "bin_left_um": distance_bins_um[:-1],
            "bin_right_um": distance_bins_um[1:],
            "mean_cos_similarity": mean_cos,
            "n_pairs": counts,
        }
    )
