"""Link Cell Painting cells to LCI motility trajectories.

Notebook 03 fits, per Cell Painting FOV, a similarity transform (scale +
rotation/reflection + translation) mapping that FOV's local pixel coordinates
into the LCI **final-frame (Time64)** pixel space. This module applies those
transforms to every CellProfiler cell, links each to the nearest motility
trajectory endpoint, and summarises per-trajectory motility -- producing one
table where a cell carries both its Cell Painting id and its motility
``trajectory_id`` under a shared ``link_id``.

Why Time64 (and not the last *live* frame): cells are fixed between Time59 and
Time60, so the fixed Cell Painting positions correspond to the post-fixation
Time64 layout. Motility *metrics*, by contrast, are computed only over the live
frames (<= ``MOTILITY_LAST_FRAME``), excluding the fixation jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathlib import Path

from ..transforms import apply_similarity_transform

CP_X = "Location_Center_X"
CP_Y = "Location_Center_Y"


def cp_fov_bitmaps_in_lci(
    ims_dir,
    frame_ids: list[str],
    transforms: dict,
    *,
    channels: tuple[int, ...] = (0, 1, 2),
    multicolor: bool = True,
    channel_colors: dict | None = None,
    trim: float = 0.063,
    size: int = 1024,
    quality: int = 80,
    name_template: str = "051026_1_{fid}.ims",
) -> list[dict]:
    """Render each CP FOV to a WebP and place it in LCI space for a BitmapLayer.

    For every ``frame_id`` with a transform, reads the ``.ims`` FOV, composites
    it, and transforms the four (optionally trimmed) native-pixel image corners
    into LCI coordinates, so deck.gl can draw the tile with the same
    rotation/reflection/scale the alignment used.

    Parameters
    ----------
    multicolor:
        If True (default), false-colour composite of the 5 Cell Painting stains
        (:func:`motility_painting.pre.images.ims_multicolor_array`,
        ``channel_colors`` overrides the palette). If False, an RGB composite of
        ``channels``.
    trim:
        Fraction of each image edge to crop before placing, to remove the
        inter-tile overlap so the stitched mosaic abuts cleanly (~0.063 ≈ the
        measured 12.7% FOV-to-FOV overlap). ``0`` keeps full tiles.

    Returns a list of ``{"fov", "url", "bounds"}`` where ``bounds`` is the deck.gl
    4-corner form ``[[left,bottom],[left,top],[right,top],[right,bottom]]`` in LCI
    pixels (image row 0 maps to the "top" corners). FOVs without a transform are
    skipped.
    """
    from .images import ims_composite_array, ims_multicolor_array, _array_to_webp_url

    out: list[dict] = []
    for fid in frame_ids:
        if fid not in transforms:
            continue
        path = Path(ims_dir) / name_template.format(fid=fid)
        if multicolor:
            arr, (w, h) = ims_multicolor_array(path, channel_colors=channel_colors)
        else:
            arr, (w, h) = ims_composite_array(path, channels=channels)

        # crop the overlap border, tracking the crop box in native pixels
        mx = int(round(w * trim))
        my = int(round(h * trim))
        if mx or my:
            arr = arr[my : h - my, mx : w - mx]
        x0, y0, x1, y1 = mx, my, w - mx, h - my

        url = _array_to_webp_url(arr, size=size, quality=quality)
        t = transforms[fid]
        tf = {
            "scale": t["scale"],
            "rotation": np.asarray(t["rotation"], dtype=float),
            "translation": np.asarray(t["translation"], dtype=float),
        }
        # cropped-image corners in FOV-local px: bottom-left, top-left, top-right, bottom-right
        corners = np.array([[x0, y1], [x0, y0], [x1, y0], [x1, y1]], dtype=float)
        lci = apply_similarity_transform(corners, tf)
        out.append({"fov": fid, "url": url, "bounds": [[float(x), float(y)] for x, y in lci]})
    return out


def apply_fov_transforms(
    cp: pd.DataFrame,
    transforms: dict,
    *,
    x_col: str = CP_X,
    y_col: str = CP_Y,
    frame_col: str = "FrameID",
) -> tuple[pd.DataFrame, list[str]]:
    """Apply each FOV's CP->LCI similarity transform, adding ``x_lci``/``y_lci``.

    ``cp[frame_col]`` values must match the ``transforms`` keys (e.g. ``"F173"``).
    Uses the package convention ``transformed = scale * (xy @ R) + t`` -- the
    same one :func:`motility_painting.transforms.fit_similarity_transform`
    produced the saved transforms with, so it round-trips exactly (including the
    reflection some FOVs use, where ``det(R) < 0``).

    Returns ``(aligned_df, skipped_fovs)``.
    """
    parts: list[pd.DataFrame] = []
    skipped: list[str] = []
    for fov, sub in cp.groupby(frame_col):
        if fov not in transforms:
            skipped.append(fov)
            continue
        t = transforms[fov]
        xy = sub[[x_col, y_col]].to_numpy(dtype=float)
        xyt = apply_similarity_transform(
            xy,
            {
                "scale": t["scale"],
                "rotation": np.asarray(t["rotation"], dtype=float),
                "translation": np.asarray(t["translation"], dtype=float),
            },
        )
        s = sub.copy()
        s["x_lci"] = xyt[:, 0]
        s["y_lci"] = xyt[:, 1]
        parts.append(s)
    aligned = pd.concat(parts, ignore_index=True) if parts else cp.iloc[:0].copy()
    return aligned, skipped


def link_cp_to_trajectories(
    cp_aligned: pd.DataFrame,
    final_positions: pd.DataFrame,
    *,
    threshold: float = 15.0,
    traj_col: str = "trajectory_id",
    x_col: str = "x_lci",
    y_col: str = "y_lci",
    tx_col: str = "centroid_x",
    ty_col: str = "centroid_y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbour link aligned CP cells to trajectory endpoints.

    ``final_positions`` is one row per trajectory at the matching (final) frame,
    with columns ``(traj_col, tx_col, ty_col)``. Every CP cell gets its nearest
    trajectory and ``dist_to_lci``; matches beyond ``threshold`` px are dropped,
    and where several CP cells claim one trajectory only the closest is kept
    (an LCI cell links to at most one CP cell and vice versa).

    Returns ``(cp_scored, linked)`` -- ``cp_scored`` is every CP cell with its
    distance/trajectory (for QC histograms), ``linked`` is the deduped matches.
    """
    tgt = final_positions.dropna(subset=[tx_col, ty_col]).reset_index(drop=True)
    tree = cKDTree(tgt[[tx_col, ty_col]].to_numpy(dtype=float))
    dist, idx = tree.query(cp_aligned[[x_col, y_col]].to_numpy(dtype=float), k=1)

    scored = cp_aligned.copy()
    scored["dist_to_lci"] = dist
    scored[traj_col] = tgt[traj_col].to_numpy()[idx]

    linked = scored[scored["dist_to_lci"] < threshold].copy()
    linked = linked.sort_values("dist_to_lci").drop_duplicates(traj_col, keep="first")
    return scored, linked


def trajectory_motility_metrics(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
    um_per_px: float | None = None,
) -> pd.DataFrame:
    """Per-trajectory motility summary from a long (one row per frame) table.

    ``traj`` should already be restricted to the live frames used for motility
    (e.g. the ``*_to_frame59`` table). For each trajectory, positions are ordered
    by frame and reduced to:

    ``n_frames, frame_first, frame_last, start_x/y, end_x/y,
    net_dx, net_dy, net_displacement, path_length, straightness (net/path),
    mean_step, heading_deg`` (``atan2(net_dy, net_dx)`` in image pixel space,
    y-down), plus ``*_um`` variants of the distance columns when ``um_per_px``
    is given.
    """
    out = []
    for tid, g in traj.sort_values(frame_col).groupby(traj_col):
        xs = g[x_col].to_numpy(dtype=float)
        ys = g[y_col].to_numpy(dtype=float)
        steps = np.hypot(np.diff(xs), np.diff(ys))
        net_dx = xs[-1] - xs[0]
        net_dy = ys[-1] - ys[0]
        net = float(np.hypot(net_dx, net_dy))
        path = float(steps.sum())
        out.append(
            {
                traj_col: tid,
                "n_frames": int(len(g)),
                "frame_first": int(g[frame_col].iloc[0]),
                "frame_last": int(g[frame_col].iloc[-1]),
                "start_x": float(xs[0]),
                "start_y": float(ys[0]),
                "end_x": float(xs[-1]),
                "end_y": float(ys[-1]),
                "net_dx": float(net_dx),
                "net_dy": float(net_dy),
                "net_displacement": net,
                "path_length": path,
                "straightness": (net / path) if path > 0 else np.nan,
                "mean_step": float(steps.mean()) if len(steps) else np.nan,
                "heading_deg": float(np.degrees(np.arctan2(net_dy, net_dx))),
            }
        )
    df = pd.DataFrame(out)
    if um_per_px is not None and len(df):
        for c in ("net_displacement", "path_length", "mean_step"):
            df[f"{c}_um"] = df[c] * um_per_px
    return df


def transform_orientation_deg(orientation_deg, rotation) -> np.ndarray:
    """Map a CellProfiler ``AreaShape_Orientation`` axis angle into LCI space.

    Orientation is the angle of a cell's major *axis*, so it is defined modulo
    180 deg and has no arrowhead. We rotate the unit direction ``(cos, sin)`` by
    the transform's linear part (same ``v @ R`` convention as the coordinate
    transform, so a reflected FOV -- ``det(R) < 0`` -- mirrors the angle
    correctly), then fold the result back to ``[-90, 90)``. Scale and
    translation don't affect an angle, so only ``rotation`` is needed.
    """
    R = np.asarray(rotation, dtype=float)
    th = np.radians(np.asarray(orientation_deg, dtype=float))
    v = np.stack([np.cos(th), np.sin(th)], axis=-1)
    vt = v @ R
    ang = np.degrees(np.arctan2(vt[..., 1], vt[..., 0]))
    return (ang + 90.0) % 180.0 - 90.0
