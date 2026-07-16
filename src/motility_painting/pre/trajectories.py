"""Cell tracking across LCI frames by overlap, with distance + gap fallbacks.

For each consecutive frame pair we link cells **one-to-one**: first by largest
polygon overlap (each target cell claimed by at most one source), then a
shift-compensated nearest-centroid fallback for cells that moved without
overlapping their previous selves. A final gap-closing pass stitches a track that
ends at frame *i* to one that starts at frame *i+2* (a single-frame segmentation
dropout) when their endpoints are close.

This replaces the draft ``generating_cell_trajectories`` notebook. The earlier
rebuild let several source cells claim the same target and then silently
overwrote the trajectory tail, which shattered tracks into ~5x too many
fragments; the one-to-one matching here removes that, and the distance/gap
fallbacks recover moving cells and dropouts (including the ~8px fixation shift at
Time59->Time60).

Defaults are tuned to this dataset (median cell spacing ~18.6px, median
frame-to-frame motion ~1.3px); override per call if your data differs.
"""

from __future__ import annotations

from collections import defaultdict
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .centroids import parse_frame_index

# Tracking defaults (pixels). Chosen < the ~18.6px median neighbour spacing so a
# fallback link can't jump to a different cell, while still covering the ~10px
# 90th-percentile frame-to-frame motion and the ~8px fixation shift.
MAX_CENTROID_DIST = 12.0   # consecutive-frame nearest-centroid fallback radius
MAX_GAP = 1                # bridge this many missing frames (1 => connect i to i+2)
MAX_GAP_DIST = 18.0        # endpoint distance allowed when bridging a gap


def load_frame_gdfs(gdf_dir: str | Path, pattern: str = "*.parquet") -> list[gpd.GeoDataFrame]:
    """Load per-frame GeoDataFrames sorted by timepoint.

    Each returned GeoDataFrame gets a positional ``row`` column (0..n-1) and
    ``centroid_x/y`` (computed from geometry if absent) so links can reference
    cells by position and the distance fallback has coordinates.
    """
    files = sorted(glob(str(Path(gdf_dir) / pattern)), key=lambda f: parse_frame_index(Path(f).name))
    frames = []
    for path in files:
        gdf = gpd.read_parquet(path).reset_index(drop=True)
        gdf["frame_index"] = parse_frame_index(Path(path).name)
        gdf["row"] = range(len(gdf))
        if "centroid_x" not in gdf.columns or "centroid_y" not in gdf.columns:
            gdf["centroid_x"] = gdf.geometry.centroid.x
            gdf["centroid_y"] = gdf.geometry.centroid.y
        frames.append(gdf)
    return frames


def _centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf["centroid_x"].to_numpy(), gdf["centroid_y"].to_numpy()])


def match_consecutive(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    *,
    max_centroid_dist: float = MAX_CENTROID_DIST,
) -> pd.DataFrame:
    """One-to-one links from ``gdf_a`` to ``gdf_b``.

    Stage 1 (overlap): consider every intersecting pair and greedily accept them
    by descending intersection area, each source and target used at most once.
    Stage 2 (distance): for cells still unmatched on both sides, link by nearest
    centroid within ``max_centroid_dist``, after removing the global median shift
    estimated from the overlap matches (handles the fixation displacement).

    Returns columns ``row_a, row_b, intersection_area, distance, method``.
    """
    geoms_a = gdf_a.geometry.values
    geoms_b = gdf_b.geometry.values
    ca, cb = _centroids(gdf_a), _centroids(gdf_b)

    # --- stage 1: largest-overlap, injective ---
    sindex_b = gdf_b.sindex
    candidates = []  # (area, row_a, row_b)
    for row_a, geom_a in enumerate(geoms_a):
        for row_b in sindex_b.query(geom_a, predicate="intersects"):
            area = geom_a.intersection(geoms_b[int(row_b)]).area
            if area > 0:
                candidates.append((area, row_a, int(row_b)))
    candidates.sort(reverse=True)  # by area descending

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int, float, float, str]] = []
    for area, row_a, row_b in candidates:
        if row_a in used_a or row_b in used_b:
            continue
        used_a.add(row_a)
        used_b.add(row_b)
        matches.append((row_a, row_b, float(area), 0.0, "overlap"))

    # --- stage 2: shift-compensated nearest-centroid fallback ---
    if max_centroid_dist > 0:
        rem_a = [i for i in range(len(geoms_a)) if i not in used_a]
        rem_b = [j for j in range(len(geoms_b)) if j not in used_b]
        if rem_a and rem_b:
            shift = np.zeros(2)
            if matches:
                vecs = np.array([cb[rb] - ca[ra] for ra, rb, *_ in matches])
                shift = np.median(vecs, axis=0)
            tree = cKDTree(cb[rem_b])
            dist, idx = tree.query(ca[rem_a] + shift, k=1)
            claimed_b: set[int] = set()
            for k in np.argsort(dist):  # closest first, injective
                if dist[k] > max_centroid_dist:
                    break
                j = int(idx[k])
                if j in claimed_b:
                    continue
                claimed_b.add(j)
                matches.append((rem_a[k], rem_b[j], 0.0, float(dist[k]), "distance"))

    return pd.DataFrame(matches, columns=["row_a", "row_b", "intersection_area", "distance", "method"])


def track(frames: list[gpd.GeoDataFrame], *, max_centroid_dist: float = MAX_CENTROID_DIST) -> pd.DataFrame:
    """Compute frame-to-frame links for an ordered list of frame GeoDataFrames.

    Returns columns ``frame_a, frame_b, row_a, row_b, intersection_area,
    distance, method`` where ``frame_*`` are positions in ``frames`` (0-based).
    """
    all_links = []
    for i in range(len(frames) - 1):
        links = match_consecutive(frames[i], frames[i + 1], max_centroid_dist=max_centroid_dist)
        links["frame_a"] = i
        links["frame_b"] = i + 1
        all_links.append(links)
    return pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()


def build_trajectories(frames: list[gpd.GeoDataFrame], links: pd.DataFrame) -> pd.DataFrame:
    """Chain one-to-one frame links into trajectories.

    Returns one row per (trajectory, frame) with columns
    ``trajectory_id, frame_pos, frame_index, cell_id, label, centroid_x, centroid_y``.
    """
    tail: dict[tuple[int, int], int] = {}  # (frame_pos, row) -> trajectory_id at its current tail
    members: dict[int, list[tuple[int, int]]] = {}
    next_id = 0

    for link in links.sort_values(["frame_a", "row_a"]).itertuples(index=False):
        key_a = (int(link.frame_a), int(link.row_a))
        key_b = (int(link.frame_b), int(link.row_b))
        tid = tail.pop(key_a, None)
        if tid is None:
            tid = next_id
            next_id += 1
            members[tid] = [key_a]
        members[tid].append(key_b)
        tail[key_b] = tid

    rows = []
    for tid, cells in members.items():
        for frame_pos, row in cells:
            rec = frames[frame_pos].iloc[row]
            rows.append(
                {
                    "trajectory_id": tid,
                    "frame_pos": frame_pos,
                    "frame_index": int(rec["frame_index"]),
                    "cell_id": rec["cell_id"],
                    "label": int(rec["label"]),
                    "centroid_x": float(rec["centroid_x"]),
                    "centroid_y": float(rec["centroid_y"]),
                }
            )
    traj = pd.DataFrame(rows)
    return traj.drop_duplicates(subset=["trajectory_id", "frame_pos"]).reset_index(drop=True)


def close_gaps(traj: pd.DataFrame, *, max_gap: int = MAX_GAP, max_gap_dist: float = MAX_GAP_DIST) -> pd.DataFrame:
    """Stitch tracks broken by short segmentation dropouts.

    A trajectory ending at frame *i* is merged with one starting at frame *j*
    (``i+2 <= j <= i+1+max_gap``) when the gap-spanning endpoints are within
    ``max_gap_dist``. Trajectory ids are merged via union-find and relabelled.
    """
    if traj.empty or max_gap < 1:
        return traj

    g = traj.sort_values(["trajectory_id", "frame_pos"])
    first = g.groupby("trajectory_id").first()[["frame_pos", "centroid_x", "centroid_y"]]
    last = g.groupby("trajectory_id").last()[["frame_pos", "centroid_x", "centroid_y"]]

    heads_by_frame: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for tid, r in first.iterrows():
        heads_by_frame[int(r.frame_pos)].append((int(tid), r.centroid_x, r.centroid_y))

    parent = {int(t): int(t) for t in first.index}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    used_head: set[int] = set()
    for tid, r in last.sort_values("frame_pos").iterrows():
        tid = int(tid)
        fi = int(r.frame_pos)
        for j in range(fi + 2, fi + 2 + max_gap):
            cands = [(t, x, y) for (t, x, y) in heads_by_frame.get(j, []) if t not in used_head and find(t) != find(tid)]
            if not cands:
                continue
            pts = np.array([[x, y] for _, x, y in cands])
            d = np.hypot(pts[:, 0] - r.centroid_x, pts[:, 1] - r.centroid_y)
            k = int(d.argmin())
            if d[k] <= max_gap_dist:
                parent[find(cands[k][0])] = find(tid)
                used_head.add(cands[k][0])
                break

    out = traj.copy()
    out["trajectory_id"] = out["trajectory_id"].map(lambda t: find(int(t)))
    return out.sort_values(["trajectory_id", "frame_pos"]).reset_index(drop=True)


def trajectories_reaching(traj: pd.DataFrame, final_frame_pos: int | None = None) -> pd.DataFrame:
    """Keep only trajectories that include the final frame position.

    These are the cells linkable to Cell Painting. If ``final_frame_pos`` is
    None, the maximum ``frame_pos`` present is used.
    """
    if final_frame_pos is None:
        final_frame_pos = int(traj["frame_pos"].max())
    keep = traj.loc[traj["frame_pos"] == final_frame_pos, "trajectory_id"].unique()
    return traj[traj["trajectory_id"].isin(keep)].reset_index(drop=True)


def track_from_dir(
    gdf_dir: str | Path,
    pattern: str = "*.parquet",
    *,
    max_centroid_dist: float = MAX_CENTROID_DIST,
    max_gap: int = MAX_GAP,
    max_gap_dist: float = MAX_GAP_DIST,
):
    """Convenience: load frames, track, chain, and close gaps in one call.

    Returns ``(frames, trajectories)``.
    """
    frames = load_frame_gdfs(gdf_dir, pattern=pattern)
    links = track(frames, max_centroid_dist=max_centroid_dist)
    traj = build_trajectories(frames, links)
    traj = close_gaps(traj, max_gap=max_gap, max_gap_dist=max_gap_dist)
    return frames, traj
