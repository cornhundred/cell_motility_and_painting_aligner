"""Centroid + morphology cell tracking via the linear assignment problem (LAP).

An alternative to :mod:`.trajectories`'s greedy overlap/nearest-centroid tracker:
instead of accepting matches by descending overlap area then nearest distance,
this gates candidate pairs by a spatial radius and solves a proper minimum-cost
bipartite assignment (:func:`scipy.optimize.linear_sum_assignment`) over a cost
that combines centroid distance *and* shape-feature similarity (area,
elongation, solidity, ... from :mod:`.morphology`). The intent is to recover
correct links in ambiguous cases -- e.g. two cells that both moved close to
where the other used to be -- where distance alone can pick the wrong pair but
distance-plus-shape usually can't (a big flat cell and a small round one look
very different even at the same distance from your last position).

The candidate graph is naturally sparse (cells barely move relative to their
neighbour spacing here), so the full bipartite graph is split into connected
components first and each is solved as its own small dense LAP -- solving one
~4,700x4,700 matrix directly would be O(n^3) and intractable.

Output has the same ``row_a, row_b`` link schema as
:func:`.trajectories.match_consecutive`, so it plugs directly into the existing
:func:`.trajectories.build_trajectories` / :func:`.trajectories.close_gaps`
chaining -- nothing about those functions needed to change, and
:mod:`.trajectories`'s own tracker is untouched so the two can be compared.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .morphology import SHAPE_COLS, frame_shape_features

# Gating radius (pixels): only cells within this distance are even considered as
# candidates. Kept below the ~18.6px median neighbour spacing (see
# trajectories.py) so a link can't jump to a different cell -- shape similarity
# only re-ranks among genuinely nearby candidates, it doesn't replace gating.
GATE_DIST = 15.0
MAX_COST = 15.0  # reject assignments whose combined cost exceeds this


def _centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf["centroid_x"].to_numpy(), gdf["centroid_y"].to_numpy()])


def match_consecutive_lap(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    *,
    shape_cols: list[str] = SHAPE_COLS,
    gate_dist: float = GATE_DIST,
    dist_weight: float = 1.0,
    shape_weight: float = 1.0,
    max_cost: float | None = MAX_COST,
) -> pd.DataFrame:
    """One-to-one links from ``gdf_a`` to ``gdf_b`` via a gated LAP solve.

    Cost per candidate pair is ``dist_weight * centroid_distance +
    shape_weight * ||z-scored shape feature difference||``, z-scored jointly
    over both frames' cells so the two frames share one scale. Candidates are
    gated to pairs within ``gate_dist`` pixels (a :class:`~scipy.spatial.cKDTree`
    radius query); the bipartite candidate graph is split into connected
    components and each solved as an independent dense assignment, so this
    stays fast even though a global Hungarian solve over all cells would not.

    Returns columns ``row_a, row_b, cost, method`` (``method`` is always
    ``"lap"``), the same shape as :func:`.trajectories.match_consecutive`'s
    output so it can be passed straight to
    :func:`.trajectories.build_trajectories`.
    """
    n_a, n_b = len(gdf_a), len(gdf_b)
    if n_a == 0 or n_b == 0:
        return pd.DataFrame(columns=["row_a", "row_b", "cost", "method"])

    ca, cb = _centroids(gdf_a), _centroids(gdf_b)
    shape_a = frame_shape_features(gdf_a, dedupe=False)[shape_cols].to_numpy()
    shape_b = frame_shape_features(gdf_b, dedupe=False)[shape_cols].to_numpy()

    combined = np.vstack([shape_a, shape_b])
    mu = np.nanmean(combined, axis=0)
    sigma = np.nanstd(combined, axis=0)
    sigma[sigma == 0] = 1.0
    za = np.nan_to_num((shape_a - mu) / sigma)
    zb = np.nan_to_num((shape_b - mu) / sigma)

    tree_b = cKDTree(cb)
    candidates = tree_b.query_ball_point(ca, r=gate_dist)

    rows_i: list[int] = []
    cols_j: list[int] = []
    costs: list[float] = []
    for i, cand in enumerate(candidates):
        for j in cand:
            d = float(np.hypot(*(ca[i] - cb[j])))
            shape_d = float(np.linalg.norm(za[i] - zb[j]))
            costs.append(dist_weight * d + shape_weight * shape_d)
            rows_i.append(i)
            cols_j.append(j)

    if not rows_i:
        return pd.DataFrame(columns=["row_a", "row_b", "cost", "method"])

    # split the bipartite candidate graph into connected components (b-nodes
    # offset by n_a) so each is solved as its own small dense LAP instead of
    # one huge (and cubically slow) matrix over every cell in the frame.
    row_idx = np.asarray(rows_i)
    col_idx = np.asarray(cols_j) + n_a
    adj = coo_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(n_a + n_b, n_a + n_b))
    adj = adj + adj.T
    _, labels = connected_components(adj, directed=False)

    edges_by_component: dict[int, list[tuple[int, int, float]]] = {}
    for i, j, c in zip(rows_i, cols_j, costs):
        edges_by_component.setdefault(int(labels[i]), []).append((i, j, c))

    matches: list[tuple[int, int, float, str]] = []
    for edges in edges_by_component.values():
        uniq_a = sorted({i for i, _, _ in edges})
        uniq_b = sorted({j for _, j, _ in edges})
        a_index = {a: k for k, a in enumerate(uniq_a)}
        b_index = {b: k for k, b in enumerate(uniq_b)}

        infeasible = (max_cost if max_cost is not None else max(c for *_, c in edges)) * 10 + 1.0
        cost_mat = np.full((len(uniq_a), len(uniq_b)), infeasible)
        for i, j, c in edges:
            cost_mat[a_index[i], b_index[j]] = c

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        for r, c in zip(row_ind, col_ind):
            cost = cost_mat[r, c]
            if max_cost is not None and cost > max_cost:
                continue
            matches.append((uniq_a[r], uniq_b[c], float(cost), "lap"))

    return pd.DataFrame(matches, columns=["row_a", "row_b", "cost", "method"])


def track_lap(
    frames: list[gpd.GeoDataFrame],
    *,
    shape_cols: list[str] = SHAPE_COLS,
    gate_dist: float = GATE_DIST,
    dist_weight: float = 1.0,
    shape_weight: float = 1.0,
    max_cost: float | None = MAX_COST,
) -> pd.DataFrame:
    """Compute LAP-based frame-to-frame links for an ordered list of frames.

    Same shape/contract as :func:`.trajectories.track`: returns columns
    ``frame_a, frame_b, row_a, row_b, cost, method`` where ``frame_*`` are
    positions in ``frames`` (0-based).
    """
    all_links = []
    for i in range(len(frames) - 1):
        links = match_consecutive_lap(
            frames[i],
            frames[i + 1],
            shape_cols=shape_cols,
            gate_dist=gate_dist,
            dist_weight=dist_weight,
            shape_weight=shape_weight,
            max_cost=max_cost,
        )
        links["frame_a"] = i
        links["frame_b"] = i + 1
        all_links.append(links)
    return pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()
