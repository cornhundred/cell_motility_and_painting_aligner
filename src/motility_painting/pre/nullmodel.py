"""Random-walk null model for LCI trajectories.

Cells could look "directional" simply because a random walk with the same
step-length (speed) distribution occasionally drifts. This module builds that
null model and tests observed net displacement / heading persistence against
it.

Note on the null model: reshuffling a trajectory's own steps by *order* alone
(as in the early exploratory draft this replaces) cannot change net
displacement or path length -- vector summation is order-invariant, so
"shuffled" and "real" always sum to the identical endpoint. The null here
instead **keeps each step's speed but randomizes its direction** (iid uniform
angle), which destroys any correlation between consecutive step directions
while preserving the per-cell speed profile -- the standard "correlated vs.
uncorrelated random walk with matched step-length distribution" test, and the
only version of this null that can actually differ from the observed
trajectory.

Works on a long (one row per trajectory per frame) table like
``outputs/lci_trajectories_to_frame59.parquet`` -- no restriction to
trajectories reaching the final frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _steps_xy(g: pd.DataFrame, x_col: str, y_col: str, frame_col: str) -> np.ndarray:
    g = g.sort_values(frame_col)
    xs = g[x_col].to_numpy(dtype=float)
    ys = g[y_col].to_numpy(dtype=float)
    return np.column_stack([np.diff(xs), np.diff(ys)])


def stepwise_turning_angles(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    """Per-frame turning angle: how much heading changes at each interior frame.

    For every trajectory, the turning angle *at* frame ``i`` (for interior
    frames only -- the first and last frame of a trajectory have no turning
    angle) is the signed angle between the step arriving at frame ``i`` and
    the step leaving it. Useful as a general "how sharply is this cell
    redirecting right now" signal, e.g. to test whether a collision event
    coincides with a larger-than-usual turn.

    Returns ``trajectory_id, frame_index, turning_angle_deg``.
    """
    rows = []
    for tid, g in traj.groupby(traj_col):
        g = g.sort_values(frame_col)
        steps = _steps_xy(g, x_col, y_col, frame_col)
        if len(steps) < 2:
            continue
        headings = np.arctan2(steps[:, 1], steps[:, 0])
        turning = np.degrees(headings[1:] - headings[:-1])
        turning = (turning + 180.0) % 360.0 - 180.0
        frame_values = g[frame_col].to_numpy()[1:-1]
        for f, t in zip(frame_values, turning):
            rows.append({traj_col: tid, frame_col: int(f), "turning_angle_deg": float(t)})
    return pd.DataFrame(rows)


def _random_direction_steps(steps: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Same-speed, iid-random-direction resample of a trajectory's steps.

    Returns an ``(n_steps, 2)`` array of (dx, dy) -- same shape and per-step
    magnitude as ``steps``, but with each direction replaced by an independent
    uniform-random angle.
    """
    magnitudes = np.hypot(steps[:, 0], steps[:, 1])
    angles = rng.uniform(0.0, 2.0 * np.pi, size=len(steps))
    return magnitudes[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])


def randomize_step_directions(steps: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Return a trajectory (starting at the origin) with the same step speeds but iid random directions.

    ``steps`` is an ``(n_steps, 2)`` array of per-frame (dx, dy) displacements
    for one trajectory. Each step's magnitude (speed) is kept; its direction
    is replaced with an independent uniform-random angle, which destroys
    directional persistence while matching the cell's own speed profile.
    """
    random_steps = _random_direction_steps(steps, rng=rng)
    return np.vstack([np.zeros(2), np.cumsum(random_steps, axis=0)])


def permutation_displacement_test(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
    min_frames: int = 10,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-trajectory net displacement vs. a direction-randomized null.

    For each trajectory with at least ``min_frames`` positions, computes the
    observed net displacement and, from ``n_shuffles`` direction-randomized
    resamples (:func:`randomize_step_directions`), a null distribution of net
    displacement. Returns one row per trajectory: ``trajectory_id, n_frames,
    observed_displacement, null_mean, null_std, p_value`` (``p_value`` =
    fraction of null resamples with displacement >= observed -- a one-sided
    empirical permutation p-value).

    A population-level test (paired: is observed greater than each
    trajectory's own null mean?) can be run on the returned table with
    :func:`population_displacement_test`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for tid, g in traj.groupby(traj_col):
        if len(g) < min_frames:
            continue
        steps = _steps_xy(g, x_col, y_col, frame_col)
        observed = float(np.hypot(*steps.sum(axis=0)))

        null_disp = np.empty(n_shuffles)
        for i in range(n_shuffles):
            path = randomize_step_directions(steps, rng=rng)
            null_disp[i] = np.hypot(*(path[-1] - path[0]))

        rows.append(
            {
                traj_col: tid,
                "n_frames": int(len(g)),
                "observed_displacement": observed,
                "null_mean": float(null_disp.mean()),
                "null_std": float(null_disp.std()),
                "p_value": float((null_disp >= observed).mean()),
            }
        )
    return pd.DataFrame(rows)


def population_displacement_test(per_trajectory: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank test: observed vs. each trajectory's own null mean.

    ``per_trajectory`` is the output of :func:`permutation_displacement_test`.
    Returns ``{"n", "statistic", "p_value", "median_diff"}`` -- a single
    headline number for "are real trajectories more directed than a
    direction-randomized walk with the same speed profile, across the
    population".
    """
    diff = per_trajectory["observed_displacement"] - per_trajectory["null_mean"]
    stat, p = stats.wilcoxon(diff, alternative="greater")
    return {
        "n": int(len(diff)),
        "statistic": float(stat),
        "p_value": float(p),
        "median_diff": float(diff.median()),
    }


def turning_angle_autocorrelation(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
    min_frames: int = 10,
    max_lag: int = 5,
    n_shuffles: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Step-heading autocorrelation vs. lag, real vs. a direction-randomized null.

    For each trajectory, headings are ``atan2(dy, dx)`` per step; the
    autocorrelation at lag *k* is the mean cosine of the heading difference
    ``k`` steps apart, averaged first within a trajectory then across
    trajectories (real motion) and separately for ``n_shuffles``
    direction-randomized resamples of each trajectory's own step speeds
    (null, :func:`randomize_step_directions`). Persistent/directed motion
    shows real autocorrelation decaying slower than the null, which is flat
    near 0 at all lags by construction (iid random directions).

    Returns one row per lag: ``lag, real_mean, null_mean, null_std``.
    """
    rng = np.random.default_rng(seed)

    def _heading_autocorr(steps: np.ndarray, lag: int) -> float:
        if len(steps) <= lag:
            return np.nan
        headings = np.arctan2(steps[:, 1], steps[:, 0])
        return float(np.mean(np.cos(headings[lag:] - headings[:-lag])))

    real_by_lag: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}
    null_by_lag: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}

    for _, g in traj.groupby(traj_col):
        if len(g) < min_frames:
            continue
        steps = _steps_xy(g, x_col, y_col, frame_col)
        for lag in range(1, max_lag + 1):
            v = _heading_autocorr(steps, lag)
            if not np.isnan(v):
                real_by_lag[lag].append(v)

        null_vals: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}
        for _ in range(n_shuffles):
            random_steps = _random_direction_steps(steps, rng=rng)
            for lag in range(1, max_lag + 1):
                v = _heading_autocorr(random_steps, lag)
                if not np.isnan(v):
                    null_vals[lag].append(v)
        for lag in range(1, max_lag + 1):
            if null_vals[lag]:
                null_by_lag[lag].append(float(np.mean(null_vals[lag])))

    rows = []
    for lag in range(1, max_lag + 1):
        rows.append(
            {
                "lag": lag,
                "real_mean": float(np.mean(real_by_lag[lag])) if real_by_lag[lag] else np.nan,
                "null_mean": float(np.mean(null_by_lag[lag])) if null_by_lag[lag] else np.nan,
                "null_std": float(np.std(null_by_lag[lag])) if null_by_lag[lag] else np.nan,
            }
        )
    return pd.DataFrame(rows)
