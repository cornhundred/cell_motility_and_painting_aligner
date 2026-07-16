"""Random-walk null model for LCI trajectories, built only from real observed steps.

Cells could look "directional" simply because a walk with the same step
lengths occasionally drifts. This module builds a null model and tests
observed net displacement / heading persistence against it.

Two simpler nulls were ruled out:

- Reshuffling a trajectory's own steps by *order* alone cannot change net
  displacement or path length -- vector summation is order-invariant, so
  "shuffled" and "real" always sum to the identical endpoint.
- Keeping each step's speed but replacing its direction with a synthetic
  random angle *can* differ from the observed trajectory, but the angles
  aren't real data -- too far from what a cell actually does.

The null here instead **pools every real (dx, dy) step from every trajectory**
and builds each null trajectory by resampling (with replacement) that many
real steps from the pool. This uses only steps cells actually took -- never a
fabricated angle -- while breaking a trajectory's own temporal order/identity
(a null trajectory's steps come from many different cells at many different
moments), which is exactly the "no memory" hypothesis: if a cell's own step
sequence carries no directional persistence, its net displacement should look
like a random draw from the population-wide step pool.

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


def pool_all_steps(
    traj: pd.DataFrame,
    *,
    traj_col: str = "trajectory_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    frame_col: str = "frame_index",
) -> np.ndarray:
    """Every real (dx, dy) step from every trajectory, pooled into one ``(n_steps, 2)`` array.

    This is the "big pot of steps" the resampling null draws from -- every
    entry is a step some real cell actually took somewhere in the movie.
    """
    parts = [
        _steps_xy(g, x_col, y_col, frame_col) for _, g in traj.groupby(traj_col) if len(g) >= 2
    ]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 2))


def resample_from_pool(n_steps: int, pool: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_steps`` real steps at random (with replacement) from ``pool``.

    Returns an ``(n_steps, 2)`` array of (dx, dy) -- a mix of real steps taken
    from (potentially many) different cells at different moments, so a
    trajectory built from this has no directional memory by construction, but
    every individual step is a real observed one.
    """
    idx = rng.integers(0, len(pool), size=n_steps)
    return pool[idx]


def _pool_resampled_path(n_steps: int, pool: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """A trajectory (starting at the origin) built from ``n_steps`` pool-resampled real steps."""
    steps = resample_from_pool(n_steps, pool, rng=rng)
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


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
    """Per-trajectory net displacement vs. a pool-resampled null (real steps only).

    For each trajectory with at least ``min_frames`` positions, computes the
    observed net displacement and, from ``n_shuffles`` null trajectories built
    by resampling that many real steps from the whole-population pool
    (:func:`pool_all_steps` / :func:`resample_from_pool`), a null distribution
    of net displacement. Returns one row per trajectory: ``trajectory_id,
    n_frames, observed_displacement, null_mean, null_std, p_value``
    (``p_value`` = fraction of null resamples with displacement >= observed --
    a one-sided empirical permutation p-value).

    A population-level test (paired: is observed greater than each
    trajectory's own null mean?) can be run on the returned table with
    :func:`population_displacement_test`.
    """
    rng = np.random.default_rng(seed)
    pool = pool_all_steps(traj, traj_col=traj_col, x_col=x_col, y_col=y_col, frame_col=frame_col)

    rows = []
    for tid, g in traj.groupby(traj_col):
        if len(g) < min_frames:
            continue
        steps = _steps_xy(g, x_col, y_col, frame_col)
        observed = float(np.hypot(*steps.sum(axis=0)))

        null_disp = np.empty(n_shuffles)
        for i in range(n_shuffles):
            path = _pool_resampled_path(len(steps), pool, rng=rng)
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
    pool-resampled walk built from real steps, across the population".
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
    """Step-heading autocorrelation vs. lag, real vs. a pool-resampled null (real steps only).

    For each trajectory, headings are ``atan2(dy, dx)`` per step; the
    autocorrelation at lag *k* is the mean cosine of the heading difference
    ``k`` steps apart, averaged first within a trajectory then across
    trajectories (real motion) and separately for ``n_shuffles`` pool-resampled
    trajectories built from that trajectory's own step count
    (null, :func:`resample_from_pool`). Persistent/directed motion shows real
    autocorrelation decaying slower than the null, which is flat near 0 at all
    lags by construction (steps are drawn independently from the pool, so
    consecutive null steps have no directional relationship).

    Returns one row per lag: ``lag, real_mean, null_mean, null_std``.
    """
    rng = np.random.default_rng(seed)
    pool = pool_all_steps(traj, traj_col=traj_col, x_col=x_col, y_col=y_col, frame_col=frame_col)

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
            random_steps = resample_from_pool(len(steps), pool, rng=rng)
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
