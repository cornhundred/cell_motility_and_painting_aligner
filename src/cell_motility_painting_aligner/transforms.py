from __future__ import annotations

from typing import Any

import numpy as np


def _as_point_array(points: Any, *, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_points, 2)")
    return arr


def fit_similarity_transform(source: Any, target: Any, allow_reflection: bool = False) -> dict:
    """Fit a 2D similarity transform from ``source`` points to ``target`` points.

    The returned transform uses row-vector coordinates::

        transformed = scale * (source @ rotation) + translation

    For this package, ``source`` is usually Cell Painting coordinates and
    ``target`` is motility-image coordinates.
    """
    source = _as_point_array(source, name="source")
    target = _as_point_array(target, name="target")

    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if source.shape[0] < 2:
        raise ValueError("at least two paired landmarks are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(np.sum(source_centered**2))
    if source_variance <= 0:
        raise ValueError("source landmarks have zero variance")

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)

    signs = np.ones(2)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        signs[-1] = -1

    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (source_mean @ rotation)
    transformed = apply_similarity_transform(source, {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    })
    residuals = target - transformed
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    rotation_radians = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    return {
        "scale": scale,
        "rotation": rotation,
        "rotation_radians": rotation_radians,
        "rotation_degrees": float(np.degrees(rotation_radians)),
        "translation": translation,
        "transformed": transformed,
        "residuals": residuals,
        "rmse": rmse,
    }


def apply_similarity_transform(points: Any, transform: dict) -> np.ndarray:
    """Apply a transform returned by ``fit_similarity_transform`` to points."""
    points = _as_point_array(points)
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    scale = float(transform["scale"])

    if rotation.shape != (2, 2):
        raise ValueError("transform rotation must have shape (2, 2)")
    if translation.shape != (2,):
        raise ValueError("transform translation must have shape (2,)")
    if scale == 0:
        raise ValueError("transform scale must be non-zero")

    return scale * (points @ rotation) + translation


def invert_similarity_transform(transform: dict) -> dict:
    """Return the inverse of a row-vector similarity transform."""
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    scale = float(transform["scale"])

    if rotation.shape != (2, 2):
        raise ValueError("transform rotation must have shape (2, 2)")
    if translation.shape != (2,):
        raise ValueError("transform translation must have shape (2,)")
    if scale == 0:
        raise ValueError("transform scale must be non-zero")

    inverse_rotation = rotation.T
    inverse_scale = 1.0 / scale
    inverse_translation = -(translation @ inverse_rotation) / scale
    rotation_radians = float(np.arctan2(inverse_rotation[1, 0], inverse_rotation[0, 0]))

    return {
        "scale": inverse_scale,
        "rotation": inverse_rotation,
        "rotation_radians": rotation_radians,
        "rotation_degrees": float(np.degrees(rotation_radians)),
        "translation": inverse_translation,
    }


def serialize_transform(transform: dict) -> dict:
    """Convert a transform/result dictionary into JSON-friendly values."""
    serialized = {
        "scale": float(transform["scale"]),
        "rotation": np.asarray(transform["rotation"], dtype=float).tolist(),
        "rotation_radians": float(transform["rotation_radians"]),
        "rotation_degrees": float(transform["rotation_degrees"]),
        "translation": np.asarray(transform["translation"], dtype=float).tolist(),
    }

    for key in ("transformed", "residuals"):
        if key in transform:
            serialized[key] = np.asarray(transform[key], dtype=float).tolist()
    if "rmse" in transform:
        serialized["rmse"] = float(transform["rmse"])

    return serialized

