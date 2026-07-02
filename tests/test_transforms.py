import numpy as np
import pytest

from motility_painting import (
    apply_similarity_transform,
    fit_similarity_transform,
    invert_similarity_transform,
    serialize_transform,
)


def test_fit_similarity_transform_exact_row_vector_solution():
    source = np.array([[0.0, 0.0], [2.0, 0.0], [0.25, 1.5], [2.5, 1.75]])
    theta = 0.4
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    scale = 0.35
    translation = np.array([12.0, -8.5])
    target = scale * (source @ rotation) + translation

    result = fit_similarity_transform(source, target)

    np.testing.assert_allclose(result["scale"], scale)
    np.testing.assert_allclose(result["rotation"], rotation)
    np.testing.assert_allclose(result["translation"], translation)
    np.testing.assert_allclose(result["transformed"], target)
    assert result["rmse"] < 1e-12


def test_inverse_transform_round_trips_points():
    points = np.array([[10.0, 20.0], [30.0, -5.0], [2.0, 8.0]])
    theta = -0.2
    transform = {
        "scale": 2.25,
        "rotation": np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]),
        "translation": np.array([4.0, 9.0]),
        "rotation_radians": theta,
        "rotation_degrees": np.degrees(theta),
    }

    moved = apply_similarity_transform(points, transform)
    restored = apply_similarity_transform(moved, invert_similarity_transform(transform))

    np.testing.assert_allclose(restored, points)


def test_serialize_transform_is_json_friendly():
    source = np.array([[0.0, 0.0], [1.0, 0.0]])
    target = np.array([[2.0, 3.0], [3.0, 3.0]])

    serialized = serialize_transform(fit_similarity_transform(source, target))

    assert serialized["scale"] == pytest.approx(1.0)
    assert serialized["rotation"] == [[1.0, 0.0], [0.0, 1.0]]
    assert serialized["translation"] == [2.0, 3.0]
    assert serialized["rmse"] == pytest.approx(0.0)

