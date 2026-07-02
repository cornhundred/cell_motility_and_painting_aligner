import numpy as np
import pytest

from motility_painting import MotilityPaintingAligner


def test_widget_fit_updates_forward_and_inverse_transforms():
    widget = MotilityPaintingAligner.from_urls(
        motility_image_url="motility.png",
        cell_painting_image_urls=["cp.png"],
        motility_size=[100, 100],
        cell_painting_size=[200, 200],
    )
    widget.matches_by_image = {
        "0": [
            {
                "match_id": "L1",
                "motility": {"id": "m1", "x": 1.0, "y": 2.0},
                "cell_painting": {"id": "c1", "x": 10.0, "y": 20.0},
            },
            {
                "match_id": "L2",
                "motility": {"id": "m2", "x": 5.0, "y": 2.0},
                "cell_painting": {"id": "c2", "x": 30.0, "y": 20.0},
            },
            {
                "match_id": "L3",
                "motility": {"id": "m3", "x": 1.0, "y": 6.0},
                "cell_painting": {"id": "c3", "x": 10.0, "y": 40.0},
            },
        ]
    }

    result = widget.fit()

    assert result["rmse"] == pytest.approx(0.0)
    assert "0" in widget.transform_by_image
    assert "0" in widget.inverse_transform_by_image
    np.testing.assert_allclose(
        widget.transform_cell_painting_points([[30.0, 20.0]]),
        [[5.0, 2.0]],
    )
    np.testing.assert_allclose(
        widget.transform_motility_points([[5.0, 2.0]]),
        [[30.0, 20.0]],
    )


def test_export_matches_flattens_match_records():
    widget = MotilityPaintingAligner.from_urls(
        motility_image_url="motility.png",
        cell_painting_image_urls=["cp.png"],
    )
    widget.matches_by_image = {
        "0": [
            {
                "match_id": "L1",
                "motility": {"id": "m1", "x": 1.0, "y": 2.0},
                "cell_painting": {"id": "c1", "x": 10.0, "y": 20.0},
            }
        ]
    }

    assert widget.export_matches() == [
        {
            "image_index": 0,
            "match_id": "L1",
            "motility_id": "m1",
            "motility_x": 1.0,
            "motility_y": 2.0,
            "cell_painting_id": "c1",
            "cell_painting_x": 10.0,
            "cell_painting_y": 20.0,
        }
    ]

