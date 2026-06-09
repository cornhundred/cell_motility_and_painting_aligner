from pathlib import Path

import numpy as np

from cell_motility_painting_aligner.pipeline import (
    build_cellpose_command,
    build_cellprofiler_command,
    candidate_matches_from_transform,
    discover_cell_painting_tiles,
    discover_live_cell_frames,
    label_objects_from_mask,
    link_objects_by_centroid,
    terminal_detections,
    terminal_live_cell_frame,
    track_feature_table,
)


def test_discover_frames_and_tiles(tmp_path: Path):
    live = tmp_path / "live"
    cp = tmp_path / "cell_painting"
    live.mkdir()
    cp.mkdir()
    (live / "Time00002_ChannelBrightfield LEB_Seq0002.tiff").touch()
    (live / "Time00000_ChannelBrightfield LEB_Seq0000.tiff").touch()
    (cp / "051026_1_F010.ims").touch()
    (cp / "051026_1_F002.ims").touch()

    frames = discover_live_cell_frames(live)
    tiles = discover_cell_painting_tiles(cp)

    assert [frame.time_index for frame in frames] == [0, 2]
    assert terminal_live_cell_frame(frames).time_index == 2
    assert [tile.tile_id for tile in tiles] == ["F002", "F010"]


def test_build_cellpose_command_uses_absolute_dir(tmp_path: Path):
    cmd = build_cellpose_command(
        tmp_path,
        output_dir=tmp_path / "masks",
        model="cyto2",
        diameter=30,
        use_gpu=True,
    )

    assert cmd[:3] == [cmd[0], "-m", "cellpose"]
    assert "--dir" in cmd
    assert str(tmp_path.resolve()) in cmd
    assert "--savedir" in cmd
    assert "--pretrained_model" in cmd
    assert "cyto2" in cmd
    assert "--use_gpu" in cmd


def test_build_cellprofiler_command():
    cmd = build_cellprofiler_command(
        "pipeline.cppipe",
        "out",
        input_dir="images",
        groups={"Plate": "P1"},
        first_image_set=1,
        last_image_set=5,
    )

    assert cmd[:3] == ["cellprofiler", "-c", "-r"]
    assert "-p" in cmd
    assert "-i" in cmd
    assert "-g" in cmd
    assert "Plate=P1" in cmd
    assert "-f" in cmd
    assert "-l" in cmd


def test_label_objects_from_mask():
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [2, 2, 0, 0],
        ]
    )

    objects = label_objects_from_mask(mask, image_id="frame_000", frame=0)

    assert len(objects) == 2
    assert objects[0]["object_id"] == "frame_000:1"
    assert objects[0]["x"] == 1.3333333333333333
    assert objects[0]["y"] == 0.3333333333333333
    assert objects[1]["area"] == 2


def test_link_objects_and_track_features():
    objects = [
        {"frame": 0, "object_id": "a0", "x": 0.0, "y": 0.0},
        {"frame": 0, "object_id": "b0", "x": 20.0, "y": 0.0},
        {"frame": 1, "object_id": "a1", "x": 1.0, "y": 0.0},
        {"frame": 1, "object_id": "b1", "x": 20.0, "y": 2.0},
    ]

    tracked = link_objects_by_centroid(objects, max_distance=5)
    terminals = terminal_detections(tracked)
    features = track_feature_table(tracked)

    assert len({row["track_id"] for row in tracked}) == 2
    assert {row["object_id"] for row in terminals} == {"a1", "b1"}
    assert [feature["n_detections"] for feature in features] == [2, 2]
    assert features[0]["path_length_px"] == 1.0


def test_candidate_matches_from_transform():
    terminal = [{"track_id": "track_1", "object_id": "live_1", "x": 10.0, "y": 10.0}]
    cell_painting = [
        {"object_id": "cp_1", "tile_id": "F000", "x": 20.0, "y": 20.0},
        {"object_id": "cp_2", "tile_id": "F000", "x": 100.0, "y": 100.0},
    ]
    transform = {
        "scale": 1.0,
        "rotation": [[1.0, 0.0], [0.0, 1.0]],
        "translation": [-10.0, -10.0],
    }

    candidates = candidate_matches_from_transform(
        terminal,
        cell_painting,
        transform,
        max_distance=5,
    )

    assert len(candidates) == 1
    assert candidates[0]["live_track_id"] == "track_1"
    assert candidates[0]["cell_painting_object_id"] == "cp_1"

