from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import apply_similarity_transform


@dataclass(frozen=True)
class LiveFrame:
    """A discovered live-cell imaging frame."""

    path: Path
    time_index: int
    sequence_index: int | None = None
    channel: str = ""


@dataclass(frozen=True)
class CellPaintingTile:
    """A discovered Cell Painting tile or field of view."""

    path: Path
    tile_index: int | None
    tile_id: str


def _glob_files(root: str | Path, pattern: str, *, recursive: bool) -> list[Path]:
    root = Path(root)
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber(pattern) if path.is_file())


def _parse_int(pattern: str, value: str, default: int | None = None) -> int | None:
    found = re.search(pattern, value)
    if found is None:
        return default
    return int(found.group(1))


def discover_live_cell_frames(
    root: str | Path,
    *,
    pattern: str = "Time*_Channel*.tif*",
    recursive: bool = False,
) -> list[LiveFrame]:
    """Discover live-cell TIFF frames and sort by time index."""
    frames: list[LiveFrame] = []
    for path in _glob_files(root, pattern, recursive=recursive):
        time_index = _parse_int(r"Time(\d+)", path.name)
        if time_index is None:
            continue
        sequence_index = _parse_int(r"Seq(\d+)", path.name)
        channel_match = re.search(r"Channel(.+?)_Seq", path.stem)
        frames.append(
            LiveFrame(
                path=path,
                time_index=time_index,
                sequence_index=sequence_index,
                channel=channel_match.group(1).strip() if channel_match else "",
            )
        )
    return sorted(frames, key=lambda frame: (frame.time_index, frame.sequence_index or -1))


def terminal_live_cell_frame(frames: Sequence[LiveFrame]) -> LiveFrame:
    """Return the live-cell frame with the largest time index."""
    if not frames:
        raise ValueError("no live-cell frames were provided")
    return max(frames, key=lambda frame: frame.time_index)


def discover_cell_painting_tiles(
    root: str | Path,
    *,
    pattern: str = "*.ims",
    recursive: bool = False,
) -> list[CellPaintingTile]:
    """Discover Cell Painting tile files and sort by tile index when present."""
    tiles: list[CellPaintingTile] = []
    for path in _glob_files(root, pattern, recursive=recursive):
        tile_index = _parse_int(r"_F(\d+)", path.stem)
        tile_id = f"F{tile_index:03d}" if tile_index is not None else path.stem
        tiles.append(CellPaintingTile(path=path, tile_index=tile_index, tile_id=tile_id))
    return sorted(
        tiles,
        key=lambda tile: (tile.tile_index is None, tile.tile_index if tile.tile_index is not None else tile.tile_id),
    )


def build_cellpose_command(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    model: str = "cyto2",
    diameter: float | None = None,
    chan: int = 0,
    chan2: int = 0,
    use_gpu: bool = False,
    save_tif: bool = True,
    save_png: bool = False,
    save_outlines: bool = True,
    save_rois: bool = False,
    python_executable: str = sys.executable,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a Cellpose command for segmenting a directory of live-cell frames."""
    cmd = [
        python_executable,
        "-m",
        "cellpose",
        "--dir",
        str(Path(input_dir).resolve()),
        "--pretrained_model",
        model,
        "--chan",
        str(chan),
        "--chan2",
        str(chan2),
    ]
    if output_dir is not None:
        cmd.extend(["--savedir", str(Path(output_dir).resolve())])
    if diameter is not None:
        cmd.extend(["--diameter", str(float(diameter))])
    if use_gpu:
        cmd.append("--use_gpu")
    if save_tif:
        cmd.append("--save_tif")
    if save_png:
        cmd.append("--save_png")
    if save_outlines:
        cmd.append("--save_outlines")
    if save_rois:
        cmd.append("--save_rois")
    cmd.extend(extra_args)
    return cmd


def run_cellpose_cli(*args: Any, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run Cellpose using ``build_cellpose_command`` arguments."""
    cmd = build_cellpose_command(*args, **kwargs)
    return subprocess.run(cmd, check=check)


def load_cellpose_segmentation(seg_npy_path: str | Path) -> dict:
    """Load a Cellpose ``*_seg.npy`` output file."""
    return np.load(seg_npy_path, allow_pickle=True).item()


def label_objects_from_mask(
    mask: Any,
    *,
    image_id: str,
    frame: int | None = None,
    min_area: int = 1,
    object_id_prefix: str | None = None,
) -> list[dict]:
    """Convert a labeled segmentation mask into object centroid records."""
    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise ValueError("mask must be a 2D labeled array")

    object_id_prefix = object_id_prefix or image_id
    objects: list[dict] = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        ys, xs = np.nonzero(labels == label)
        area = int(xs.size)
        if area < min_area:
            continue
        row = {
            "image_id": image_id,
            "object_label": label,
            "object_id": f"{object_id_prefix}:{label}",
            "x": float(xs.mean()),
            "y": float(ys.mean()),
            "area": area,
            "bbox_x_min": int(xs.min()),
            "bbox_y_min": int(ys.min()),
            "bbox_x_max": int(xs.max()),
            "bbox_y_max": int(ys.max()),
        }
        if frame is not None:
            row["frame"] = int(frame)
        objects.append(row)
    return objects


def objects_from_cellpose_segmentation(
    seg_npy_path: str | Path,
    *,
    image_id: str | None = None,
    frame: int | None = None,
    min_area: int = 1,
) -> list[dict]:
    """Load a Cellpose ``*_seg.npy`` file and return object centroid records."""
    seg_npy_path = Path(seg_npy_path)
    data = load_cellpose_segmentation(seg_npy_path)
    if "masks" not in data:
        raise ValueError(f"{seg_npy_path} does not contain a 'masks' array")
    return label_objects_from_mask(
        data["masks"],
        image_id=image_id or seg_npy_path.stem.replace("_seg", ""),
        frame=frame,
        min_area=min_area,
    )


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))


def _assignment_pairs(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    max_distance: float,
) -> list[tuple[int, int, float]]:
    if len(previous_points) == 0 or len(current_points) == 0:
        return []

    distances = _pairwise_distances(previous_points, current_points)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        pairs: list[tuple[int, int, float]] = []
        used_previous: set[int] = set()
        used_current: set[int] = set()
        candidates = [
            (float(distances[i, j]), i, j)
            for i in range(distances.shape[0])
            for j in range(distances.shape[1])
            if distances[i, j] <= max_distance
        ]
        for distance, previous_index, current_index in sorted(candidates):
            if previous_index in used_previous or current_index in used_current:
                continue
            pairs.append((previous_index, current_index, distance))
            used_previous.add(previous_index)
            used_current.add(current_index)
        return pairs

    previous_indices, current_indices = linear_sum_assignment(distances)
    return [
        (int(previous_index), int(current_index), float(distances[previous_index, current_index]))
        for previous_index, current_index in zip(previous_indices, current_indices, strict=False)
        if distances[previous_index, current_index] <= max_distance
    ]


def link_objects_by_centroid(
    objects: Iterable[Mapping[str, Any]],
    *,
    max_distance: float,
    frame_key: str = "frame",
    x_key: str = "x",
    y_key: str = "y",
    track_prefix: str = "track",
    max_frame_gap: int = 1,
) -> list[dict]:
    """Link per-frame objects into tracks using centroid nearest-neighbor assignment."""
    grouped: dict[int, list[dict]] = {}
    for obj in objects:
        frame = int(obj[frame_key])
        grouped.setdefault(frame, []).append(dict(obj))

    active: dict[str, dict] = {}
    tracked: list[dict] = []
    next_track_number = 1

    for frame in sorted(grouped):
        current = grouped[frame]
        eligible = [
            track
            for track in active.values()
            if frame - int(track["last_frame"]) <= max_frame_gap
        ]
        previous_points = np.asarray([[track[x_key], track[y_key]] for track in eligible], dtype=float)
        current_points = np.asarray([[obj[x_key], obj[y_key]] for obj in current], dtype=float)
        pairs = _assignment_pairs(previous_points, current_points, max_distance=max_distance)

        used_current: set[int] = set()
        for previous_index, current_index, distance in pairs:
            track_id = str(eligible[previous_index]["track_id"])
            row = dict(current[current_index])
            row["track_id"] = track_id
            row["link_distance"] = distance
            tracked.append(row)
            active[track_id] = {
                "track_id": track_id,
                "last_frame": frame,
                x_key: float(row[x_key]),
                y_key: float(row[y_key]),
            }
            used_current.add(current_index)

        for current_index, obj in enumerate(current):
            if current_index in used_current:
                continue
            track_id = f"{track_prefix}_{next_track_number:05d}"
            next_track_number += 1
            row = dict(obj)
            row["track_id"] = track_id
            row["link_distance"] = np.nan
            tracked.append(row)
            active[track_id] = {
                "track_id": track_id,
                "last_frame": frame,
                x_key: float(row[x_key]),
                y_key: float(row[y_key]),
            }

        active = {
            track_id: track
            for track_id, track in active.items()
            if frame - int(track["last_frame"]) <= max_frame_gap
        }

    return sorted(tracked, key=lambda row: (int(row[frame_key]), str(row["track_id"])))


def terminal_detections(
    tracked_objects: Iterable[Mapping[str, Any]],
    *,
    track_key: str = "track_id",
    frame_key: str = "frame",
) -> list[dict]:
    """Return the last detection for each track."""
    terminal: dict[str, dict] = {}
    for row in tracked_objects:
        track_id = str(row[track_key])
        frame = int(row[frame_key])
        if track_id not in terminal or frame > int(terminal[track_id][frame_key]):
            terminal[track_id] = dict(row)
    return sorted(terminal.values(), key=lambda row: str(row[track_key]))


def track_feature_table(
    tracked_objects: Iterable[Mapping[str, Any]],
    *,
    track_key: str = "track_id",
    frame_key: str = "frame",
    x_key: str = "x",
    y_key: str = "y",
) -> list[dict]:
    """Compute simple motility phenotype features from linked tracks."""
    grouped: dict[str, list[dict]] = {}
    for row in tracked_objects:
        grouped.setdefault(str(row[track_key]), []).append(dict(row))

    features: list[dict] = []
    for track_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row[frame_key]))
        points = np.asarray([[row[x_key], row[y_key]] for row in rows], dtype=float)
        diffs = np.diff(points, axis=0)
        step_distances = np.sqrt(np.sum(diffs**2, axis=1))
        displacement_vector = points[-1] - points[0]
        displacement = float(np.sqrt(np.sum(displacement_vector**2)))
        path_length = float(np.sum(step_distances))
        first_frame = int(rows[0][frame_key])
        last_frame = int(rows[-1][frame_key])
        duration = max(last_frame - first_frame, 0)
        features.append(
            {
                "track_id": track_id,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "n_detections": len(rows),
                "start_x": float(points[0, 0]),
                "start_y": float(points[0, 1]),
                "end_x": float(points[-1, 0]),
                "end_y": float(points[-1, 1]),
                "displacement_px": displacement,
                "path_length_px": path_length,
                "mean_speed_px_per_frame": path_length / duration if duration > 0 else np.nan,
                "persistence": displacement / path_length if path_length > 0 else np.nan,
                "endpoint_angle_degrees": float(
                    np.degrees(np.arctan2(displacement_vector[1], displacement_vector[0]))
                ),
            }
        )
    return sorted(features, key=lambda row: str(row["track_id"]))


def candidate_matches_from_transform(
    terminal_objects: Iterable[Mapping[str, Any]],
    cell_painting_objects: Iterable[Mapping[str, Any]],
    transform: Mapping[str, Any],
    *,
    max_distance: float,
    max_candidates: int = 3,
    live_id_key: str = "track_id",
    cell_painting_id_key: str = "object_id",
    tile_key: str = "tile_id",
) -> list[dict]:
    """Generate candidate terminal-live to Cell Painting object matches."""
    terminals = [dict(row) for row in terminal_objects]
    cell_objects = [dict(row) for row in cell_painting_objects]
    if not terminals or not cell_objects:
        return []

    cp_points = np.asarray([[row["x"], row["y"]] for row in cell_objects], dtype=float)
    cp_in_live = apply_similarity_transform(cp_points, dict(transform))

    candidates: list[dict] = []
    for terminal in terminals:
        terminal_point = np.asarray([float(terminal["x"]), float(terminal["y"])])
        distances = np.sqrt(np.sum((cp_in_live - terminal_point) ** 2, axis=1))
        ranked = sorted(
            (
                (float(distance), index)
                for index, distance in enumerate(distances)
                if distance <= max_distance
            ),
            key=lambda item: item[0],
        )[:max_candidates]
        for rank, (distance, index) in enumerate(ranked, start=1):
            cp = cell_objects[index]
            candidates.append(
                {
                    "live_track_id": str(terminal[live_id_key]),
                    "terminal_object_id": str(terminal.get("object_id", terminal[live_id_key])),
                    "cell_painting_object_id": str(cp[cell_painting_id_key]),
                    "cell_painting_tile_id": str(cp.get(tile_key, "")),
                    "rank": rank,
                    "distance_px": distance,
                    "terminal_x": float(terminal["x"]),
                    "terminal_y": float(terminal["y"]),
                    "cell_painting_x": float(cp["x"]),
                    "cell_painting_y": float(cp["y"]),
                    "cell_painting_x_in_live": float(cp_in_live[index, 0]),
                    "cell_painting_y_in_live": float(cp_in_live[index, 1]),
                }
            )
    return candidates


def merge_curated_matches(
    widget_matches: Iterable[Mapping[str, Any]],
    terminal_objects: Iterable[Mapping[str, Any]],
    cell_painting_objects: Iterable[Mapping[str, Any]],
    *,
    live_id_key: str = "track_id",
    cell_painting_id_key: str = "object_id",
) -> list[dict]:
    """Join widget-exported matches to terminal live objects and Cell Painting objects."""
    terminal_by_id = {str(row[live_id_key]): dict(row) for row in terminal_objects}
    cp_by_id = {str(row[cell_painting_id_key]): dict(row) for row in cell_painting_objects}

    joined: list[dict] = []
    for match in widget_matches:
        live_id = str(match.get("motility_id", match.get("live_track_id", "")))
        cp_id = str(match.get("cell_painting_id", match.get("cell_painting_object_id", "")))
        row = {
            "live_track_id": live_id,
            "cell_painting_object_id": cp_id,
            "match_id": match.get("match_id", ""),
            "image_index": match.get("image_index", ""),
        }
        if live_id in terminal_by_id:
            row.update({f"live_{key}": value for key, value in terminal_by_id[live_id].items()})
        if cp_id in cp_by_id:
            row.update({f"cell_painting_{key}": value for key, value in cp_by_id[cp_id].items()})
        joined.append(row)
    return joined


def build_cellprofiler_command(
    pipeline_path: str | Path,
    output_dir: str | Path,
    *,
    input_dir: str | Path | None = None,
    data_file: str | Path | None = None,
    file_list: str | Path | None = None,
    groups: Mapping[str, str] | None = None,
    first_image_set: int | None = None,
    last_image_set: int | None = None,
    executable: str = "cellprofiler",
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a headless CellProfiler command for a saved pipeline."""
    cmd = [
        executable,
        "-c",
        "-r",
        "-p",
        str(Path(pipeline_path)),
        "-o",
        str(Path(output_dir)),
    ]
    if input_dir is not None:
        cmd.extend(["-i", str(Path(input_dir))])
    if data_file is not None:
        cmd.extend(["--data-file", str(Path(data_file))])
    if file_list is not None:
        cmd.extend(["--file-list", str(Path(file_list))])
    for key, value in (groups or {}).items():
        cmd.extend(["-g", f"{key}={value}"])
    if first_image_set is not None:
        cmd.extend(["-f", str(first_image_set)])
    if last_image_set is not None:
        cmd.extend(["-l", str(last_image_set)])
    cmd.extend(extra_args)
    return cmd


def run_cellprofiler_cli(*args: Any, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run CellProfiler using ``build_cellprofiler_command`` arguments."""
    cmd = build_cellprofiler_command(*args, **kwargs)
    return subprocess.run(cmd, check=check)


def write_cellprofiler_file_list(paths: Iterable[str | Path], output_path: str | Path) -> Path:
    """Write one image path per line for CellProfiler's ``--file-list`` option."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{Path(path).resolve()}\n" for path in paths),
        encoding="utf-8",
    )
    return output_path
