"""Trajectory scrubber: a deck.gl/anywidget playback viewer for LCI cell tracking.

Scrub or play through the live-cell frames and watch each tracked cell. Cells
whose trajectory reaches the final frame (the ones linkable to Cell Painting) are
drawn in colour with their full past+future path; every other detected cell is
drawn as a small black dot so you can see what tracking missed.

Frames are delivered as 8-bit WebP data URLs in the ``frame_urls`` traitlet (see
``motility_painting.pre.images``); centroids/paths are kept in native pixel space
and the frame bitmap is stretched to ``image_size`` so no rescaling is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import traitlets

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


class TrajectoryScrubber(anywidget.AnyWidget):
    """Play/scrub through LCI frames with tracked-cell paths and background cells."""

    _esm = _PACKAGE_DIR / "static" / "scrubber.js"
    _css = _PACKAGE_DIR / "static" / "scrubber.css"

    # One WebP data URL per frame position, ordered by timepoint.
    frame_urls = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    # The Time index of each frame position (for labelling only).
    frame_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    image_size = traitlets.List(traitlets.Int(), default_value=[2960, 2960]).tag(sync=True)

    # Trajectories that reach the final frame. Each entry:
    #   {"id": int, "pts": [[frame_pos, x, y], ...]}  (sorted by frame_pos)
    trajectories = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    # All detected centroids per frame position, drawn black under the tracks:
    #   [[[x, y], ...],  ...]  (outer index = frame_pos)
    background_points = traitlets.List(traitlets.List(traitlets.List(traitlets.Float())), default_value=[]).tag(sync=True)

    current_frame = traitlets.Int(0).tag(sync=True)
    show_full_path = traitlets.Bool(True).tag(sync=True)  # full past+future vs. windowed trail
    trail_length = traitlets.Int(12).tag(sync=True)       # windowed-trail length when show_full_path=False
    point_radius = traitlets.Float(6.0).tag(sync=True)    # head radius in image (world) px; shrinks when zoomed out
    show_image = traitlets.Bool(True).tag(sync=True)
    show_background = traitlets.Bool(True).tag(sync=True)
    playing = traitlets.Bool(False).tag(sync=True)
    fps = traitlets.Int(8).tag(sync=True)

    width = traitlets.Int(820).tag(sync=True)
    height = traitlets.Int(820).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    @classmethod
    def from_trajectories(
        cls,
        trajectories: Any,
        frame_urls: list[str],
        frame_indices: list[int],
        *,
        image_size: tuple[int, int] = (2960, 2960),
        only_reaching_last: bool = True,
        background_frames: Any = None,
        **kwargs: Any,
    ) -> "TrajectoryScrubber":
        """Build a scrubber from a trajectories DataFrame and prepared frames.

        ``trajectories`` is the table from
        :func:`motility_painting.pre.trajectories.build_trajectories` with columns
        ``trajectory_id, frame_index, centroid_x, centroid_y`` (``frame_pos`` is
        recomputed against ``frame_indices`` so the two stay consistent even if
        only a subset of frames were rendered).

        ``background_frames`` is an optional list of per-frame GeoDataFrames (the
        list returned by ``track_from_dir``); when given, every detected centroid
        is sent as a black background dot for that frame. Pass ``None`` to skip.
        """
        from ..pre.trajectories import trajectories_reaching

        traj = trajectories
        index_to_pos = {idx: pos for pos, idx in enumerate(frame_indices)}

        if only_reaching_last:
            traj = traj.copy()
            traj["frame_pos"] = traj["frame_index"].map(index_to_pos)
            traj = traj.dropna(subset=["frame_pos"])
            traj["frame_pos"] = traj["frame_pos"].astype(int)
            traj = trajectories_reaching(traj, len(frame_indices) - 1)

        records = []
        for tid, group in traj.groupby("trajectory_id"):
            pts = []
            for row in group.sort_values("frame_index").itertuples(index=False):
                pos = index_to_pos.get(int(row.frame_index))
                if pos is None:
                    continue
                pts.append([int(pos), round(float(row.centroid_x), 1), round(float(row.centroid_y), 1)])
            if pts:
                records.append({"id": int(tid), "pts": pts})

        background_points: list[list[list[float]]] = []
        if background_frames is not None:
            background_points = [[] for _ in frame_indices]
            for gdf in background_frames:
                fidx = int(gdf["frame_index"].iloc[0]) if "frame_index" in gdf.columns else None
                pos = index_to_pos.get(fidx)
                if pos is None:
                    continue
                xs = gdf["centroid_x"].to_numpy()
                ys = gdf["centroid_y"].to_numpy()
                background_points[pos] = [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)]

        return cls(
            frame_urls=list(frame_urls),
            frame_indices=list(frame_indices),
            image_size=[int(image_size[0]), int(image_size[1])],
            trajectories=records,
            background_points=background_points,
            **kwargs,
        )
