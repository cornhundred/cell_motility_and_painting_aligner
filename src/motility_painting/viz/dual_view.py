"""Two-panel QC viewer for the CP <-> LCI linkage.

Left panel: the motility (LCI) final frame with its trajectory-endpoint
centroids. Right panel: the Cell Painting FOV tiles stitched into the *same* LCI
coordinate space (via the notebook-03 per-FOV transforms) with their cell
centroids. The two deck.gl views share one pan/zoom state, so the same region is
shown in both at once -- the visual sanity check that the alignment is right.

Interactivity (handled in ``static/dual_view.js``):
  * hover a centroid -> tooltip with its id and match status;
  * click a *matched* centroid -> highlight it and its partner in the other panel
    (they share a ``link_id``); the selection syncs to ``selected_link_id``;
  * matched vs unmatched cells are drawn in different colours, each toggleable.

This is a **separate** widget from :class:`TrajectoryScrubber`; the scrubber is
untouched. Only ``BitmapLayer``/``ScatterplotLayer`` are used, so the JS bundle
builds offline against the vendored deck.gl (no editable-layers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import traitlets

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


class MotilityCellPaintingView(anywidget.AnyWidget):
    """Side-by-side motility | stitched-Cell-Painting QC view in shared LCI coords."""

    _esm = _PACKAGE_DIR / "static" / "dual_view.js"
    _css = _PACKAGE_DIR / "static" / "dual_view.css"

    # --- left: motility ---
    # One or more WebP frame URLs; a slider scrubs the backdrop when >1 is given.
    motility_frame_urls = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    motility_frame_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    motility_size = traitlets.List(traitlets.Int(), default_value=[2960, 2960]).tag(sync=True)
    # Endpoint centroids: {"id": <traj id>, "x", "y", "link_id": int (-1 if unmatched)}
    motility_points = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)

    # --- right: stitched Cell Painting ---
    # Bitmaps already placed in LCI space: {"fov", "url", "bounds": [[x,y]*4]}
    cp_bitmaps = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    # CP centroids in LCI space: {"id": CellID, "x", "y", "link_id": int (-1 if unmatched)}
    cp_points = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)

    # --- shared state / controls ---
    current_frame = traitlets.Int(0).tag(sync=True)
    selected_link_id = traitlets.Int(-1).tag(sync=True)  # last-clicked matched pair (-1 = none)
    point_radius = traitlets.Float(6.0).tag(sync=True)   # world (common) units; shrinks when zoomed out
    show_matched = traitlets.Bool(True).tag(sync=True)
    show_unmatched = traitlets.Bool(True).tag(sync=True)
    show_cp_images = traitlets.Bool(True).tag(sync=True)
    show_motility_image = traitlets.Bool(True).tag(sync=True)
    show_labels = traitlets.Bool(True).tag(sync=True)   # link_id labels on matched cells (zoom-gated)
    cp_opacity = traitlets.Float(0.85).tag(sync=True)

    width = traitlets.Int(1180).tag(sync=True)
    height = traitlets.Int(620).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    @classmethod
    def from_linked(
        cls,
        linked: Any,
        motility_points: Any,
        cp_bitmaps: list[dict],
        *,
        motility_frame_urls: list[str],
        motility_frame_indices: list[int] | None = None,
        motility_size: tuple[int, int] = (2960, 2960),
        traj_col: str = "trajectory_id",
        cp_id_col: str = "CellID",
        cp_x: str = "x_lci",
        cp_y: str = "y_lci",
        link_col: str = "link_id",
        **kwargs: Any,
    ) -> "MotilityCellPaintingView":
        """Build the view from the notebook-04 linking outputs.

        Parameters
        ----------
        linked:
            The deduped links table (``linked_cells``/``linked_master``) with
            ``link_col``, ``traj_col``, ``cp_id_col`` and ``cp_x``/``cp_y``.
        motility_points:
            LCI endpoint centroids -- a DataFrame with ``trajectory_id`` +
            ``centroid_x``/``centroid_y`` (e.g. the Time64 endpoints). Matched
            trajectories get their ``link_id``; the rest are marked unmatched.
        cp_bitmaps:
            Output of :func:`motility_painting.pre.linking.cp_fov_bitmaps_in_lci`.
        """
        import pandas as pd

        link_by_traj = dict(zip(linked[traj_col], linked[link_col]))
        link_by_cp = dict(zip(linked[cp_id_col], linked[link_col]))

        mpts = []
        for r in motility_points.itertuples(index=False):
            tid = getattr(r, traj_col)
            mpts.append(
                {
                    "id": int(tid) if pd.notna(tid) else -1,
                    "x": float(getattr(r, "centroid_x")),
                    "y": float(getattr(r, "centroid_y")),
                    "link_id": int(link_by_traj.get(tid, -1)),
                }
            )

        # CP points: pass a full aligned CP table via cp_points= to also show the
        # unmatched cells (link_id == -1); otherwise only the linked cells render.
        cp_points = kwargs.pop("cp_points", None)
        if cp_points is None:
            cp_points = linked
        cpts = []
        for r in cp_points.itertuples(index=False):
            cid = getattr(r, cp_id_col)
            cpts.append(
                {
                    "id": str(cid),
                    "x": float(getattr(r, cp_x)),
                    "y": float(getattr(r, cp_y)),
                    "link_id": int(link_by_cp.get(cid, -1)),
                }
            )

        return cls(
            motility_frame_urls=list(motility_frame_urls),
            motility_frame_indices=list(motility_frame_indices or []),
            motility_size=[int(motility_size[0]), int(motility_size[1])],
            motility_points=mpts,
            cp_bitmaps=list(cp_bitmaps),
            cp_points=cpts,
            **kwargs,
        )
