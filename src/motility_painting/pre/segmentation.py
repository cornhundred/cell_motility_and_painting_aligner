"""Labeled instance mask -> polygon GeoDataFrame conversion.

The original Cellpose mask -> polygon step (that produced ``1.2GeoDataFrames/``)
was done upstream and its code no longer exists anywhere in this repo. This
module reconstructs an equivalent conversion so any new segmentation source
(e.g. InstanSeg) can be turned into the same per-frame polygon GeoDataFrame
schema (``cell_id, frame, label, geometry, source_file, centroid_x,
centroid_y``) that :mod:`.trajectories` and :mod:`.morphology` already consume.

Contours are traced with marching squares (:func:`skimage.measure.find_contours`
at level 0.5) rather than pixel-corner tracing: inspecting the existing
Cellpose-derived polygons shows half-integer vertex coordinates (e.g.
``(311.5, 26.0)``), which is the signature of marching squares, not blocky
pixel-boundary polygonization -- so this matches the original convention.

A label whose mask is disconnected (segmentation split one cell into several
polygons) yields multiple rows sharing the same ``label``/``cell_id``, exactly
like the existing data -- :func:`.centroids.dedupe_split_cells` already handles
that downstream.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely.geometry
from skimage.measure import find_contours, regionprops


def _contour_polygons(contour: np.ndarray, *, row_offset: int, col_offset: int) -> list[shapely.geometry.Polygon]:
    """One ``find_contours`` ring (row, col) -> one or more valid (x, y) polygons."""
    coords = [(col_offset + c, row_offset + r) for r, c in contour]
    poly = shapely.geometry.Polygon(coords)
    if poly.is_valid and poly.area > 0:
        return [poly]
    fixed = poly.buffer(0)
    if fixed.is_empty:
        return []
    if isinstance(fixed, shapely.geometry.MultiPolygon):
        return [p for p in fixed.geoms if p.area > 0]
    return [fixed] if fixed.area > 0 else []


def masks_to_polygons(mask: np.ndarray, *, source_file: str, min_area: float = 1.0) -> gpd.GeoDataFrame:
    """Convert a labeled instance mask (0 = background) into a polygon GeoDataFrame.

    Returns the same columns as ``1.2GeoDataFrames/Time*.parquet``:
    ``cell_id, frame, label, geometry, source_file, centroid_x, centroid_y``.
    ``cell_id`` is ``f"{source_file}_{label}"``, matching the existing convention.
    """
    rows: list[dict] = []
    for prop in regionprops(mask):
        label = int(prop.label)
        r0, c0, r1, c1 = prop.bbox
        # pad the crop by 1px so a contour touching the bbox edge still closes
        r0p, c0p = max(r0 - 1, 0), max(c0 - 1, 0)
        r1p, c1p = min(r1 + 1, mask.shape[0]), min(c1 + 1, mask.shape[1])
        crop = mask[r0p:r1p, c0p:c1p] == label

        for contour in find_contours(crop.astype(np.float32), level=0.5):
            if len(contour) < 4:
                continue
            for poly in _contour_polygons(contour, row_offset=r0p, col_offset=c0p):
                if poly.area < min_area:
                    continue
                rows.append(
                    {
                        "cell_id": f"{source_file}_{label}",
                        "frame": source_file,
                        "label": label,
                        "geometry": poly,
                        "source_file": source_file,
                        "centroid_x": poly.centroid.x,
                        "centroid_y": poly.centroid.y,
                    }
                )

    return gpd.GeoDataFrame(rows, geometry="geometry", columns=[
        "cell_id", "frame", "label", "geometry", "source_file", "centroid_x", "centroid_y",
    ])
