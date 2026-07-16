"""Centroid extraction for live-cell-imaging (LCI) cells.

Segmentation (Cellpose) and the mask -> polygon GeoDataFrame conversion are done
upstream; this module reads the resulting per-frame GeoDataFrames (or the
consolidated MASTER parquet) and produces clean centroid tables.

The downstream goal only needs cells present in the *final* tracked frame, since
those are the ones linkable to the Cell Painting data, but the helpers here work
for any frame.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Pixel calibration (micrometres per pixel).
LCI_UM_PER_PX = 3.25  # 6.5 µm camera pixel / 2.0x objective

_TIME_RE = re.compile(r"Time(\d+)")


def parse_frame_index(frame_value: str) -> int:
    """Extract the integer timepoint from a frame/cell_id string.

    e.g. ``"Time00064_ChannelBrightfield LEB_Seq0064_cellpose_outputs"`` -> 64.
    """
    match = _TIME_RE.search(str(frame_value))
    if match is None:
        raise ValueError(f"could not parse a Time index from {frame_value!r}")
    return int(match.group(1))


def load_master(master_parquet: str | Path) -> gpd.GeoDataFrame:
    """Load the consolidated per-cell GeoDataFrame (all frames)."""
    gdf = gpd.read_parquet(master_parquet)
    if "frame_index" not in gdf.columns:
        gdf["frame_index"] = gdf["frame"].map(parse_frame_index)
    return gdf


def dedupe_split_cells(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse cells whose mask split into multiple polygons.

    Some ``(frame, label)`` pairs appear more than once because segmentation
    produced several disconnected polygons for one cell. We keep the
    largest-area polygon per ``(frame, label)`` and recompute its centroid so
    downstream point sets have exactly one centroid per cell.
    """
    if not {"frame", "label"}.issubset(gdf.columns):
        raise ValueError("expected 'frame' and 'label' columns")

    areas = gdf.geometry.area
    order = areas.sort_values(ascending=False).index
    deduped = gdf.loc[order].drop_duplicates(subset=["frame", "label"], keep="first")
    deduped = deduped.sort_index()
    # recompute centroids from the retained geometry to stay consistent
    deduped = deduped.copy()
    deduped["centroid_x"] = deduped.geometry.centroid.x
    deduped["centroid_y"] = deduped.geometry.centroid.y
    return deduped


def frame_centroids(
    gdf: gpd.GeoDataFrame,
    frame_index: int,
    *,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Return a centroid table for a single timepoint.

    Columns: ``cell_id, label, frame_index, centroid_x, centroid_y``.
    """
    sub = gdf[gdf["frame_index"] == frame_index]
    if dedupe:
        sub = dedupe_split_cells(sub)
    cols = ["cell_id", "label", "frame_index", "centroid_x", "centroid_y"]
    return pd.DataFrame(sub[cols]).reset_index(drop=True)


def final_frame_centroids(
    master_parquet: str | Path,
    *,
    last_frame: int | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Centroids of cells in the last tracked frame (the alignment point set).

    If ``last_frame`` is None the maximum frame index present is used.
    """
    gdf = load_master(master_parquet)
    if last_frame is None:
        last_frame = int(gdf["frame_index"].max())
    return frame_centroids(gdf, last_frame, dedupe=dedupe)


def to_widget_points(
    centroids: pd.DataFrame,
    *,
    id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> list[dict]:
    """Convert a centroid table to the ``[{"id","x","y"}, ...]`` schema the
    deck.gl aligner/scrubber widgets expect."""
    return [
        {"id": str(row[id_col]), "x": float(row[x_col]), "y": float(row[y_col])}
        for _, row in centroids.iterrows()
    ]
