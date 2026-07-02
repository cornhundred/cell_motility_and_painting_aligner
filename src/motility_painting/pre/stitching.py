"""Cell Painting FOV stitching into a global mosaic coordinate system.

CellProfiler reports each cell's position *within* its field of view (FOV/tile).
To compare against the live-cell mosaic we place every cell into one global
coordinate frame using the per-tile stage positions recorded in the TeraStitcher
XML.

Two placements are provided:
  * ``add_global_coords`` (primary) -- absolute stage coordinates (ABS_H/ABS_V).
  * ``add_snake_grid_coords`` (sanity check) -- reconstruct the snake tile layout
    from COL/ROW with an assumed overlap; useful to confirm the stage-coordinate
    mosaic looks right.

This consolidates the draft ``2.1 Combining MD with SC`` and ``2.3 Stitching SC
via MD`` notebooks (which had two competing methods, a path typo, and leftover
mask->GDF code that belongs to the segmentation stage we now skip).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

VOXEL_SIZE_UM = 0.3015  # Cell Painting µm/px
TILE_SIZE_PX = 2048
DEFAULT_OVERLAP = 0.10
_FRAME_RE = r"(F\d{3})"


def parse_stage_positions(xml_path: str | Path) -> pd.DataFrame:
    """Parse per-tile stage positions from the TeraStitcher XML.

    Returns columns ``FrameID, ABS_HORIZONTAL, ABS_VERTICAL, COL, ROW``.
    """
    root = ET.parse(Path(xml_path)).getroot()
    records = []
    for stack in root.findall(".//Stack"):
        img = stack.attrib.get("IMG_REGEX", "")
        frame = re.search(_FRAME_RE, img)
        if frame is None:
            continue
        records.append(
            {
                "FrameID": frame.group(1),
                "ABS_HORIZONTAL": float(stack.attrib["ABS_H"]),
                "ABS_VERTICAL": float(stack.attrib["ABS_V"]),
                "COL": int(stack.attrib["COL"]),
                "ROW": int(stack.attrib["ROW"]),
            }
        )
    return pd.DataFrame(records)


def combine_sc_with_positions(
    sc_csv: str | Path,
    xml_path: str | Path,
    *,
    filename_col: str = "FileName_OrigActin_Golgi_Membrane",
) -> pd.DataFrame:
    """Merge CellProfiler single-cell measurements with per-tile stage positions.

    Reproduces notebook ``2.1``: extract the ``F###`` tile id from each cell's
    image filename and left-join the XML stage metadata.
    """
    df = pd.read_csv(sc_csv)
    df["FrameID"] = df[filename_col].str.extract(_FRAME_RE)
    positions = parse_stage_positions(xml_path)
    merged = df.merge(positions, on="FrameID", how="left")
    return merged


def add_global_coords(
    df: pd.DataFrame,
    *,
    voxel_size_um: float = VOXEL_SIZE_UM,
    x_col: str = "Location_Center_X",
    y_col: str = "Location_Center_Y",
) -> pd.DataFrame:
    """Add ``Local_*_um`` and stage-based ``Global_*_um`` columns (primary method).

    Requires ``ABS_HORIZONTAL``/``ABS_VERTICAL`` from
    :func:`combine_sc_with_positions`.
    """
    out = df.copy()
    out["Local_X_um"] = out[x_col] * voxel_size_um
    out["Local_Y_um"] = out[y_col] * voxel_size_um
    out["Global_X_um"] = out["Local_X_um"] + out["ABS_HORIZONTAL"]
    out["Global_Y_um"] = out["Local_Y_um"] + out["ABS_VERTICAL"]
    return out


def add_snake_grid_coords(
    df: pd.DataFrame,
    tile_columns: list[list[int]],
    *,
    voxel_size_um: float = VOXEL_SIZE_UM,
    tile_size_px: int = TILE_SIZE_PX,
    overlap: float = DEFAULT_OVERLAP,
    filename_col: str = "FileName_OrigActin_Golgi_Membrane",
) -> pd.DataFrame:
    """Add ``Vis_X``/``Vis_Y`` from a reconstructed snake tile layout (sanity check).

    ``tile_columns`` is a list of frame-number lists, one per mosaic column,
    laid out in a boustrophedon (snake) order as in notebook ``2.3``.
    """
    out = df.copy()
    tile_to_grid: dict[str, tuple[int, int]] = {}
    for col_idx, column in enumerate(tile_columns):
        ordered = column if col_idx % 2 == 0 else column[::-1]
        for row_idx, tile in enumerate(ordered):
            tile_to_grid[f"F{tile:03d}"] = (col_idx, row_idx)

    out["tile_id"] = out[filename_col].str.extract(_FRAME_RE)
    out["grid_x"] = out["tile_id"].map(lambda t: tile_to_grid.get(t, (float("nan"),) * 2)[0])
    out["grid_y"] = out["tile_id"].map(lambda t: tile_to_grid.get(t, (float("nan"),) * 2)[1])

    step_um = tile_size_px * voxel_size_um * (1 - overlap)
    if "Local_X_um" not in out.columns:
        out["Local_X_um"] = out["Location_Center_X"] * voxel_size_um
        out["Local_Y_um"] = out["Location_Center_Y"] * voxel_size_um
    out["Vis_X"] = out["grid_x"] * step_um + out["Local_X_um"]
    out["Vis_Y"] = out["grid_y"] * step_um + out["Local_Y_um"]
    return out


def cp_fov_local_point_sets(
    df: pd.DataFrame,
    frame_ids: list[str],
    *,
    x_col: str = "Location_Center_X",
    y_col: str = "Location_Center_Y",
    id_col: str = "ObjectNumber",
) -> list[list[dict]]:
    """Per-FOV cell centroids in FOV-local pixel coords, for the aligner.

    Returns a list aligned with ``frame_ids``; each item is a list of
    ``{"id", "x", "y}`` dicts (the widget's ``cell_painting_centroids_by_image``
    schema). ``df`` must have a ``FrameID`` column (from
    :func:`combine_sc_with_positions`).
    """
    sets = []
    for fid in frame_ids:
        sub = df[df["FrameID"] == fid]
        pts = []
        for r in sub.itertuples(index=False):
            oid = getattr(r, id_col, None)
            pts.append(
                {
                    "id": f"{fid}_{oid}" if oid is not None else fid,
                    "x": float(getattr(r, x_col)),
                    "y": float(getattr(r, y_col)),
                }
            )
        sets.append(pts)
    return sets


def cp_global_centroids(
    df: pd.DataFrame,
    *,
    id_col: str | None = None,
    orientation_col: str = "AreaShape_Orientation",
) -> pd.DataFrame:
    """Compact table of CP cell centroids in global coords for alignment.

    Keeps ``AreaShape_Orientation`` (if present) so the downstream analysis can
    compare CP cell orientation against motility heading after applying the
    Procrustes rotation.
    """
    cols = {
        "cell_id": df[id_col] if id_col and id_col in df.columns else df.index.astype(str),
        "FrameID": df.get("FrameID"),
        "centroid_x": df["Global_X_um"],
        "centroid_y": df["Global_Y_um"],
    }
    if orientation_col in df.columns:
        cols["orientation_deg"] = df[orientation_col]
    return pd.DataFrame(cols).reset_index(drop=True)
