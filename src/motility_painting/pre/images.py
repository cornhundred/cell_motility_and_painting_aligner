"""Prepare LCI brightfield frames for display in the deck.gl scrubber widget.

The CLAHE-processed frames are 8-bit 2960x2960 TIFFs. Browsers can't show TIFF
and 65 full-res frames are large, so we downsample and re-encode each frame as an
8-bit WebP data URL. Centroids are kept in the original pixel space and the
bitmap is stretched to the original image bounds in the widget, so no centroid
rescaling is needed.
"""

from __future__ import annotations

import base64
import io
import re
from glob import glob
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_TIME_RE = re.compile(r"Time(\d+)")

# Default false-colour palette for the 5 Cell Painting stains, keyed by the
# channel index in the .ims file (channel 5 is a combo channel, skipped):
#   0 DNA (405)  1 Mito (637)  2 WGA/AGP (561)  3 SYTO14/RNA (514)  4 ConA/ER (488)
CP_STAIN_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.15, 0.35, 1.00),   # DNA          -> blue
    1: (1.00, 0.20, 0.25),   # Mito         -> red
    2: (1.00, 0.55, 0.10),   # WGA / AGP    -> orange
    3: (1.00, 0.95, 0.25),   # SYTO14 / RNA -> yellow
    4: (0.25, 1.00, 0.35),   # ConA / ER    -> green
}
CP_STAIN_NAMES: dict[int, str] = {
    0: "DNA", 1: "Mito", 2: "AGP (WGA)", 3: "RNA (SYTO14)", 4: "ER (ConA)", 5: "combo",
}


def frame_image_paths(image_dir: str | Path, pattern: str = "*.tif*") -> list[Path]:
    """Return frame image paths sorted by their ``TimeNNNNN`` index."""
    files = glob(str(Path(image_dir) / pattern))
    return [Path(f) for f in sorted(files, key=lambda f: int(_TIME_RE.search(Path(f).name).group(1)))]


def frame_to_webp_data_url(
    path: str | Path,
    *,
    size: int = 1536,
    quality: int = 75,
) -> str:
    """Read one frame, downsample to ``size`` (longest side), encode as WebP data URL.

    Handles 16-bit input by contrast-stretching to 8-bit; 8-bit input (e.g. the
    CLAHE frames) passes through unchanged.
    """
    arr = tifffile.imread(str(path))
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if max(img.size) > size:
        img.thumbnail((size, size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def frames_to_webp_data_urls(
    image_dir: str | Path,
    *,
    size: int = 1536,
    quality: int = 75,
    pattern: str = "*.tif*",
    limit: int | None = None,
) -> tuple[list[str], list[int]]:
    """Convert a directory of frames to WebP data URLs.

    Returns ``(data_urls, frame_indices)`` ordered by timepoint. ``frame_indices``
    are the parsed ``TimeNNNNN`` values, useful for aligning with trajectory
    ``frame_index`` columns.
    """
    paths = frame_image_paths(image_dir, pattern=pattern)
    if limit is not None:
        paths = paths[:limit]
    urls = [frame_to_webp_data_url(p, size=size, quality=quality) for p in paths]
    indices = [int(_TIME_RE.search(p.name).group(1)) for p in paths]
    return urls, indices


def _stretch(a: np.ndarray, pct: tuple[float, float]) -> np.ndarray:
    """Percentile contrast-stretch a plane to [0, 1] float."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, pct)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def ims_channel_planes(
    path: str | Path,
    channels: tuple[int, ...],
    *,
    pct: tuple[float, float] = (1.0, 99.5),
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Read + contrast-stretch the given ``.ims`` channels.

    Returns ``(planes, (width, height))`` with one [0,1] float plane per channel
    in native FOV pixels.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        tp = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]
        planes = [_stretch(np.asarray(tp[f"Channel {c}"]["Data"])[0], pct) for c in channels]
    native_h, native_w = planes[0].shape
    return planes, (int(native_w), int(native_h))


def ims_composite_array(
    path: str | Path,
    *,
    channels: tuple[int, ...] = (0, 1, 2),
    pct: tuple[float, float] = (1.0, 99.5),
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compose ``channels`` straight into an RGB (or grayscale) [0,1] array."""
    planes, (w, h) = ims_channel_planes(path, channels, pct=pct)
    arr = planes[0] if len(planes) == 1 else np.stack(planes[:3], axis=-1)
    return arr, (w, h)


def ims_multicolor_array(
    path: str | Path,
    *,
    channel_colors: dict[int, tuple[float, float, float]] | None = None,
    pct: tuple[float, float] = (1.0, 99.5),
) -> tuple[np.ndarray, tuple[int, int]]:
    """False-colour composite of several stains into one RGB [0,1] array.

    Each channel is contrast-stretched, multiplied by its RGB colour and summed
    (additive blend, then clipped) -- the standard Cell Painting overlay look.
    ``channel_colors`` maps channel index -> (r,g,b) in [0,1]; defaults to
    :data:`CP_STAIN_COLORS` (the 5 stains, combo channel excluded).
    """
    channel_colors = channel_colors or CP_STAIN_COLORS
    chans = tuple(sorted(channel_colors))
    planes, (w, h) = ims_channel_planes(path, chans, pct=pct)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch, plane in zip(chans, planes):
        color = np.asarray(channel_colors[ch], dtype=np.float32)
        rgb += plane[..., None] * color[None, None, :]
    return np.clip(rgb, 0.0, 1.0), (w, h)


def _array_to_webp_url(arr: np.ndarray, *, size: int = 1024, quality: int = 80) -> str:
    """Encode a [0,1] float image array as a downsampled WebP data URL."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    if max(img.size) > size:
        img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ims_to_webp_data_url(
    path: str | Path,
    *,
    channels: tuple[int, ...] = (0, 1, 2),
    size: int = 1024,
    quality: int = 80,
    pct: tuple[float, float] = (1.0, 99.5),
) -> tuple[str, tuple[int, int]]:
    """Read a Cell Painting ``.ims`` (Imaris/HDF5) FOV, compose ``channels`` into
    an 8-bit (RGB or grayscale) WebP data URL.

    Returns ``(data_url, (width, height))`` in native FOV pixels (before
    downsample), so centroids in FOV-local pixel space can be overlaid by
    stretching the bitmap to those bounds.
    """
    arr, (w, h) = ims_composite_array(path, channels=channels, pct=pct)
    return _array_to_webp_url(arr, size=size, quality=quality), (w, h)


def cp_fov_webp_data_urls(
    ims_dir: str | Path,
    frame_ids: list[str],
    *,
    channels: tuple[int, ...] = (0, 1, 2),
    size: int = 1024,
    quality: int = 80,
    name_template: str = "051026_1_{fid}.ims",
) -> tuple[list[str], list[list[int]]]:
    """Build per-FOV CP WebP URLs for the given ``frame_ids`` (e.g. ``["F173", ...]``).

    Returns ``(urls, sizes)`` aligned with ``frame_ids`` (sizes = native FOV
    [w, h] for each, for the aligner's ``cell_painting_sizes`` trait).
    """
    urls, sizes = [], []
    for fid in frame_ids:
        path = Path(ims_dir) / name_template.format(fid=fid)
        url, (w, h) = ims_to_webp_data_url(path, channels=channels, size=size, quality=quality)
        urls.append(url)
        sizes.append([w, h])
    return urls, sizes


def native_frame_size(image_dir: str | Path, pattern: str = "*.tif*") -> tuple[int, int]:
    """Return the (width, height) of the first frame in native pixels."""
    paths = frame_image_paths(image_dir, pattern=pattern)
    if not paths:
        raise FileNotFoundError(f"no frames matching {pattern} in {image_dir}")
    arr = tifffile.imread(str(paths[0]))
    h, w = arr.shape[:2]
    return int(w), int(h)
