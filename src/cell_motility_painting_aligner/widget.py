from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import traitlets

from .io import image_file_to_data_url
from .transforms import (
    apply_similarity_transform,
    fit_similarity_transform,
    invert_similarity_transform,
    serialize_transform,
)

_PACKAGE_DIR = Path(__file__).resolve().parent


class MotilityPaintingAligner(anywidget.AnyWidget):
    """Landmark matcher for one motility image and many Cell Painting images.

    The motility image is displayed on the left and treated as the reference
    coordinate system. The current Cell Painting image is displayed on the
    right. ``fit`` maps Cell Painting coordinates into motility coordinates;
    the inverse transform maps motility coordinates into a selected Cell
    Painting image.
    """

    _esm = _PACKAGE_DIR / "bundled" / "widget.js"
    _css = _PACKAGE_DIR / "style.css"

    motility_image_url = traitlets.Unicode("").tag(sync=True)
    cell_painting_image_urls = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    cell_painting_index = traitlets.Int(0).tag(sync=True)

    motility_size = traitlets.List(traitlets.Int(), default_value=[512, 512]).tag(sync=True)
    cell_painting_size = traitlets.List(traitlets.Int(), default_value=[2048, 2048]).tag(sync=True)
    cell_painting_sizes = traitlets.List(traitlets.List(traitlets.Int()), default_value=[]).tag(
        sync=True
    )

    # Expected point schema: {"id": "cell_1", "x": 100.0, "y": 200.0, ...}
    motility_centroids = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    cell_painting_centroids_by_image = traitlets.List(
        traitlets.List(traitlets.Dict()), default_value=[]
    ).tag(sync=True)

    # Stored per Cell Painting image:
    # {"0": [{"match_id": "...", "motility": {...}, "cell_painting": {...}}]}
    matches_by_image = traitlets.Dict(default_value={}).tag(sync=True)
    pending_motility = traitlets.Dict(default_value={}).tag(sync=True)
    active_match_id = traitlets.Unicode("").tag(sync=True)
    transform_by_image = traitlets.Dict(default_value={}).tag(sync=True)
    inverse_transform_by_image = traitlets.Dict(default_value={}).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    width = traitlets.Int(1100).tag(sync=True)
    height = traitlets.Int(560).tag(sync=True)
    point_radius = traitlets.Int(4).tag(sync=True)
    motility_label = traitlets.Unicode("MOTILITY / REFERENCE").tag(sync=True)
    cell_painting_label = traitlets.Unicode("CELL PAINTING / MOVING").tag(sync=True)

    @classmethod
    def from_paths(
        cls,
        motility_image_path: str | Path,
        cell_painting_image_paths: list[str | Path],
        **kwargs: Any,
    ) -> MotilityPaintingAligner:
        """Create a widget from local image files by embedding them as data URLs.

        This is convenient for prototypes and small images. For large Cell
        Painting images, prefer ``from_urls`` and serve files through HTTP.
        """
        return cls(
            motility_image_url=image_file_to_data_url(motility_image_path),
            cell_painting_image_urls=[
                image_file_to_data_url(path) for path in cell_painting_image_paths
            ],
            **kwargs,
        )

    @classmethod
    def from_urls(
        cls,
        motility_image_url: str,
        cell_painting_image_urls: list[str],
        **kwargs: Any,
    ) -> MotilityPaintingAligner:
        """Create a widget from browser-accessible image URLs."""
        return cls(
            motility_image_url=motility_image_url,
            cell_painting_image_urls=cell_painting_image_urls,
            **kwargs,
        )

    def current_matches(self, image_index: int | None = None) -> list[dict]:
        """Return matched landmark records for one Cell Painting image."""
        if image_index is None:
            image_index = self.cell_painting_index
        return list(self.matches_by_image.get(str(image_index), []))

    def paired_landmarks(self, image_index: int | None = None, *, min_matches: int = 2):
        """Return source/Cell Painting and target/motility landmark arrays."""
        if image_index is None:
            image_index = self.cell_painting_index
        matches = self.current_matches(image_index)
        if len(matches) < min_matches:
            raise ValueError(f"at least {min_matches} matched landmarks are required")

        source = np.asarray(
            [[m["cell_painting"]["x"], m["cell_painting"]["y"]] for m in matches],
            dtype=float,
        )
        target = np.asarray([[m["motility"]["x"], m["motility"]["y"]] for m in matches], dtype=float)
        return source, target

    def fit(
        self,
        image_index: int | None = None,
        *,
        min_matches: int = 2,
        allow_reflection: bool = False,
    ) -> dict:
        """Fit the current Cell Painting image into motility-image coordinates."""
        if image_index is None:
            image_index = self.cell_painting_index
        source, target = self.paired_landmarks(image_index, min_matches=min_matches)
        result = fit_similarity_transform(source, target, allow_reflection=allow_reflection)
        inverse = invert_similarity_transform(result)

        next_transform_by_image = dict(self.transform_by_image)
        next_inverse_by_image = dict(self.inverse_transform_by_image)
        next_transform_by_image[str(image_index)] = serialize_transform(result)
        next_inverse_by_image[str(image_index)] = serialize_transform(inverse)
        self.transform_by_image = next_transform_by_image
        self.inverse_transform_by_image = next_inverse_by_image
        self.status = f"fit image {image_index}: rmse={result['rmse']:.3f}px"
        return result

    def fit_all(
        self,
        *,
        min_matches: int = 2,
        allow_reflection: bool = False,
        skip_unmatched: bool = True,
    ) -> dict[int, dict]:
        """Fit transforms for every Cell Painting image with enough matches."""
        results: dict[int, dict] = {}
        next_transform_by_image = dict(self.transform_by_image)
        next_inverse_by_image = dict(self.inverse_transform_by_image)

        for image_index in range(len(self.cell_painting_image_urls)):
            try:
                source, target = self.paired_landmarks(image_index, min_matches=min_matches)
            except ValueError:
                if skip_unmatched:
                    continue
                raise

            result = fit_similarity_transform(source, target, allow_reflection=allow_reflection)
            inverse = invert_similarity_transform(result)
            results[image_index] = result
            next_transform_by_image[str(image_index)] = serialize_transform(result)
            next_inverse_by_image[str(image_index)] = serialize_transform(inverse)

        self.transform_by_image = next_transform_by_image
        self.inverse_transform_by_image = next_inverse_by_image
        self.status = f"fit {len(results)} image(s)"
        return results

    def transform_cell_painting_points(self, points: Any, image_index: int | None = None):
        """Map Cell Painting points into motility-image coordinates."""
        if image_index is None:
            image_index = self.cell_painting_index
        transform = self.transform_by_image.get(str(image_index))
        if transform is None:
            raise ValueError(f"no transform has been fit for image {image_index}")
        return apply_similarity_transform(points, transform)

    def transform_motility_points(self, points: Any, image_index: int | None = None):
        """Map motility points into the selected Cell Painting image coordinates."""
        if image_index is None:
            image_index = self.cell_painting_index
        transform = self.inverse_transform_by_image.get(str(image_index))
        if transform is None:
            raise ValueError(f"no inverse transform has been fit for image {image_index}")
        return apply_similarity_transform(points, transform)

    def clear_matches(self, image_index: int | None = None) -> None:
        """Clear all matches for one Cell Painting image."""
        if image_index is None:
            image_index = self.cell_painting_index
        next_matches = dict(self.matches_by_image)
        next_matches[str(image_index)] = []
        self.matches_by_image = next_matches
        self.pending_motility = {}
        self.status = f"cleared matches for image {image_index}"

    def export_matches(self) -> list[dict]:
        """Return matches as a flat list suitable for DataFrame construction."""
        rows: list[dict] = []
        for image_key, matches in sorted(self.matches_by_image.items(), key=lambda item: int(item[0])):
            image_index = int(image_key)
            for match in matches:
                motility = match["motility"]
                cell_painting = match["cell_painting"]
                rows.append(
                    {
                        "image_index": image_index,
                        "match_id": match.get("match_id", ""),
                        "motility_id": motility.get("id", ""),
                        "motility_x": float(motility["x"]),
                        "motility_y": float(motility["y"]),
                        "cell_painting_id": cell_painting.get("id", ""),
                        "cell_painting_x": float(cell_painting["x"]),
                        "cell_painting_y": float(cell_painting["y"]),
                    }
                )
        return rows
