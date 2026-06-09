# cell-motility-painting-aligner

A pip-installable Jupyter `anywidget` for landmark-based alignment of one
motility image against many larger Cell Painting images.

The package follows the same general framework as small custom `anywidget`
projects such as `bike_network_traffic`: Python traitlets define the widget
state, the JavaScript frontend is bundled with esbuild, and Hatch ships the
compiled widget module inside the wheel.

## Install for development

```bash
npm --prefix js install
npm --prefix js run build
pip install -e ".[dev]"
```

For a source or wheel build:

```bash
python -m build
```

## Notebook usage

```python
from cell_motility_painting_aligner import MotilityPaintingAligner

motility_centroids = [
    {"id": "mot_1", "x": 120, "y": 90},
    {"id": "mot_2", "x": 180, "y": 145},
    {"id": "mot_3", "x": 250, "y": 210},
]

cell_painting_centroids = [
    [
        {"id": "cp_a", "x": 1010, "y": 820},
        {"id": "cp_b", "x": 1210, "y": 995},
        {"id": "cp_c", "x": 1475, "y": 1210},
    ],
    [
        {"id": "cp_a", "x": 890, "y": 760},
        {"id": "cp_b", "x": 1090, "y": 930},
        {"id": "cp_c", "x": 1350, "y": 1145},
    ],
]

w = MotilityPaintingAligner.from_paths(
    "motility.png",
    ["cell_painting_001.png", "cell_painting_002.png"],
    motility_size=[512, 512],
    cell_painting_sizes=[[2048, 2048], [2048, 2048]],
    motility_centroids=motility_centroids,
    cell_painting_centroids_by_image=cell_painting_centroids,
)

w
```

Click a motility centroid on the left, then click the matching Cell Painting
centroid on the right. After creating at least two pairs for the current Cell
Painting image:

```python
fit = w.fit()
w.transform_by_image
w.inverse_transform_by_image
w.export_matches()
```

`transform_by_image` maps Cell Painting coordinates into motility coordinates.
`inverse_transform_by_image` maps motility coordinates into each Cell Painting
image.

For large image sets, prefer browser-accessible URLs instead of embedding local
files as base64 data URLs:

```python
w = MotilityPaintingAligner.from_urls(
    motility_image_url="http://localhost:8000/motility.png",
    cell_painting_image_urls=[
        "http://localhost:8000/cell_painting_001.png",
        "http://localhost:8000/cell_painting_002.png",
    ],
    motility_size=[512, 512],
    cell_painting_sizes=[[4096, 4096], [4096, 4096]],
)
```

## Data model

Coordinates are stored in native pixel space.

- `motility_centroids`: list of `{id, x, y}` records for the reference image
- `cell_painting_centroids_by_image`: one list of `{id, x, y}` records per
  Cell Painting image
- `matches_by_image`: per-image matched landmark pairs
- `transform_by_image`: per-image similarity transforms from Cell Painting to
  motility coordinates
- `inverse_transform_by_image`: per-image inverse transforms from motility to
  Cell Painting coordinates

The fitted transform is a 2D similarity transform: translation, uniform scale,
and rotation. Use three or more landmarks when possible so residuals and RMSE
are meaningful.

