# Site

Static, kernel-free HTML export of the project's widgets and analysis notebooks — browse the
results without opening Jupyter. Built the same way as
[bike_network_traffic](https://cornhundred.github.io/bike_network_traffic/): nbconvert's
`HTMLExporter` on already-executed notebooks (not `ipywidgets.embed.embed_minimal_html`, which
doesn't render anywidget's ESM-based widgets correctly).

## View it locally

```bash
cd site
python -m http.server 8000
```

Then open `http://localhost:8000/`.

## Rebuild

Source notebooks live in `claude_notebooks/notebooks/` (a separate, non-git directory) and must
already be executed (`jupyter nbconvert --to notebook --execute ...`) before running:

```bash
python build_site.py
```

This regenerates every `*.html` page in this directory and prints each file's size plus the
running total (kept under 100MB).

## Not yet built (tracked as follow-up work)

Search-by-cell-id with pan/zoom, per-cell highlight-by-criteria coloring, a scrubber finale that
appends the aligned Cell Painting mosaic, and a Celldega Yearbook integration are all deferred —
this first pass ships the widgets exactly as they are today.
