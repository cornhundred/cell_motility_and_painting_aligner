"""Export the already-executed result notebooks to standalone static HTML pages.

Two kinds of pages, two different export mechanisms:

- **Widget pages** (scrubber, linked_view, clustergram) go through
  ``export_widget_page``: the widget's serialized state (``_esm``, ``_css``,
  and its traits) is pulled directly out of the executed notebook's
  ``metadata.widgets`` and rendered with a small custom static host
  (``vendor/afm_host.js``) implementing anywidget's own AFM host contract
  (https://anywidget.dev/en/afm). See ``site/engineering.html`` for why: the
  classic ipywidgets static-embedding path (nbconvert's HTMLExporter,
  ipywidgets.embed.embed_minimal_html -- both route through
  @jupyter-widgets/html-manager's requirejs/AMD loader) cannot load
  anywidget's real npm package, since it ships pure ESM with no AMD/UMD
  build. No CDN dependency at all with this approach.
- **Analysis pages** (NF01-05) go through nbconvert's ``HTMLExporter`` as
  before -- they're plain matplotlib notebooks with no widgets, so none of
  the above applies; nbconvert's default export works fine for these.

Every source notebook must already have executed outputs (including saved
ipywidgets state in `metadata.widgets` for the widget pages) -- this script
does not execute notebooks itself, since re-running e.g. the scrubber
notebook would needlessly redo expensive cached-asset prep. Run
`jupyter nbconvert --execute` on a notebook first if it doesn't have outputs
yet.

Usage: python build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

NOTEBOOKS_DIR = Path("/home/jovyan/workbench/claude_notebooks/notebooks")
SITE_DIR = Path(__file__).resolve().parent
SIZE_BUDGET_BYTES = 100 * 1024 * 1024

# (source notebook, output slug, nav title, one-line summary for the index card)
WIDGET_PAGES = [
    ("0_viz_1_motility_scrubber.ipynb", "scrubber", "Motility scrubber",
     "Scrub/play through all 65 LCI frames with tracked-cell paths."),
    ("0_viz_2_motility_vs_cellpainting.ipynb", "linked_view", "Linked view (motility ↔ CP)",
     "Synced final-frame motility | stitched Cell Painting panels, matched cells share a link_id."),
    ("0_viz_3_correlation_explorer.ipynb", "clustergram", "Clustergram & correlations",
     "Celldega clustergram of CP features (the CP-feature-vs-motility scatter picker needs a live "
     "kernel and isn't shown here -- see nb06/viz_3 for that)."),
]

ANALYSIS_PAGES = [
    ("NF01_displacement_null_model.ipynb", "nf01_displacement", "NF01 · Displacement null model",
     "Real trajectories travel farther than a direction-randomized walk (p=9.9e-16), but the effect is small (median +0.59px)."),
    ("NF02_morphology_over_time.ipynb", "nf02_morphology", "NF02 · Morphology over time",
     "Shape variability correlates with speed far more strongly (ρ≈0.6) than any fixed Cell Painting feature."),
    ("NF03_crowding_collective_motion.ipynb", "nf03_crowding", "NF03 · Crowding & collective motion",
     "Crowded cells move measurably slower (ρ=-0.091); weak local coordination in heading between neighbors."),
    ("NF04_multiple_testing_correction.ipynb", "nf04_bh_correction", "NF04 · Multiple-testing correction",
     "Real BH-FDR code confirms the previously-cited 41 Bonferroni / 87 BH-significant CP↔motility associations."),
    ("NF05_collisions.ipynb", "nf05_collisions", "NF05 · Collisions (\"bumper cars\")",
     "96,408 genuine polygon-contact collisions; weak contact-inhibition-of-locomotion signal."),
]

PAGES = WIDGET_PAGES + ANALYSIS_PAGES

# Keys that are ipywidgets/anywidget bookkeeping, not part of the widget's own
# traits -- excluded from the state object handed to the static model.
_BOOKKEEPING_KEYS = {
    "_esm", "_css", "_anywidget_id", "_dom_classes", "_model_module",
    "_model_module_version", "_model_name", "_view_count", "_view_module",
    "_view_module_version", "_view_name", "layout", "tabbable", "tooltip",
}

_WIDGET_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  #widget-root {{ width: {width}px; max-width: 100%; }}
  #widget-error {{ display: none; color: #b91c1c; font-family: ui-monospace, monospace; white-space: pre-wrap; }}
</style>
</head>
<body>
<div id="widget-root"></div>
<pre id="widget-error"></pre>
<script type="module">
  import {{ renderAFM }} from "./vendor/afm_host.js";
  const state = {state_json};
  // Resolve relative to THIS page (import.meta.url here), not to afm_host.js's own
  // location -- a relative specifier passed into renderAFM would otherwise be resolved
  // relative to vendor/afm_host.js by the import() call inside it, landing one directory
  // off (vendor/{slug}_widget.mjs instead of {slug}_widget.mjs).
  const esmUrl = new URL("./{slug}_widget.mjs", import.meta.url).href;
  const cssUrl = new URL("./{slug}_widget.css", import.meta.url).href;
  try {{
    await renderAFM({{ esmUrl, cssUrl, state, el: document.getElementById("widget-root") }});
  }} catch (err) {{
    console.error(err);
    const pre = document.getElementById("widget-error");
    pre.textContent = "Widget failed to render: " + err;
    pre.style.display = "block";
  }}
</script>
</body>
</html>
"""


def export_widget_page(source_name: str, slug: str, title: str) -> int:
    path = NOTEBOOKS_DIR / source_name
    nb = nbformat.read(path, as_version=4)

    widgets_meta = nb["metadata"].get("widgets", {})
    state_bundle = widgets_meta.get("application/vnd.jupyter.widget-state+json")
    if state_bundle is None:
        raise RuntimeError(f"{source_name} has no saved widget state -- execute it with a live kernel first")

    any_models = [m["state"] for m in state_bundle["state"].values() if m.get("model_name") == "AnyModel"]
    if len(any_models) != 1:
        raise RuntimeError(f"expected exactly one AnyModel widget in {source_name}, found {len(any_models)}")
    model_state = any_models[0]

    esm = model_state["_esm"]
    css = model_state.get("_css") or ""
    widget_state = {k: v for k, v in model_state.items() if k not in _BOOKKEEPING_KEYS}

    (SITE_DIR / f"{slug}_widget.mjs").write_text(esm, encoding="utf-8")
    (SITE_DIR / f"{slug}_widget.css").write_text(css, encoding="utf-8")

    html = _WIDGET_PAGE_TEMPLATE.format(
        title=title,
        width=widget_state.get("width", 900),
        state_json=json.dumps(widget_state).replace("</script", "<\\/script"),
        slug=slug,
    )
    out_path = SITE_DIR / f"{slug}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path.stat().st_size + len(esm.encode("utf-8")) + len(css.encode("utf-8"))


def export_analysis_page(source_name: str, slug: str) -> int:
    path = NOTEBOOKS_DIR / source_name
    nb = nbformat.read(path, as_version=4)

    has_outputs = any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    if not has_outputs:
        raise RuntimeError(
            f"{source_name} has no executed outputs -- run `jupyter nbconvert --execute` on it first"
        )

    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.embed_images = True

    body, _ = exporter.from_notebook_node(nb)
    out_path = SITE_DIR / f"{slug}.html"
    out_path.write_text(body, encoding="utf-8")
    return out_path.stat().st_size


def main() -> None:
    total = 0
    print(f"{'page':<20} {'size (MB)':>10}")
    for source_name, slug, _title, _summary in WIDGET_PAGES:
        size = export_widget_page(source_name, slug, _title)
        total += size
        print(f"{slug:<20} {size / 1e6:>10.1f}")
    for source_name, slug, _title, _summary in ANALYSIS_PAGES:
        size = export_analysis_page(source_name, slug)
        total += size
        print(f"{slug:<20} {size / 1e6:>10.1f}")

    print(f"{'TOTAL':<20} {total / 1e6:>10.1f}")
    if total > SIZE_BUDGET_BYTES:
        print(f"WARNING: total site size {total / 1e6:.1f}MB exceeds the 100MB budget!")
    else:
        print(f"OK: under the 100MB budget ({total / 1e6:.1f}MB used).")


if __name__ == "__main__":
    main()
