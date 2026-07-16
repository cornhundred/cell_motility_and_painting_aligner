"""Export the already-executed result notebooks to standalone static HTML pages.

Uses nbconvert's HTMLExporter (exclude_input, embed_images), NOT
ipywidgets.embed.embed_minimal_html -- the latter has RequireJS loader
limitations that don't work with anywidget's ESM-based `_esm` (confirmed by
inspecting cornhundred/bike_network_traffic's own build script, which hit this
exact issue and switched to nbconvert). TrajectoryScrubber,
MotilityCellPaintingView, and celldega's Clustergram/Yearbook are all
anywidget.AnyWidget subclasses, so the same export path applies to all of them.

Every source notebook must already have executed outputs (including saved
ipywidgets state in `metadata.widgets`) -- this script does not execute
notebooks itself, since re-running e.g. the scrubber notebook would needlessly
redo expensive cached-asset prep. Run `jupyter nbconvert --execute` on a
notebook first if it doesn't have outputs yet.

Usage: python build_site.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

NOTEBOOKS_DIR = Path("/home/jovyan/workbench/claude_notebooks/notebooks")
SITE_DIR = Path(__file__).resolve().parent
SIZE_BUDGET_BYTES = 100 * 1024 * 1024

# (source notebook, output slug, nav title, one-line summary for the index card)
PAGES = [
    ("0_viz_1_motility_scrubber.ipynb", "scrubber", "Motility scrubber",
     "Scrub/play through all 65 LCI frames with tracked-cell paths."),
    ("0_viz_2_motility_vs_cellpainting.ipynb", "linked_view", "Linked view (motility ↔ CP)",
     "Synced final-frame motility | stitched Cell Painting panels, matched cells share a link_id."),
    ("0_viz_3_correlation_explorer.ipynb", "clustergram", "Clustergram & correlations",
     "Celldega clustergram of CP features + the CP↔motility correlation heatmap."),
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


def export_page(source_name: str, slug: str) -> int:
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
    # nbconvert's own default (html_manager_semver_range="*") fetches whatever the LATEST
    # @jupyter-widgets/html-manager is at page-load time -- incompatible with ipywidgets 8.x's
    # widget-state schema (newer Ajv rejects it: "strict mode: use allowUnionTypes"). Pin to the
    # same range ipywidgets.embed.embed_minimal_html already uses internally and is tested against.
    exporter.html_manager_semver_range = "^1.0.1"

    body, _ = exporter.from_notebook_node(nb)
    out_path = SITE_DIR / f"{slug}.html"
    out_path.write_text(body, encoding="utf-8")
    return out_path.stat().st_size


def main() -> None:
    total = 0
    print(f"{'page':<20} {'size (MB)':>10}")
    for source_name, slug, _title, _summary in PAGES:
        size = export_page(source_name, slug)
        total += size
        print(f"{slug:<20} {size / 1e6:>10.1f}")

    print(f"{'TOTAL':<20} {total / 1e6:>10.1f}")
    if total > SIZE_BUDGET_BYTES:
        print(f"WARNING: total site size {total / 1e6:.1f}MB exceeds the 100MB budget!")
    else:
        print(f"OK: under the 100MB budget ({total / 1e6:.1f}MB used).")


if __name__ == "__main__":
    main()
