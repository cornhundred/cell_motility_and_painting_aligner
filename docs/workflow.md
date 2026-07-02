# Experiment Workflow

This project is primarily a cell correspondence workflow. Geometric alignment is
useful when it helps narrow candidate matches, but the core output is a mapping
from a terminal live-cell motility object to a Cell Painting object and its
phenotype measurements.

## Data Layout

The current experiment layout is assumed to look like this:

```text
data/
  cell_painting/
    2026-05-10/
      051026_1_F000.ims
      051026_1_F001.ims
      051026_1_F002.ims
      ...
  live_cell_imaging/
    2026-05-10/
      tiff files/
        Time00000_ChannelBrightfield LEB_Seq0000.tiff
        Time00001_ChannelBrightfield LEB_Seq0001.tiff
        Time00002_ChannelBrightfield LEB_Seq0002.tiff
        ...
```

For Cell Painting, each `.ims` file is treated as one tile or field of view.
For live-cell imaging, the terminal frame is the TIFF with the largest `Time`
index. That final frame is the image to match against the Cell Painting tiles.

The widget currently expects browser-displayable image URLs or embedded PNG/JPEG
data URLs plus centroid tables. The `.ims` and TIFF inputs should be converted
or rendered into display images before widget review, while the full-resolution
data remains the source of segmentation and feature measurements.

## Practical Goal

The immediate goal is not necessarily to warp one image onto another. The key
question is:

> Which Cell Painting cell corresponds to this live-cell motility track at the
> end of the movie?

The widget should support manual or semi-automated review of that final
correspondence:

- left panel: last live-cell frame with terminal motility cell centroids
- right panel: one Cell Painting tile at a time with CellProfiler object
  centroids
- output: `{live_track_id, final_frame_cell_id, cell_painting_tile_id,
  cell_painting_object_id}` plus any landmark or transform metadata used to
  make the match

When enough matched landmarks are available, a similarity transform can be fit
for a tile to propose candidate Cell Painting objects near each terminal
motility cell. When image geometry is unreliable, the curated cell match itself
is the source of truth.

## Processing Roadmap

1. Segment live-cell motility images

   Run Cellpose 2 on every live-cell frame to produce per-frame cell masks,
   centroids, outlines, and quality-control images.

2. Track cells through time

   Link per-frame objects into motility tracks. A first pass can use centroid
   distance, mask overlap, and assignment with a maximum displacement threshold.
   More advanced tracking can add division handling, missed detections, and
   track confidence.

3. Extract motility phenotypes

   Compute per-track measurements such as displacement, path length, speed,
   persistence, directionality, turning angle, endpoint angle, and any
   experiment-specific orientation measurements such as DAPI direction.

4. Process Cell Painting images

   Use CellProfiler to segment nuclei/cells and measure morphology, intensity,
   texture, radial distribution, neighbor context, and tile-level metadata for
   each `.ims` field of view.

5. Match terminal live cells to Cell Painting cells

   Use the last live-cell frame and the Cell Painting tile set. Candidate
   generation can use stage metadata, approximate tile location, centroid
   transforms, or neighborhood geometry. The widget is the review layer for
   confirming or correcting the correspondence.

6. Join phenotype tables

   Join motility track features to Cell Painting features by the curated match
   table. Keep unmatched, ambiguous, and low-confidence cells explicitly labeled
   so downstream statistics can filter them.

7. Test phenotype relationships

   Look for correlations or group differences between Cell Painting phenotype
   vectors and motility phenotypes such as speed, persistence, angle, and DAPI
   direction. Keep the first pass focused on reproducible, reviewable
   relationships between the two phenotype tables.

## Suggested Intermediate Files

```text
outputs/
  live_cell_imaging/
    2026-05-10/
      rendered_frames/
      cellpose_masks/
      per_frame_objects.parquet
      tracks.parquet
      motility_features.parquet
      qc/
  cell_painting/
    2026-05-10/
      rendered_tiles/
      cellprofiler_objects.parquet
      cellprofiler_features.parquet
      qc/
  matches/
    2026-05-10/
      terminal_live_to_cell_painting_matches.parquet
      terminal_live_to_cell_painting_matches.csv
      widget_state.json
  analysis/
    2026-05-10/
      joined_motility_cell_painting_features.parquet
      correlation_summary.csv
      figures/
```

## Widget Inputs

The widget should be fed derived review assets, not raw microscope containers:

- display image for the last live-cell frame
- display image for each Cell Painting tile
- terminal live-cell centroid table with stable track IDs
- Cell Painting object centroid table with stable object IDs and tile IDs
- optional candidate matches or initial landmarks

The raw TIFF and `.ims` data remain the authoritative input for segmentation,
tracking, and feature extraction.

## Starter Code

The reusable helpers live in `motility_painting.pipeline`:

- input discovery for the live-cell TIFF and Cell Painting `.ims` layouts
- Cellpose command construction for live-cell segmentation
- CellProfiler command construction for saved `.cppipe` pipelines
- conversion of Cellpose labeled masks into object centroid records
- centroid-based frame-to-frame tracking
- terminal track feature extraction
- candidate generation from aligner transforms
- curated match joins for downstream phenotype tables

Start with `examples/pipeline_starter.ipynb` and tune the segmentation, tracking,
and matching thresholds on one experiment date before scaling up.
