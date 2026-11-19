# Full-screen screenshot benchmark dataset

This directory is reserved for **user-captured gameplay screenshots** and their
manual ground-truth labels. Screenshots are intentionally not fabricated or
copied from icon crops: a valid detection benchmark needs complete, real
screenshots at their original resolution.

## Add a screenshot

1. Capture a representative Kiomet gameplay screen and save the PNG in this
   directory. Include varied zoom levels, territory colours, crowded areas,
   isolated entities, and special units across multiple matches.
2. Create or update `labels.json` using the schema below.
3. Draw a tight bounding box around every visible tower and unit marker. Do
   not label roads, territory, UI controls, minimap decorations, or text.
4. Set `fully_annotated` to `true` only after checking the entire screenshot.
   This is essential: without exhaustive labels, an unmatched model proposal
   cannot reliably be counted as a false positive.
5. Record `expected_name` where the icon identity is known. The category and
   box can still be evaluated when identity is unknown; omit the name rather
   than guessing.
6. Keep screenshots and labels in this folder, and do not include personal
   information or unrelated desktop content in captures.

## Label schema

Coordinates are normalized to the original screenshot: top-left is (0, 0),
bottom-right is (1, 1). Each box contains `x`, `y`, `width`, and `height`.
An exhaustively reviewed screenshot with no visible entities uses an empty
`entities` array.

```json
{
  "schema_version": 1,
  "label_source": "Manually reviewed gameplay screenshot",
  "screenshots": [
    {
      "name": "match_01_overview",
      "image": "match_01_overview.png",
      "fully_annotated": true,
      "entities": [
        {
          "id": "entity_001",
          "category": "tower",
          "expected_name": "Radar",
          "bbox": {
            "x": 0.21,
            "y": 0.34,
            "width": 0.035,
            "height": 0.048
          }
        }
      ]
    }
  ]
}
```

The numeric coordinates above are illustrative only. Replace them with
measurements from the actual screenshot; never use the example as real data.

## Run

From the repository root, with the project virtual environment active:

```powershell
.\.venv\Scripts\python.exe -m perception.screenshot_detection_benchmark --labels benchmark_screenshots/labels.json --max-image-side 960
```

Compare input resolutions without changing recognition thresholds:

```powershell
.\.venv\Scripts\python.exe -m perception.screenshot_detection_benchmark --max-image-side 960 --output benchmark_results/detection_960.json
.\.venv\Scripts\python.exe -m perception.screenshot_detection_benchmark --max-image-side 1280 --output benchmark_results/detection_1280.json
```

Use `--tiled` for the higher-detail tiled path. Keep model, screenshot set,
IoU threshold, and confidence thresholds identical when comparing resolution
or tiling. Reports include precision, recall, F1, matched bounding-box IoU,
identity accuracy for labelled identities, and the detector's timing metadata.

The benchmark does not change production defaults or tune thresholds. First
collect enough varied, exhaustively labelled screenshots to make comparisons
meaningful.


## Existing screenshot sources

Visual review of the public
[ForTesting screenshot collection](https://github.com/kakadd172-lab/ForTesting/tree/kiomet-screenshots)
identified fifteen gameplay candidates across two capture sessions: eleven
from November 18 and four from November 19. The November 18 images are a
short consecutive sequence; the November 19 images are also closely spaced
observer captures, with some near-identical views. They provide temporal and
scene variation, not fifteen independent matches. Their exact source blob
SHAs, byte sizes, paths, and source links are recorded in
`source_inventory.json` so they can be verified without copying the images
into this repository prematurely.

The detector-debug image with coloured bounding-box overlays, the
landing/help-page captures, and the November 19 menu capture are excluded.
The fifteen candidates are not ground-truth labels: copy only reviewed,
appropriate images into this folder, then annotate every visible tower and
unit before setting `fully_annotated` to `true`.
