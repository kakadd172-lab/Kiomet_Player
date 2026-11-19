# Temporal tracking replay benchmark

This benchmark evaluates the association layer over a sequence of frames. It
does **not** call Ollama or run the vision detector: predictions are supplied
as recorded track snapshots, so the same replay can be evaluated repeatedly
while changing tracker code or thresholds.

## Why this needs linked annotations

Each real entity must keep the same ground-truth `id` across every frame in a
sequence, including frames where it is not detected. Labels should be reviewed
across the entire sequence, not assigned independently frame by frame. The
evaluator then compares each frame's spatial detections and measures whether
the predicted `track_id` stays consistent.

Do not infer ground-truth identity solely from the tracker's own output. Use
visible game evidence and manual review. If identity cannot be followed
confidently through an occlusion or scene transition, end the ground-truth
sequence or mark the entity as a new ID rather than guessing.

## Input format

Create a JSON file with a `frames` array. Every frame has a strictly increasing
integer `frame_index`, an `expected` array and a `predicted` array. Boxes use
normalized top-left `x`, `y`, `width`, and `height` coordinates.

```json
{
  "schema_version": 1,
  "sequence": "match_01_zoomed_region",
  "frames": [
    {
      "frame_index": 1,
      "expected": [
        {
          "id": "tower_A",
          "category": "tower",
          "bbox": {"x": 0.21, "y": 0.34, "width": 0.035, "height": 0.048}
        }
      ],
      "predicted": [
        {
          "track_id": "entity-000001",
          "category": "tower",
          "bbox": {"x": 0.212, "y": 0.341, "width": 0.035, "height": 0.048}
        }
      ]
    }
  ]
}
```

The coordinates and IDs above are illustrative only. Replace them with
measurements and reviewed track snapshots from the actual sequence. Use empty
arrays when a frame contains no visible entities or no predictions.

## Run

From the repository root:

```powershell
.\\.venv\\Scripts\\python.exe -m perception.tracking_benchmark --input benchmark_tracking/replays/match_01.json --output benchmark_results/tracking_match_01.json
```

The report includes detection precision/recall/F1, mean matched IoU, ID
switches, fragmentations, reacquisitions, and per-frame counts. Spatial
matching is category-aware and uses maximum-cardinality one-to-one assignment
at the selected IoU threshold.

## Metric interpretation

- **ID switch:** one ground-truth entity is associated with a different
  predicted track ID than its previous matched observation.
- **Fragmentation:** a previously matched entity becomes unmatched and later
  becomes matched again.
- **Reacquisition:** a fragmentation followed by a later match. It is reported
  separately to make temporary detection gaps visible.
- **Precision / recall / F1:** frame-level detection correspondence, not proof
  of persistent identity correctness.
- **Mean matched IoU:** spatial overlap for matched frame-level detections.

The current evaluator is a transparent diagnostic, not a full MOTChallenge
implementation. It does not compute HOTA, IDF1, or MOTA. It also does not
distinguish a true entity leaving the camera view from a detector miss; that
must be represented accurately in the ground-truth frame annotations.
