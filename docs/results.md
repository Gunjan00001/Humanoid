# Results

> **No results yet.** This project has not run end-to-end. Everything below
> describes what will be recorded here, and the exact format it will use. No
> numbers, images, or claims will be added until they come from a real run.

## Planned contents

1. **GIF** — the 3D scene with red (removed), green (added), orange (moved)
   regions, rendered from `outputs/scene.ply`.
2. **Screenshots** — one per detected change, with the label and sentence.
3. **Detection table** — detected changes vs `data/ground_truth.md`.

   | # | Ground truth | Detected type | Label | Correct? | Confidence |
   |---|---|---|---|---|---|
   | 1 | moved: TODO | | | | |
   | 2 | removed: TODO | | | | |
   | 3 | added: TODO | | | | |

4. **Timings** — per-stage runtime and which stage ran where (CPU/GPU).
5. **Ablations** (optional, from the plan): occlusion-aware vs naive diff;
   Qwen3-VL-32B vs 8B; VGGT vs MASt3R.

## `outputs/report.json` schema (implemented; filled by real runs)

```json
{
  "meta": {
    "video_a": "data/a.mp4",
    "video_b": "data/b.mp4",
    "backend": "vggt",
    "model": "<resolved model path or id>",
    "created": "<ISO-8601 timestamp>",
    "config_hash": "<12-hex hash>",
    "num_frames": 60,
    "semantics": "vlm+dinov2 | not run",
    "confidence_note": "<how confidence is computed>"
  },
  "changes": [
    {
      "id": 1,
      "type": "moved | added | removed",
      "label": "<short object label>",
      "description": "<one-sentence description>",
      "centroid": [0.0, 0.0, 0.0],
      "bbox": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "from": [0.0, 0.0, 0.0],
      "to": [0.0, 0.0, 0.0],
      "confidence": 0.0,
      "cluster_ids": [1, 2],
      "evidence": {
        "frames_a": ["a/frame_00123.jpg"],
        "frames_b": ["b/frame_00045.jpg"],
        "confirmed_empty_in": [{"frame": 5, "file": "b/frame_00045.jpg", "n_valid": 180, "n_empty": 176}]
      }
    }
  ]
}
```

`confidence` is a documented heuristic (cluster size, visibility fraction,
DINOv2 similarity for moved objects), not a calibrated probability.
`from`/`to` are `null` for pure added/removed entries. `outputs/report.md`
presents the same information as a human-readable table plus the change
descriptions.

## How results get here

1. Finish Phases 0–5 of the plan and pass all checkpoints.
2. Run `python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/`.
3. Copy the real artifacts (GIF, screenshots, report files, timings) into this
   document and the README.
4. State limitations exactly as observed — including any misses or false
   positives — with the scene they were observed on.
