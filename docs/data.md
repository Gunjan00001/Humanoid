# Data Collection

The pipeline needs two recordings of the **same small indoor area** taken at
different times, plus a documented ground truth.

## Recording guide (from `AGENT_PLAN.md` §6, Phase 0)

1. Pick a small area — desk, shelf, room corner.
2. Record **Video A** (30–60 s).
3. Make exactly three clear changes:
   - move one object,
   - remove one object,
   - add one object.
4. Record **Video B** with similar lighting and a similar camera path.
5. Optional: repeat with a second scene to show robustness.

Rules that make reconstruction work:

- slow pans, high frame overlap between views;
- textured surfaces in view (blank walls are hard);
- no heavy motion blur;
- consistent lighting between A and B;
- changes large and obvious relative to the frame.

Keep files small for GitHub; compress if needed.

## Files

```
data/
├── a.mp4
├── b.mp4
├── ground_truth.md        # required, see template below
└── sample_*.mp4           # optional small clips allowed in git
```

`.gitignore` ignores large videos (`data/*.mp4`, `data/*.mov`) with an exception
for `data/sample_*.mp4`. Use Git LFS or a release asset for the real recordings.

## `ground_truth.md` template

```markdown
# Ground truth — demo scene

Scene:            # e.g. "desk in room 214"
Recorded:         # date/time of A and B
Camera:           # phone model, orientation
Notes:            # lighting, anything unusual

| # | Change | Object | From → To (as seen in the scene) |
|---|--------|--------|----------------------------------|
| 1 | moved  |        |                                  |
| 2 | removed|        |                                  |
| 3 | added  |        |                                  |
```

## Checklist before starting the pipeline

- [ ] `data/a.mp4` and `data/b.mp4` recorded per the guide
- [ ] `data/ground_truth.md` filled with exactly three changes
- [ ] Both videos show the same area with good overlap
- [ ] Files are small enough for the repository (or handled via LFS/release)
