# Room scale scene: template and scoring sheet

The next scene matters more than SceneDiff: one room, filmed on a phone, where
the pipeline meets floor, walls and furniture instead of a table. Nothing here is
tuned yet; the two flags that are expected to matter are already in the config,
**both default OFF**.

## What to record

| item | requirement | why |
|---|---|---|
| framing | **landscape**, whole room in shot, camera roughly chest height | the failure mode we are testing is the plane choice, so the floor has to be visible |
| length | 20 to 45 seconds per video | long enough for 8 or more sharp frames at the default threshold |
| motion | slow pan or small step, no fast whips | fast motion drops frames and blurs the crops |
| light | unchanged between the two takes | the current pipeline flags lighting changes as objects |
| A / B | same viewpoint, same zoom, seconds apart | reconstruction assumes a shared static scene |
| changes | **one furniture move, one object added on the floor, one object removed** | three change types, each with a known object |

Put the files at `data/room1/a.mp4` and `data/room1/b.mp4`.

## Scoring sheet

Copy this before looking at any output, fill the left column only, then run
`detect.py` once and fill the right column.

| ground truth | type | object | found? | label correct? | notes |
|---|---|---|---|---|---|
| 1 | moved | | | | furniture move: in place, same room |
| 2 | added | | | | on the floor, so it sits on the plane the auto mode picks |
| 3 | removed | | | | |

Record in `data/room1/ground_truth.json` (same shape as the other scenes: `id`,
`type`, `object`, `synonyms`) and the per-entry resolution in
`data/room1/matches.json`, so the existing scripts score it with no changes.

## What to watch, and the two flags

`scripts/plane_check.py` prints the plane the auto mode picked: normal, how
vertical it is, coverage within 2 cm and the height above the lowest surface.
On the SceneDiff rooms that plane was a floor, a wall or a tilted surface with
coverage between 1% and 48%.

* `diff.auto_plane_rules` (default **OFF**): when the chosen plane has low
  coverage, or is floor-like or wall-like rather than table-like, fall back to
  the plane free `geometric` mode instead of building boxes on the wrong surface.
* `diff.relative_cells` (default **OFF**): size the change cells as a fraction of
  the scene or plane extent instead of the fixed 5 mm grid, so a room scale scene
  is not quantised into table-sized cells.

Both are evaluated on this scene before either is enabled: same gates as always,
no scene specific values, and the development scenes must not regress.

## Numbers to report for this scene

Same table as everywhere else: ground-truth changes found with the correct type,
labels correct, missed, extras, and the plane the auto mode chose. Then the same
run with `diff.auto_plane_rules` on and with `diff.relative_cells` on, so the
effect of each flag is one line of evidence rather than a claim.
