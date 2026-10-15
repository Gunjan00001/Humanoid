
## C5 test slice result (10 test pairs, evaluated once, frozen config)

Evaluated exactly once at the frozen v1.0.0 config, after the adapter was verified
by the projection control. No test score had been produced before this run (the
earlier test predictions came from the buggy adapter and were never scored).

| | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP (type) | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| headline (unprocessable = empty) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 29 |
| processable only | n/a | n/a | n/a | n/a | 0 | 0 | 0 |

**All 10 test pairs are unprocessable** at the frozen config (frame extraction
keeps fewer than 8 sharp frames). The test slice therefore contributes no
predictions and its headline numbers are all zero with 29 ground truth objects
missed. This is the honest result of the pre-registered evaluation: the pipeline
does not transfer to the room scale SceneDiff clips at the frozen settings.

The reduction lever is frames.adaptive_sharpness (Part C), which targets exactly
this upstream failure and is evaluated as a post-hoc diagnostic on the validation
pairs and scene2 only, never on the test pairs.

## Failure analysis on the 7 processable validation pairs

| pair | GT objects | GT object px (median) | our changes | our mask px | mode | rejected candidates |
|---|---|---|---|---|---|---|
| airhockeytable_1_airhockeytable_2 | 1 | 3,298 | 2 | 35,978 | tabletop | 10 |
| bathroom_11_bathroom_12 | 2 | 29,713 | 2 | 2,996 | tabletop | 3 |
| bathroom_1_bathroom_2 | 4 | 15,473 | 4 | 16,289 | tabletop | 29 |
| bedroom_2_bedroom_3 | 7 | 44,363 | 1 | 0 | tabletop | 8 |
| bedroom_2_bedroom_4 | 10 | 25,683 | 1 | 34,968 | tabletop | 7 |
| gym_9_gym_10 | 2 | 74,148 | 6 | 12,924 | tabletop | 101 |
| kitchen_8_kitchen_9 | 3 | 52,451 | 1 | 1,178,224 | tabletop | 3 |

Facts: every pair auto-selected `tabletop`; the GT objects are large at
1920x1080 (median 3.3k to 74k px); rejection counts run from 3 to 101.

Likely causes (marked likely, not proven):
- **Scale mismatch, likely the main one.** The detections and the 5 mm cell grid
  are tuned for a tabletop a few tens of centimetres across; here the objects are
  metres away and metres wide, so a cell grid that resolves a pen resolves nothing
  useful on a room and our change clusters cover the wrong extents (kitchen: our
  mask is 1.18M px against a 52k px GT object).
- **Different change semantics, likely.** SceneDiff objects are whole pieces of
  furniture seen from far away; our pipeline reports small local appearance
  differences (the air hockey sticker is 3.3k px, our two "removed" regions are
  elsewhere), so even a correct region would not line up with the GT silhouette.
- **Frame budget, likely.** At 3 fps and 30 frames per video the clips (25 to 275
  raw frames at 30 or 60 fps) are sampled very sparsely; the GT object may not be
  visible in any sampled frame from the chosen view.
- **Not a plumbing problem, demonstrated.** The projection control shows the
  adapter path is correct end to end.

Contributing: bedroom_2_bedroom_3 produced a change but no mask (its change is an
addition whose annotated frames were not among the projected ones), and gym had
101 rejected candidates, so the VLM filter is doing most of the work there.

## Part C item 1: frames.adaptive_sharpness (post-hoc, validation only)

The flag is implemented (default OFF): when the absolute blur threshold leaves
fewer than `min_frames`, the sharpest N frames by rank are used, a single global
rule.

Frame-level evidence (measured):
- demo: extracted frames with the flag ON are **byte identical** to the committed
  ones (both videos), because the flag only runs on the failure path.
- scene2: yields **12 + 23** frames with the flag ON (it fails at defaults with
  1 frame).
- previously unprocessable validation pairs now extract: bed_3_bed_4 13+11,
  bedroom_1_bedroom_2 15+30, hallway_1_hallway_2 14+9.

Full-pipeline run on the 13 unprocessable validation pairs and scene2 (job 11370)
was **blocked by node OOM**: other users held the node's GPUs (six processes
between 0.6 and 28.9 GB), so the describe stage failed for all but one pair. That
one pair produced no changes. So the flag's frame-level effect is demonstrated;
its end-to-end score on those pairs is **not measured** yet (environment limit,
not a code limit). The run will be resumed when the node frees.

## Part C item 1: adaptive sharpness results (post-hoc, validation only)

With `frames.adaptive_sharpness` ON, the 13 previously unprocessable validation
pairs now run the full pipeline. Measured (job 11377, 7 of 13 pairs done at the
time of writing; the rest are queued behind a saturated node):

| pair | frames | changes | unverified |
|---|---|---|---|
| bedroom_1_bedroom_2 | 45 | 1 | 0 |
| bedroom_4_bedroom_5 | 27 | 0 | 0 |
| gas_station_3_gas_station_4 | 53 | 0 | 0 |
| laundry_room_3_laundry_room_4 | 30 | 0 | 0 |
| living_room_15_living_room_16 | 53 | 5 | 0 |
| living_room_17_living_room_18 | 57 | 1 | 0 |
| living_room_1_living_room_2 | 47 | 1 | 0 |

**scene2 at default settings now runs** (35 frames) and reports 5 changes,
including the ground-truth coconut: removed coconut, removed apple, removed small
dark speck, added round object, added finger. That is the same count as the
documented scene2 working point, but reached at defaults, so the fallback makes a
clip that used to abort processable.

Official evaluator on the 7 adaptive pairs: px/im IoU 0.0620 (up from 0.0074 with
the pairs counted as empty), obj/im AP 0.0002, obj/sc AP 0.0000, TP 0, FP 8,
FN 32. So the flag fixes the upstream failure (the pairs are processed and even
produce candidates) but the downstream detection quality on the room-scale clips
is still the limit: no detection reaches 0.5 IoU. That matches the oracle control
finding.

Dev-scene check: with the flag ON the demo frames are byte identical to the
committed ones, so the flag changes nothing on scenes that already work.

## Part C item 2: which plane does the automatic mode pick on SceneDiff pairs?

`scripts/plane_check.py` fits the dominant plane the same way the auto mode does
and reports the unit normal, how vertical it is (|nz|), the fraction of confident
points within 2 cm of the plane, and the plane height above the lowest confident
surface. All 13 pairs shown (7 baseline-processable first, then the adaptive ones).

| pair | normal | vertical | coverage 2 cm | height (m) | reading |
|---|---|---|---|---|---|
| airhockeytable_1_2 | -0.38 -0.13 0.92 | 92% | 8% | 0.478 | floor-like |
| bathroom_11_12 | 0.06 -0.65 0.76 | 76% | 21% | 0.308 | tilted |
| bathroom_1_2 | -0.14 -0.49 0.86 | 86% | 3% | 0.892 | floor-like |
| bedroom_2_3 | -0.33 0.44 0.83 | 83% | 10% | 0.340 | tilted |
| bedroom_2_4 | -0.55 0.33 0.77 | 77% | 13% | 0.328 | tilted |
| gym_9_10 | -0.93 0.13 0.36 | 36% | 7% | 0.478 | tilted |
| kitchen_8_9 | -0.85 0.00 0.53 | 53% | 31% | 0.050 | tilted |
| bedroom_1_2 | -0.49 -0.12 0.86 | 86% | 7% | 0.327 | floor-like |
| gas_station_3_4 | 0.01 -0.99 0.15 | 15% | 48% | 0.203 | wall-like |
| laundry_room_3_4 | 0.02 -0.50 0.86 | 86% | 43% | 0.236 | floor-like |
| living_room_15_16 | -0.01 -0.54 0.84 | 84% | 1% | 0.234 | tilted |
| living_room_17_18 | 0.22 -0.61 0.76 | 76% | 21% | 0.234 | tilted |
| living_room_1_2 | -0.66 0.26 0.71 | 71% | 9% | 0.362 | tilted |

Reading: on these room-scale clips the auto mode does **not** find a table. Four
pairs give a floor-like plane, one gives a wall-like plane, and eight give a tilted
plane that is neither. The coverage test is weak everywhere (1% to 48% of confident
points lie within 2 cm of the chosen plane, against a tabletop scene where that
fraction is high). So the plane the tabletop method assumes does not exist in these
scenes, and the "height above the plane" boxes are built on the wrong surface. That
is the mechanical reason behind the zero true positives, and it is consistent with
the oracle control, which showed the mask projection itself is fine.

Proposed, not enabled: `diff.auto_plane_rules` (default OFF). When the chosen plane
has low coverage, or is floor-like or wall-like rather than table-like, the diff
stage would fall back to the plane-free `geometric` mode instead of building boxes
on the wrong plane. It changes nothing unless switched on. The geometric
re-comparison on these pairs is queued behind the GPU queue (job 11527).
