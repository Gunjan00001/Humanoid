
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

## Which configuration a number came from

The SceneDiff results in this document, both the validation slice and the test
slice, were produced with the frozen **v1.0.0** configuration. They are not
restated under any later configuration.

`v1.1.0` adds one thing: `frames.adaptive_sharpness: true`, a fallback that only
runs when the absolute blur threshold leaves fewer than `min_frames` frames, and
warns when it fires. On every scene that already worked it produces byte identical
frames (verified for demo, scene3, scene4, scene5 and scene6), so it cannot change
a number produced under v1.0.0 on those scenes. It changes exactly one case,
scene2, which aborted at defaults and now runs.

For completeness, the numbers from the two settings on the SceneDiff validation
pairs, with the fallback OFF (frozen) and ON (post-hoc diagnostic):

| setting | pairs processed | px/im IoU | obj/im AP | obj/sc AP | TP | FP |
|---|---|---|---|---|---|---|
| frozen v1.0.0 | 7 of 20 | 0.0074 | 0.0002 | 0.0000 | 0 | 8 |
| adaptive fallback ON | 19 of 20 | 0.0578 | 0.0001 | 0.0000 | 0 | 27 |

The fallback makes the pairs processable and produces candidates, but no detection
reaches 0.5 IoU, so the score stays effectively zero. That is the honest summary:
the frame budget was an upstream blocker, not the reason for the low score.
