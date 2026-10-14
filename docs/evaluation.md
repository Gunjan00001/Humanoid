
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
