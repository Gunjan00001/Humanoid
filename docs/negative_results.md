# Negative results

Real outcomes that did not work. Kept so they are not retried blindly. All
numbers are measured.

## Accuracy flags D, E and F: no change on any scene

Measured on branch exp/accuracy with the semantics worker (cached VLM, so no
3.5 minute model reload per run). Six development scenes each. Baseline and every
flag variant are identical:

| experiment | recall(type) | localized | strict move | name | extras | fragments |
|---|---|---|---|---|---|---|
| baseline | 71% | 86% | 25% (1/4) | 60% | 4 | 3 |
| soft_veto | 71% | 86% | 25% | 60% | 4 | 3 |
| coverage_added | 71% | 86% | 25% | 60% | 4 | 3 |
| both | 71% | 86% | 25% | 60% | 4 | 3 |
| objmatch (re-match) | 71% | 86% | 25% | 60% | 4 | 3 |
| fragment_merge | 71% | 86% | 25% | 60% | 4 | 3 |

- soft_veto changes no confirmed metric. It only moves VLM-denied candidates to
  unverified (scene6 went from 0 to 16 unverified, including the missed apple as
  "blue cup and red apple").
- coverage_added had no effect on any scene.
- objmatch (Hungarian re-match of the existing candidate crops) found no new
  move: the departure and arrival crops do not reach 0.55 similarity.
- fragment_merge (0.35 m global radius) did not join scene5's leg fragments or
  scene3's apple pair.

Under the gates nothing is accepted, so every flag stays OFF. The move misses
(demo remote, scene5 clock, scene6 cup) are not recoverable by re-weighting the
existing candidate crops.

## Track B probe: SAM 2 masking does not rescue the move match

`scripts/probe_sam2.py`, job 11017, one GPU. It refines the existing paired
crops with a SAM 2.1 base-plus mask (point prompt at the crop centre) and
compares the masked crops with DINOv2. The match gate is 0.65.

| scene | object | paired-crop DINOv2 after SAM 2 mask | pass (>= 0.65) |
|---|---|---|---|
| scene6 | cup | 0.190 | no |
| scene5 | clock | 0.219 | no |

Both are far below the gate, so the probe fails. Segmentation itself is not the
problem: SAM 2 needs no training and produces the masks. The problem is that the
candidate departure and arrival crops are not the same object, or the object
looks too different between the two views, so masking does not raise the
similarity. Track B stops here per the written pass gate. objmatch2 was not built.

## Track B controls (job 11023): the gate is calibrated, targets still fail

| scene | pair | role | CLS | patch | size_r |
|---|---|---|---|---|---|
| scene4 | apple moved | positive | 0.704 | 0.914 | 0.70 |
| scene4 | scissor vs apple | negative | 0.285 | 0.085 | 0.78 |
| scene6 | cup | target | 0.482 | 0.374 | 0.51 |
| scene5 | clock | target | 0.174 | 0.460 | 0.71 |

The positive control (a correct match) scores high and the negative control
(different objects) scores low, so the 0.65 gate is not miscalibrated. The targets
stay low, and the clock's CLS is below the negative control. Track B is closed as
a real negative result. objmatch2 was not built.

### Limits of the Track B result

The controls are thin: one positive and one negative only. On patch similarity
the two targets fell in the middle zone, between the negative control (0.085) and
the positive control (0.914): cup 0.374, clock 0.460. So the result means the cup
and clock candidates **did not match under this descriptor and this gate**, not
that they cannot be matched. A different crop choice (a different departure or
arrival cluster), a different descriptor, or a different gate could change it.
The clock is the more worrying case: its CLS similarity (0.174) is below the
negative control, which suggests the candidate pair is not the same object. Track
B is closed here and is not reopened.

## Stack selection: Qwen3.5-4B naming bake-off, attempt failed twice (2026-10-17)

Per the failure rule, two distinct causes before the run produced numbers; both
fixed or queued as described in docs/autopilot_status.md:

1. `from_pretrained("Qwen/Qwen3.5-4B")` pointed at the hub cache while the
   weights were a `local_dir`; in offline mode this resolved to an incomplete
   snapshot. Fixed by passing the local weights directory.
2. After wiring the processor's chat template with images, `generate()` fails
   for every crop with "the following model_kwargs are not used by the model:
   ['mm_token_type_ids', ...]" on 317 crops (peak mem 8.5 GB recorded, so the
   model itself loads and runs). Diagnosis incomplete this round: the reply
   truncation in the script (80 chars) hid the remainder of the message, and
   the processor emits input keys the checkpoint's generate rejects, which is
   an integration mismatch between the installed transformers and the
   checkpoint revision rather than a scorer problem.

Not adopted: the gate needs numbers and this attempt produced none; Qwen3-VL-32B stays the baseline for the naming decision until the
integration mismatch is fixed (drop the unused model_kwargs / align the
transformers revision), exactly as the conservative rule requires.

### Naming bake-off attempt 3 (Qwen3.5-4B): runs, then fails on placeholder echo (2026-10-17)

With the processor chat-template flow plus `enable_thinking=False` and a
tighter prompt, the model runs end to end: 317 crops, per-crop latency
0.44 s (p50 0.45, max 1.48), peak 8.9 GB. Sixteen answers parsed and were
wrong; 301 answers echo the literal placeholder (`{"name": "text"}`), so no
exact or parent score is measurable. Diagnosis queued for the next attempt
(docs/autopilot_status.md): drop the placeholder example from the prompt (the
model parrots the example key), and add the non-blank crop variant: the
mask-tight set blanks everything outside the mask to black, which at 4B scale
may be the harder input; the dev-scene crops under
`cache/<scene>/semantics/crops` are plain rectangles and are the same data the
32B baseline verified. No adoption decision: the gate needs exact and parent
numbers, and this attempt produced none.
