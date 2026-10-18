# Stack selection (batch 1, Phase 3)

Status: **probe results and decision rules done; bake offs pending** (they need
GPU slices; the plan below is what the next session runs, with the decision
rules already written so the choice cannot drift).

## Access probe (2026-10-17, tiny files via the Hugging Face API)

| repo | status | gated | license as probed | note |
|---|---|---|---|---|
| `facebook/sam3.1` | OK | manual (accepted) | see card at download | primary segmentation candidate |
| `facebook/sam3` | OK | manual (accepted) | see card at download | fallback segmentation candidate |
| `facebook/sam2.1-hiera-base-plus` | OK (already local) | no | apache-2.0 | segmentation fallback, installed |
| `Qwen/Qwen3.5-27B` | OK | no | per card at download | naming candidate (large) |
| `Qwen/Qwen3.5-9B` | OK | no | per card at download | naming candidate (mid) |
| `Qwen/Qwen3.5-4B` | OK | no | per card at download | naming candidate (light) |
| `Qwen/Qwen3.5-35B-A3B` | OK | no | per card at download | naming candidate (MoE) |
| `Qwen/Qwen3.6-35B-A3B` | OK | no | per card at download | naming candidate (newest MoE) |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | OK | manual (accepted) | see card at download | matching candidate;
| `depth-anything/DA3-GIANT` | OK | no | **cc-by-nc-4.0** | geometry candidate; allowed for research use |
| `depth-anything/DA3-BASE` | OK | no | apache-2.0 | geometry fallback |
| `yyfz233/Pi3X` | OK | no | **cc-by-nc-4.0** | geometry candidate; non commercial |
| `facebook/map-anything` | OK (already local) | no | apache-2.0 | geometry candidate, installed |
| `facebook/VGGT-1B` | OK (already local) | no | per card | geometry incumbent |

Disk: 322 T free on the shared filesystem, `~/models` currently 97 G. Weights
download to the login node only for the runners the bake offs need, smallest
first, `HF_HUB_OFFLINE=1` in compute jobs, no tokens in any file.

Two repo names in the earlier plan were wrong and are now resolved:
`ByteDance/Depth-Anything-3` → `depth-anything/DA3-GIANT` (`cc-by-nc-4.0`) and
`aiviet/Pi3X` → `yyfz233/Pi3X` (`cc-by-nc-4.0`). Pi3X and DA3 GIANT are
**non commercial** licences: fine for this research use, recorded in the
licence table when the README is refreshed.

## Decision rules (written before any run)

1. **Naming.** Adopt a VLM only if on the 387-crop labelled set it matches or
   beats the Qwen3-VL-32B baseline (exact and parent-category accuracy, Wilson
   intervals) at clearly lower memory, with the tiers still ordered
   plain > possibly > unidentified and none adopted alone. Latency and memory
   recorded per model.
2. **Segmentation.** Adopt SAM 3.1 only if the ground-truth objects of the dev
   scenes (comb, pen, clock, apple, scissors, cup, remote) are segmented in
   both videos with stable track IDs and fewer spurious instances than SAM 2.1
   base+.
3. **Matching.** Adopt DINOv3 ViT-L only if it stays at or above the DINOv2
   controls (apple pair CLS 0.704 / patch 0.914; scissors versus apple; cup and
   clock targets) on the same controls with masks from rule 2.
4. **Geometry.** Adopt a new backend only if, on the dev scenes (same frames,
   then 2x and 4x), the static alignment residual improves, point density on
   the GT objects increases, the known-size metric check passes, and runtime
   and memory fit one 80 GB GPU; the light profile must fit 24 GB.
5. **Licences.** A non commercial licence is acceptable for this research
   project and flagged; a licence that forbids our use is a true blocker.
6. **Conservative default.** Ties go to the incumbent (flag OFF, nothing
   merged); any gate that cannot be measured is recorded as a negative result,
   not a pass.

## Bake off plan (next session, small GPU slices)

a. **Naming:** Qwen3-VL-32B (incumbent baseline, cached) versus Qwen3.5-27B,
   9B, 4B, and the two 35B-A3B models on the 387-crop set and the dev scene
   crops; plain crop versus mask-tight crop with numbered marks; SAM 3.1
   concept naming with the closed vocabulary. Output per model: exact accuracy,
   parent accuracy with Wilson intervals, per-tier precision, latency, memory.
b. **Segmentation:** SAM 3.1 versus SAM 2.1 base+ on the dev scenes: both
   videos, GT objects segmented, stable track IDs, spurious instance count.
c. **Matching:** DINOv3 ViT-L versus DINOv2 on the recorded control pairs with
   masks from (b) (apple 0.704 CLS / 0.914 patch; scissors versus apple; cup
   0.374; clock 0.460).
d. **Geometry:** VGGT (incumbent) versus DA3 Giant 1.1 (nested metric), Pi3X
   and MapAnything on the dev scenes (same frames, then 2x and 4x): static
   alignment residual before and after ICP, point density on GT objects,
   metric scale against a known object size, runtime and memory.


## Bake-off (a): the Qwen3.5-4B attempts were a harness bug (see the correction in docs/negative_results.md)

One run over the 317 mapped crops (logs/bakeoff_naming_Qwen3.5-4B.json):
per-crop latency 0.44 s (p50 0.45, max 1.48), peak 8.9 GB, no exact or parent
measurement possible because 301 of 317 replies echo the prompt's placeholder
literal and 16 parsed names are wrong. These runs say nothing about Qwen3.5 (harness bug: the image tensors never
reached the model). The Qwen3-VL-32B baseline stays the naming reference until
a corrected run exists. Attempt-notes and the
next fix are in the bake-off log and docs/negative_results.md.
