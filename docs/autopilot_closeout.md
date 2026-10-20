# Autopilot close-out (final batch)

Saved verbatim before execution, per the instruction. The anti-loop rules, rails
and HUMAN ONLY items of docs/autopilot_final.md still apply: prepush audit before
every push; no history rewrite; no force push; no tag moved; held-out outputs,
v1.0.0 and the archive tags untouched. Resume rule: if the budget ends, the exact
next step goes at the top of docs/autopilot_status.md and the work stops on a
clean commit.

## 1. Phase 4 gate evaluation (30 minutes, cached data only)
Run the object-first path (`diff.method: objfirst` with M5 fusion) on demo and
scene2 to scene6, score with the same resolutions as the baseline, and rerun to
show determinism. Report per scene and overall: type recall, localized recall,
strict move recall, extras, fragments, fused entries versus baseline, and the
regression floor on scene4 and demo (3 true changes, 0 extras). Apply the written
gates. Expected: not accepted; record real numbers either way.

## 2. Negative result, written accurately
docs/negative_results.md and docs/objfirst.md must state that M1 to M5 inherit
the detector's clusters and 3D boxes (M1 from cached crops and VLM labels, M3
from the diff boxes), so this is NOT a full test of the object-first hypothesis;
what the gates showed; the lift sanity check and the independent SAM 3.1
inventory (per-video VLM inventory, masks lifted from cached depth and poses,
per-frame centroid scatter check) are NOT built; and list them as the first
items of "what I would do next with a week". Do not claim object-first was
tested and failed.

## 3. Close the branch
Annotated tag `archive/object-first` on exp/object-first, push the tag (normal
push), delete the remote and local branch (normal delete, no force). Copy to main
as NEW files only: docs/objfirst.md, docs/stack_selection.md,
docs/negative_results.md and docs/debugging.md updates, scripts and data files
the docs cite, and src/objfirst/ marked "experimental, not used by detect.py".
Do NOT copy changes to the frozen detection files or configs/default.yaml.
`git diff v1.0.0 main` on the frozen files must stay as it is today, and
`diff.method` stays `tabletop`.

## 4. No more experiments
Skip the naming adoption (part word ban, tiers, numbering) and every other
pending experiment; list them as future work with their evidence in DEEPDIVE.md.

## 5. Finish (docs and checks)
a. README first screen, DEEPDIVE.md, docs/code_map.md, docs/branches.md,
   docs/models.md in sync with main: final numbers (dev, held out, SceneDiff,
   SceneDiff oracle ceiling), adopted and rejected models, the Phase 4 outcome
   from step 2, limitations, and what I would do next with a week.
b. Required acknowledgements and citations according to each licence (SAM 3
   publication acknowledgement; DINOv3, Qwen, VGGT CC BY NC 4.0, SceneDiff,
   HD-Epic, UNITE as related work), and state which parts are non commercial.
c. Run tests, link, budget, privacy and history tests, the prepush audit, ruff
   and the CI workflow locally; fresh clone smoke test from GitHub; clone size
   with and without --depth 1.
d. Tag the final release (next patch or minor version), push, verify on the
   remote with a fresh clone: one branch (main), tags present, zero audit hits,
   identities clean, frozen files and held-out outputs unchanged.

## 6. Final report
One message, also saved as docs/autopilot_report_final.md: per step done or failed
with evidence; before and after tables (dev, held out, SceneDiff, Phase 4 gate
results) clearly separated; accepted and rejected changes; models and licences;
git state with proof; everything unverified; and the HUMAN ONLY list.
