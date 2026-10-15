# Branches and tags

**`main` is the submission.** Everything else is experiment or documentation
history, kept for the audit trail. No branch other than main is needed to run or
grade the project.

## Branches

| branch | what it contains | status |
|---|---|---|
| `main` | the submission. Detection frozen at v0.12.0, config v1.0.0, plus the v1.1.0 usability fallback and the docs | the one to read |
| `exp/accuracy` | the accuracy sprint: the scorer, the flag experiments (soft veto, coverage added, objmatch, fragment merge), the golden JSONs, the first SceneDiff adapter | nothing was accepted, all flags default OFF |
| `exp/semantics-v2` | the current naming round: `frames.adaptive_sharpness`, the plane check, the naming package (v3 ladder and v4 tiers), the labelled crop set, the room-scale template | active |
| `feat/adaptive-default` | the single change that became v1.1.0, merged into main | merged |
| `docs/humanize` | the prose pass over the summaries and docs | merged into main |
| `docs/challenge-fit` | the README top, the example section, the design note | merged into main |
| `docs/v1.0.2`, `docs/fix` | small documentation and viewer fixes on the way to v1.0.3 | merged into main |

## Tags

| tag | what it marks |
|---|---|
| `v0.12.0` | the frozen detection code, the version the held-out runs used |
| `v1.0.0` | the submission release: held-out scene5 and scene6 run once, digests recorded |
| `v1.0.1` | documentation and summary wording. Its summary regeneration touched the held-out summary text and was corrected by v1.0.3 |
| `v1.0.2`, `v1.0.3` | the correction: held-out summaries restored to v1.0.0, tests accept both formats, viewer fix for the swapped before and after crops |
| `v1.0.4` to `v1.0.6` | the challenge-fit documentation, the README figures, the SceneDiff limitation wording |
| `v1.1.0` | `frames.adaptive_sharpness: true` by default, a usability fallback that only fires when the absolute blur threshold leaves too few frames. Verified byte identical on demo, scene3 to scene6 |
| `v1.2.0` | this pass: language statistics, code map, DEEPDIVE.md, recruiter README, no behaviour change |
