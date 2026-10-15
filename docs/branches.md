# Repository history and archive tags

**`main` is the submission and the only branch.** Since v1.2.1 all work goes
directly to main: small commits, tested, with a tag only for a verified release (a
patch tag for documentation and fixes, a minor tag for a behaviour change, always
checked from a fresh clone).

## Where the old branches went

Before that rule, work happened on short branches. Their history is preserved in
**archive tags**, so nothing is lost and nothing left dangling:

| archive tag | what it holds |
|---|---|
| `archive/exp-accuracy` | the accuracy sprint: the scorer, the flag experiments (soft veto, coverage added, objmatch, fragment merge), the ground-truth and matches JSONs, the first SceneDiff adapter |
| `archive/exp-semantics-v2` | the naming round: the adaptive sharpness flag, the plane check, the naming package (v3 ladder, v4 tiers), the labelled crop set, the room-scale template |

The corresponding remote branches are deleted. Their local copies are kept until
the archive tags were verified on the remote. If you need something from them:

```
git tag -l 'archive/*'                 # list what exists
git show archive/exp-accuracy:docs/accuracy.md
git checkout archive/exp-semantics-v2 -- src/naming   # restore a file
```

## Tags

| tag | what it marks |
|---|---|
| `v0.12.0` | the frozen detection code, the version the held-out runs used |
| `v1.0.0` | the submission release: held-out scene5 and scene6 run once, digests recorded |
| `v1.0.1` | documentation and summary wording. Its summary regeneration touched the held-out summary text and was corrected by v1.0.3 |
| `v1.0.2`, `v1.0.3` | the correction: held-out summaries restored to v1.0.0, tests accept both formats, viewer fix for the swapped before and after crops |
| `v1.0.4` to `v1.0.6` | the challenge-fit documentation, the README figures, the SceneDiff limitation wording |
| `v1.1.0` | `frames.adaptive_sharpness: true` by default, a usability fallback that only fires when the absolute blur threshold leaves too few frames. Verified byte identical on demo, scene3 to scene6 |
| `v1.2.0` | the recruiter pass: language statistics, code map, DEEPDIVE.md, README reviewer path. No behaviour change |
| `v1.2.1` | consolidation: archive tags, evidence tooling brought onto main, link test. No behaviour change |
