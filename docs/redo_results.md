# Replay of all six uploaded scenes (2026-10-20)

One cluster job (11880, two GPUs, current main, config v1.1.0) re-ran demo and
scene2-scene6 into fresh folders `outputs/<scene>_redo` and `cache/<scene>_redo`.
Every run exited 0. The frozen outputs, including the held-out scene5 and scene6,
were not modified (verified with `git diff`). The derived artefacts are local and
gitignored by design: 86 files and about 87 MB of outputs, plus about 2.1 GB of
caches. Rebuild them with `qsub scripts/redo_scenes.pbs`.

This is a post-hoc replay, not part of the held-out result. The original scene5
and scene6 evaluations stand.

## Results as reported by the pipeline

| scene | time | entries |
|---|---|---|
| demo | 15.8 min | removed comb; added pen and device; replaced remote control |
| scene2 | 9.2 min | removed coconut; removed apple; removed small dark speck; added round object; added finger |
| scene3 | 22.6 min | removed apple; removed black stool; removed black circular objects; removed white object with orange tip; added apple |
| scene4 | 10.7 min | moved apple; removed scissors; added plastic bottle |
| scene5 (post-hoc) | 30.5 min | removed chair; removed chair leg (three entries); removed chair backrest; removed stacked bowls; added small brown object |
| scene6 (post-hoc) | 27.7 min | removed remote control; added electrical outlet |

## Comparison with the frozen run

| scene | frozen entries | redo entries | quick match frozen | quick match redo | clusters frozen -> redo |
|---|---|---|---|---|---|
| demo | 3 | 3 | 2/3, 1 extra | 2/3, 1 extra | n/a |
| scene2 | 2 | 5 | 1/1, 1 extra | 1/1, 4 extras | 8 -> 9 |
| scene3 | 4 | 5 | 0/2, 4 extras | 0/2, 5 extras | 29 -> 26 |
| scene4 | 3 | 3 | 3/3, 0 extras | 3/3, 0 extras | 8 -> 8 |
| scene5 | 5 | 7 | 1/2, 4 extras | 1/2, 2 extras | 58 -> 55 |
| scene6 | 2 | 2 | 1/3, 1 extra | 1/3, 1 extra | 35 -> 35 |

scene4 and scene6 reproduced identically. scene2, scene3 and scene5 differ at the
**cluster level**, so the difference is upstream of naming; whether the cause is
GPU nondeterminism across runs or a different working point needs a third run to
separate. The "quick match" column is an automatic type plus canonical-name
comparison, not the official human-resolved scoring; the two disagree where a
label is a paraphrase (demo is 3/3 under the official resolution).
