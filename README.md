# 3D Scene Change Detection

Detect what changed between two phone recordings of the same small indoor
space: build a shared 3D reconstruction, find **moved / added / removed**
objects, and describe each change with a semantic label and a sentence.

> **Status: work in progress.** No results yet. The full plan, phases, and
> checkpoints live in [`AGENT_PLAN.md`](AGENT_PLAN.md).

## Planned one-command interface

```bash
python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/
```

## Documentation

| File | Purpose |
| --- | --- |
| [`AGENT_PLAN.md`](AGENT_PLAN.md) | Project plan and source of truth |
| [`gpu.md`](gpu.md) | GPU / PBS cluster access reference |
| [`rules.md`](rules.md) | Working rules for agents on this project |
