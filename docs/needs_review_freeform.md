# Free-form borderline decisions (agent reviewed, pending user confirmation)

Threshold 0.72, margin 0.03; answers whose best match landed
within the margin of the threshold were counted as wrong and listed here for a human
decision. Each row is one crop answer.

| # | model | answer | ground truth | similarity detail | agent decision |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | headphones | laptop | nearest sugar bag at 0.912, lead 0.001 | counted wrong (below threshold) |

Total listed: 1; full list in logs/bakeoff_freeform_normalized.json.
