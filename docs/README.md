# Documentation Index

Start with the project plan: [`../AGENT_PLAN.md`](../AGENT_PLAN.md).

| Document | Read it for |
|---|---|
| [`pipeline.md`](pipeline.md) | Every stage, its inputs/outputs, where it runs, what it caches |
| [`verification.md`](verification.md) | The rule chain that turns 26 candidates into the final confirmed changes |
| [`debugging.md`](debugging.md) | The cache map, the inspector, and a playbook per symptom |
| [`setup.md`](setup.md) | Creating the environments and running stages (cluster + laptop) |
| [`models.md`](models.md) | Model stack, GPU/VRAM plan, download commands, license checklist |
| [`data.md`](data.md) | How to record the two videos and document ground truth |
| [`architecture.md`](architecture.md) | Repository layout, module responsibilities, config reference |
| [`results.md`](results.md) | The real results of the first end-to-end run |
| [`troubleshooting.md`](troubleshooting.md) | Known cluster and model failure modes and fixes |
| [`roadmap.md`](roadmap.md) | Phase-by-phase status checklist |
| [`../gpu.md`](../gpu.md) | Cluster/GPU access reference (PBS, CUDA, verified specs) |
| [`../rules.md`](../rules.md) | Working rules for agents on this project |

**Honesty rule:** no document may contain results, metrics, or screenshots that
did not come from a real run. Until a run happens, sections say so explicitly.
