"""EXPERIMENTAL, not used by detect.py. Phase 4 object-first pipeline (branch
exp/object-first), wired behind `diff.method: objfirst`, default OFF.

Why this exists: the current detector finds changes first and names them second,
so a missed detection is lost. The object-first path turns the pipeline around:
inventory the objects in each video, segment and track them, lift them into the
shared 3D frame, match across videos, and only then call the differences.

Module map (one responsibility each, per docs/autopilot_batch2.md section 4):

  inventory.py   M1  per-video object inventory with the selected VLM
  segment.py     M2  SAM 3.1 masks (or SAM 2.1 automatic) + track ids
  lift.py        M3  lift instances into the shared frame (cached depth + poses)
  match.py       M4  DINOv3 patch matching + label/size/location prior, Hungarian
  verify.py      M5  visibility/coverage checks, VLM verification, tiers, fusion

Nothing here runs unless the config says `diff.method: objfirst`. The default
path stays byte-identical.
"""
from src.objfirst.match import embed_patches, match_instances, score_matrix  # noqa: F401

__all__ = ["embed_patches", "match_instances", "score_matrix"]
