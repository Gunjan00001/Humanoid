"""The repository must stay small enough to clone and review.

Every bulk artifact that goes in makes the clone slower for a reviewer who does
not care about it. These are the guards:

1. the whole tracked tree fits in a budget in megabytes;
2. no single file exceeds a per-file limit, except the short allowlist of
   genuinely heavy deliverables: the recordings and the scene meshes that the
   viewer and the challenge requirements link to;
3. the number of tracked files stays below a cap, because bulk per-pair output
   was the main offender before the slimming.

The limits are constants at the top, with the reason each one has that value.
New experiments write bulk output into a gitignored folder and commit only
compact summaries, which is what keeps this test green.

  python -m unittest tests.test_repo_budget
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Measured at 163.6 MB after the slimming in v1.2.2; 175 gives room for compact
# summaries and figures without letting bulk output back in.
TOTAL_BUDGET_MB = 175

# 6 MB is enough for any report, viewer or figure. The allowlist covers the
# recordings and scene meshes, which the viewer and the challenge requirements
# need, plus the two reference GIFs. Anything new that heavy needs a discussion.
FILE_LIMIT_MB = 6
ALLOW = (
    "^data/scene4/(a|b)\\.mp4$",          # the example scene the README links
    "^data/scene6/original/(a|b)\\.mp4$",  # originals kept for reproduction
    "^outputs/scene[0-9]*/scene_mesh(\\.ply|-small\\.ply)$",
    "^outputs/scene[0-9]*/scene[0-9]*\\.ply$",
    "^outputs/scene6/change\\.gif$",       # the held-out evidence the README shows
)

# 377 files before the cap of 420, mostly documentation and per-scene reports.
FILE_COUNT_LIMIT = 420


def tracked_files() -> list[tuple[str, int]]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    rows = []
    for f in out.stdout.split():
        path = ROOT / f
        rows.append((f, path.stat().st_size if path.exists() else 0))
    return rows


class TestRepoBudget(unittest.TestCase):
    def test_total_tracked_size(self):
        rows = tracked_files()
        mb = sum(s for _, s in rows) / 1_048_576
        self.assertLess(mb, TOTAL_BUDGET_MB,
                        f"tracked tree is {mb:.1f} MB, budget is {TOTAL_BUDGET_MB} MB. "
                        "Commit compact summaries, not bulk output.")

    def test_per_file_limit(self):
        limit = FILE_LIMIT_MB * 1_048_576
        offenders = [(f, s) for f, s in tracked_files()
                     if s > limit and not any(re.search(p, f) for p in ALLOW)]
        self.assertEqual(offenders, [], "files above the per-file limit (no allowlist "
                        f"match): {offenders}")

    def test_file_count(self):
        n = len(tracked_files())
        self.assertLess(n, FILE_COUNT_LIMIT,
                        f"{n} tracked files, cap is {FILE_COUNT_LIMIT}")


if __name__ == "__main__":
    unittest.main()
