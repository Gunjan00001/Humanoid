"""The held-out numbers in README/docs must match the committed reports.

The held-out scenes were run once. This test fails if the documents drift from
outputs/<scene>/report.json, data/<scene>/ground_truth.json or the
human-resolved matches, which is exactly how the numbers got restated wrongly
once already.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_heldout_claims", ROOT / "scripts" / "check_heldout_claims.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class TestHeldoutClaims(unittest.TestCase):
    def test_artifacts_match_expectations(self):
        self.assertEqual(_mod.check_artifacts(), [])

    def test_prose_matches_artifacts(self):
        self.assertEqual(_mod.check_prose(), [])


if __name__ == "__main__":
    unittest.main()
