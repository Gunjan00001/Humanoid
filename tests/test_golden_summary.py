"""Golden summaries: regenerating from the committed reports must be byte
identical to the fixtures checked in under tests/golden_summary/.

The fixtures exist so that refactoring src/summary.py (or anything it reads)
can prove it did not change the wording a reader sees. Record again only when a
wording change is deliberate, and say so in the commit message.

Held-out scenes are out of scope on purpose: their outputs are frozen at
v1.0.0 and must never be regenerated. See docs/rules.md and scripts/golden.py.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "golden", ROOT / "scripts" / "golden.py")
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


class TestGoldenSummaries(unittest.TestCase):
    def test_fixtures_recorded_for_the_development_scenes(self):
        for scene in golden.DEFAULT_SCENES:
            self.assertTrue(
                (golden.FIXTURES / scene / "summary.txt").exists(),
                f"no golden fixture for {scene}: run scripts/golden.py record")

    def test_regeneration_is_byte_identical(self):
        bad = []
        for scene in golden.DEFAULT_SCENES:
            fixture = golden.FIXTURES / scene / "summary.txt"
            current = golden.regenerate(scene)
            if current != fixture.read_text():
                bad.append(scene)
        self.assertEqual(bad, [],
                         "regeneration differs from the golden fixtures for: "
                         + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
