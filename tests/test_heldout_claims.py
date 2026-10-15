"""The held-out numbers in README/docs must match the committed reports."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "check_heldout_claims", ROOT / "scripts" / "check_heldout_claims.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_artifacts_match_expectations():
    assert mod.check_artifacts() == []


def test_prose_matches_artifacts():
    assert mod.check_prose() == []
