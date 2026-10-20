"""The README first screen must not overclaim SceneDiff or model adoption.

Two specific regressions are guarded, because both happened:
  1. a SceneDiff score above the perfect-mask oracle ceiling (0.1147) reads as a
     system score when the system's own score is 0.0000;
  2. calling SAM 3.1 or DINOv3 "adopted" or the "primary segmenter" in the first
     screen, when the released detect.py path loads neither.
"""
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
ORACLE = 0.1147


def first_screen() -> str:
    text = README.read_text()
    cut = text.find("## Quick start")
    return text[:cut] if cut != -1 else text


def test_no_scenediff_score_above_oracle() -> None:
    for sentence in re.split(r"(?<=[.!?])\s+", first_screen()):
        if "SceneDiff" not in sentence:
            continue
        for number in re.findall(r"\b0\.\d+\b", sentence):
            assert float(number) <= ORACLE + 1e-9, (
                f"SceneDiff sentence states {number}, above the oracle {ORACLE}: {sentence!r}")


def test_no_adoption_claims_for_unwired_models() -> None:
    screen = first_screen()
    for model in ("SAM 3.1", "DINOv3"):
        for match in re.finditer(re.escape(model), screen):
            window = screen[max(0, match.start() - 220): match.end() + 220].lower()
            for phrase in ("adopted", "primary segmenter"):
                assert phrase not in window, (
                    f"{model!r} is described as {phrase!r} in the first screen: "
                    f"{window!r}")
