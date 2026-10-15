"""Naming with abstention (flag `semantics.naming_v3`, default OFF).

The package is split by responsibility so each file can be read and tested on
its own:

  text.py         string cleaning and phrase extraction
  vocabulary.py   the closed vocabulary and its lookups
  descriptor.py   the "unidentified small dark object (#2)" wording
  ladder.py       the four rung decision, the heart of the feature
  rerank.py       SigLIP 2 scoring of crops against the vocabulary
  cache.py        reading cached crops, VLM answers and human scoring
  evaluation.py   coverage, precision, wrong name rate, abstention, LOSO

Typical use:

    from src.naming import load_vocabulary, decide, RerankModel

    vocab = load_vocabulary()
    model = RerankModel(vocab, device="cpu")
    views = model.score_crops(["crop_a.png", "crop_b.png"])
    print(decide("metal chair", views, vocab, index=1).label)

Debugging:

  * every module logs under the "naming.*" logger. Set
    `logging.basicConfig(level=logging.DEBUG)` to watch each rung decide.
  * `python -m src.naming.cache` prints the entry inventory with no models.
  * `python -m src.naming.vocabulary` prints vocabulary lookups.
  * `python scripts/naming_v3_eval.py --scenes demo` runs the whole thing on one
    scene with the numbers printed at the end.
"""
from src.naming.cache import EntryInput, load_all, load_scene
from src.naming.descriptor import (COLOR_WORDS, SIZE_WORDS, color_word, describe,
                                   index_of, is_descriptor, size_word)
from src.naming.evaluation import Outcomes, label_is_correct, leave_one_scene_out
from src.naming.ladder import Decision, decide
from src.naming.rerank import RerankModel
from src.naming.vocabulary import Vocabulary, load_vocabulary

__all__ = [
    "EntryInput", "load_all", "load_scene",
    "COLOR_WORDS", "SIZE_WORDS", "color_word", "describe", "index_of",
    "is_descriptor", "size_word",
    "Outcomes", "label_is_correct", "leave_one_scene_out",
    "Decision", "decide",
    "RerankModel",
    "Vocabulary", "load_vocabulary",
]
