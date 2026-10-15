"""SigLIP 2 rerank over cached crops.

Given a crop image, score it against the closed vocabulary with SigLIP 2 and
return the best canonical names. This is the second opinion that rung (a) of the
ladder requires before it prints an object name.

Design notes that matter for debugging:

  * the model is loaded once and reused; loading it costs more than scoring;
  * text embeddings are computed once per process, because they never change;
  * everything runs on CPU by default. Pass device="cuda" for a small GPU slice;
  * if the weights are missing the caller gets an empty candidate list and a
    loud warning rather than a crash, so the evaluation can still report the
    ladder's behaviour without the rerank.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("naming.rerank")

DEFAULT_MODEL_DIR = Path.home() / "models" / "siglip2-base"

# SigLIP text prompts. A single fixed template keeps the comparison fair across
# every object; "a photo of a <name>" is the template the model was tuned with.
PROMPT = "a photo of a {name}"


def _embedding(out):
    """Pull the embedding tensor out of whatever the model returned.

    Depending on the transformers version, get_text_features and
    get_image_features return either a plain tensor or a model output object
    holding the pooled embedding. Both shapes are accepted here so the code does
    not break on a version bump.
    """
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]        # CLS token as the fallback
    return out


class RerankModel:
    """Thin wrapper around SigLIP 2 for image-to-vocabulary scoring."""

    def __init__(self, vocabulary, model_dir: str | Path | None = None,
                 device: str = "cpu", batch_size: int = 16):
        self.vocabulary = vocabulary
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._names: list[str] = []
        self._text_features = None

    # -- setup -------------------------------------------------------------
    def available(self) -> bool:
        """True when the weights are on disk. Checked before loading so the
        caller can fall back cleanly instead of handling an exception."""
        ok = self.model_dir.exists() and any(self.model_dir.glob("*.safetensors"))
        if not ok:
            log.warning("SigLIP 2 weights not found at %s; the rerank will return no "
                        "candidates and the ladder will abstain", self.model_dir)
        return ok

    def load(self) -> bool:
        """Load the model and pre-compute the vocabulary text embeddings."""
        if self._model is not None:
            return True
        if not self.available():
            return False
        import torch
        from transformers import AutoModel, AutoProcessor

        log.info("loading SigLIP 2 from %s on %s", self.model_dir, self.device)
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
        self._model = AutoModel.from_pretrained(str(self.model_dir)).to(self.device).eval()
        self._names = sorted(self.vocabulary.canonical)      # deterministic order
        prompts = [PROMPT.format(name=n) for n in self._names]
        with torch.no_grad():
            inputs = self._processor(text=prompts, padding="max_length",
                                     return_tensors="pt").to(self.device)
            feats = _embedding(self._model.get_text_features(**inputs))
        self._text_features = torch.nn.functional.normalize(feats, dim=-1)
        log.info("text embeddings ready for %d vocabulary names", len(self._names))
        return True

    # -- scoring -----------------------------------------------------------
    def score_image(self, image_path: str | Path, top_k: int = 5
                    ) -> list[tuple[str, float]]:
        """Top-k (canonical name, cosine similarity) for one crop image.

        Returns an empty list when the model is unavailable or the image cannot
        be read, so callers never have to special-case failures.
        """
        if not self.load():
            return []
        from PIL import Image
        import torch

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:                     # unreadable crop
            log.warning("could not open crop %s (%s)", image_path, exc)
            return []

        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            feats = _embedding(self._model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats, dim=-1)
            sims = (feats @ self._text_features.T).squeeze(0)

        k = min(top_k, len(self._names))
        values, indices = torch.topk(sims, k)
        out = [(self._names[i], float(v)) for v, i in zip(values, indices)]
        log.debug("score_image(%s) -> %s", Path(image_path).name,
                  [(n, round(s, 3)) for n, s in out[:3]])
        return out

    def score_crops(self, image_paths: list[str | Path], top_k: int = 5
                    ) -> list[list[tuple[str, float]]]:
        """Score several crops that belong to the same entry. One list per crop,
        which is exactly what the ladder calls "views"."""
        views = [self.score_image(p, top_k=top_k) for p in image_paths]
        log.debug("score_crops: %d view(s) -> heads %s", len(views),
                  [v[0][0] if v else None for v in views])
        return views
