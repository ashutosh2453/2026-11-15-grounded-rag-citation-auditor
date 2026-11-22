"""Optional natural-language-inference verifier for the claim auditor.

The lexical auditor is precise about numbers, negation and attribution, but it can't
recognise a faithful paraphrase ("refunds within a week" vs "returns within 7 days") and
it misses contradictions that don't use negation words ("slower" vs "faster"). An NLI
cross-encoder scores (evidence, claim) pairs as entailment / neutral / contradiction and
fills exactly those gaps.

Enable with RAG_NLI=1 (default model: cross-encoder/nli-deberta-v3-xsmall, about 280 MB,
CPU-friendly). Requires `pip install sentence-transformers`.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class NLIScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """For each (premise, hypothesis) pair return {'entailment', 'neutral', 'contradiction'} probabilities."""


class CrossEncoderNLI:
    def __init__(self, model: str = DEFAULT_NLI_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device)
        id2label = self.model.model.config.id2label
        self.index = {label.lower(): int(i) for i, label in id2label.items()}
        missing = {"entailment", "neutral", "contradiction"} - set(self.index)
        if missing:
            raise ValueError(f"model {model} is not a 3-way NLI model (missing labels {missing})")
        self.name = model

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False), dtype=np.float64)
        logits = logits.reshape(len(pairs), -1)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return [{k: float(p[i]) for k, i in self.index.items()} for p in probs]


def nli_from_env() -> NLIScorer | None:
    if os.getenv("RAG_NLI", "0") not in ("1", "true", "yes"):
        return None
    return CrossEncoderNLI(os.getenv("RAG_NLI_MODEL", DEFAULT_NLI_MODEL))
