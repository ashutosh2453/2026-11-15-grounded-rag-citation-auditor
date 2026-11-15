"""Hybrid lexical retrieval: BM25 + TF-IDF cosine, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass

from .store import Chunk
from .text import content_tokens


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # fused RRF score
    bm25: float
    cosine: float


class HybridRetriever:
    """In-memory index rebuilt from the store whenever the corpus changes.

    BM25 rewards exact term matches with length normalisation; TF-IDF cosine
    rewards overall vocabulary overlap. Fusing their *ranks* (RRF) rather than
    raw scores avoids having to calibrate two incomparable score scales.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60) -> None:
        self.k1, self.b, self.rrf_k = k1, b, rrf_k
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._avgdl = 0.0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        tfs = [Counter(content_tokens(c.text)) for c in chunks]
        n = len(chunks)
        df: Counter[str] = Counter()
        for tf in tfs:
            df.update(tf.keys())
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        lengths = [sum(tf.values()) for tf in tfs]
        vectors = [self._vectorise(tf, idf) for tf in tfs]
        with self._lock:
            self._chunks, self._tfs, self._lengths = list(chunks), tfs, lengths
            self._idf, self._vectors = idf, vectors
            self._avgdl = (sum(lengths) / n) if n else 0.0

    @staticmethod
    def _vectorise(tf: Counter[str], idf: dict[str, float]) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _bm25(self, q_terms: list[str], i: int) -> float:
        tf, dl = self._tfs[i], self._lengths[i]
        score = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += self._idf[term] * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q_terms = content_tokens(query)
        with self._lock:
            if not q_terms or not self._chunks:
                return []
            q_vec = self._vectorise(Counter(q_terms), self._idf)
            bm25 = [self._bm25(q_terms, i) for i in range(len(self._chunks))]
            cos = [sum(w * vec.get(t, 0.0) for t, w in q_vec.items()) for vec in self._vectors]
            candidates = [i for i in range(len(self._chunks)) if bm25[i] > 0 or cos[i] > 0]
            fused: dict[int, float] = {i: 0.0 for i in candidates}
            for scores in (bm25, cos):
                ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
                for rank, i in enumerate(ranked, start=1):
                    fused[i] += 1.0 / (self.rrf_k + rank)
            order = sorted(candidates, key=lambda i: (fused[i], bm25[i]), reverse=True)[:top_k]
            return [ScoredChunk(self._chunks[i], round(fused[i], 6), round(bm25[i], 4), round(cos[i], 4)) for i in order]
