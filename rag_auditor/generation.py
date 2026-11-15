"""Answer generators.

* ``ExtractiveGenerator`` - zero-cost, offline: selects the most relevant,
  non-redundant source sentences with Maximal Marginal Relevance and cites them.
* ``OllamaGenerator`` - free local LLM via an Ollama server; prompted to cite
  sources as ``[n]`` so the auditor can verify every claim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from .retrieval import ScoredChunk
from .text import content_tokens, split_sentences

ABSTAIN_MESSAGE = "I could not find information about this in the indexed documents."


@dataclass(frozen=True)
class Generation:
    text: str
    model: str


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation: ...


def _is_heading(sentence: str) -> bool:
    """Short fragments without terminal punctuation are titles, not answerable statements."""
    return sentence.rstrip()[-1:] not in ".!?" and len(sentence.split()) < 8


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class ExtractiveGenerator:
    name = "extractive-mmr"

    def __init__(self, max_sentences: int = 3, diversity: float = 0.3, min_relevance: float = 0.15) -> None:
        self.max_sentences = max_sentences
        self.diversity = diversity
        self.min_relevance = min_relevance

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation:
        q = set(content_tokens(question))
        candidates: list[tuple[float, int, str, set[str]]] = []
        seen: set[str] = set()
        for idx, ctx in enumerate(contexts, start=1):
            for sentence in split_sentences(ctx.chunk.text):
                key = sentence.lower()
                toks = set(content_tokens(sentence))
                if key in seen or not toks or _is_heading(sentence):
                    continue
                seen.add(key)
                relevance = len(q & toks) / len(q) if q else 0.0
                if relevance >= self.min_relevance:
                    candidates.append((relevance, idx, sentence, toks))

        selected: list[tuple[float, int, str, set[str]]] = []
        while candidates and len(selected) < self.max_sentences:
            def mmr(c: tuple[float, int, str, set[str]]) -> float:
                redundancy = max((_jaccard(c[3], s[3]) for s in selected), default=0.0)
                return (1 - self.diversity) * c[0] - self.diversity * redundancy

            best = max(candidates, key=mmr)
            candidates.remove(best)
            selected.append(best)

        if not selected:
            return Generation(ABSTAIN_MESSAGE, self.name)
        parts = []
        for _, idx, sentence, _ in selected:
            body = sentence.rstrip()
            end = body[-1] if body[-1] in ".!?" else "."
            body = body.rstrip(".!?")
            parts.append(f"{body} [{idx}]{end}")
        return Generation(" ".join(parts), self.name)


PROMPT_TEMPLATE = """You are a careful assistant. Answer the question using ONLY the numbered sources.
After every sentence, cite the supporting source numbers in square brackets, e.g. [1] or [2][3].
If the sources do not contain the answer, reply exactly: "{abstain}"

Sources:
{sources}

Question: {question}
Answer:"""


class OllamaGenerator:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def build_prompt(self, question: str, contexts: list[ScoredChunk]) -> str:
        sources = "\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(contexts, start=1))
        return PROMPT_TEMPLATE.format(abstain=ABSTAIN_MESSAGE, sources=sources or "(none)", question=question)

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation:
        payload = {"model": self.model, "prompt": self.build_prompt(question, contexts), "stream": False,
                   "options": {"temperature": 0}}
        try:
            resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        if not text:
            raise GenerationError("Ollama returned an empty response")
        return Generation(text, self.name)


def generator_from_env() -> Generator:
    backend = os.getenv("RAG_GENERATOR", "extractive").lower()
    if backend == "ollama":
        return OllamaGenerator(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if backend != "extractive":
        raise ValueError(f"Unknown RAG_GENERATOR '{backend}' (expected 'extractive' or 'ollama')")
    return ExtractiveGenerator()
