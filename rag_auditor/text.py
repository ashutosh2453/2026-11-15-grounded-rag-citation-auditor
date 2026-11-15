"""Text utilities: tokenisation, light stemming, sentence splitting, number extraction.

Everything here is dependency-free so the retrieval and auditing layers stay
deterministic and easy to test.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as is are was were
    be been being this that these those it its into over under than so such can could
    should would will shall may might must do does did have has had i you he she we they
    them his her our their your my me us what which who whom whose when where why how all
    any both each few more most other some own same very just also about above below
    between through during before after up down out off again further once here there
    only too s t
    """.split()
)

NEGATIONS = frozenset({"not", "no", "never", "none", "nor", "cannot", "without", "neither", "nobody", "nothing"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['.][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:%)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_SUFFIXES = ("ization", "ational", "fulness", "iveness", "ations", "ation", "ments", "ment",
             "ings", "ing", "edly", "ies", "ied", "ed", "ly", "es", "s")


def raw_tokens(text: str) -> list[str]:
    """Lower-cased word tokens, including stopwords and negations."""
    text = text.lower().replace("n't", " not")
    return _TOKEN_RE.findall(text)


def stem(token: str) -> str:
    """A tiny, conservative suffix stripper (keeps stems >= 3 chars)."""
    if token.isdigit() or len(token) <= 3:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix in ("ies", "ied"):
                base += "y"
            return base
    return token


def content_tokens(text: str) -> list[str]:
    """Stemmed tokens with stopwords and negations removed (order preserved)."""
    return [stem(t) for t in raw_tokens(text) if t not in STOPWORDS and t not in NEGATIONS]


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in raw_tokens(text))


def numbers(text: str) -> set[str]:
    """Normalised numeric literals: '1,200' -> '1200', '3.50' -> '3.5', keeps '%'."""
    out: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        pct = match.endswith("%")
        value = match.rstrip("%").replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        out.add(value + ("%" if pct else ""))
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (O(len(a) * len(b)) with O(len(b)) memory)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]
