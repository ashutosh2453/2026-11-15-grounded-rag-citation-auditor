"""Sentence-aware chunking with overlap."""

from __future__ import annotations

import re

from .text import raw_tokens, split_sentences

_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Group sentences into chunks of at most ``max_words`` words.

    Paragraph boundaries (blank lines) always start a new chunk so unrelated
    sections never get blended. ``overlap_sentences`` trailing sentences of a
    chunk are repeated at the start of the next chunk in the same paragraph to
    preserve context across boundaries. A single sentence longer than
    ``max_words`` becomes its own chunk.
    """
    if max_words < 10:
        raise ValueError("max_words must be >= 10")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    text = text.replace("\r\n", "\n")
    # Markdown headings become standalone paragraphs so they never fuse with body sentences.
    text = _HEADING_RE.sub(lambda m: f"\n\n{m.group(1).strip()}\n\n", text)
    chunks: list[str] = []
    for paragraph in (p for p in re.split(r"\n\s*\n", text) if p.strip()):
        sentences = split_sentences(paragraph)
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            words = len(raw_tokens(sentence))
            if current and current_words + words > max_words:
                chunks.append(" ".join(current))
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_words = sum(len(raw_tokens(s)) for s in current)
                if current and current_words + words > max_words:
                    current, current_words = [], 0
            current.append(sentence)
            current_words += words
        if current:
            chunks.append(" ".join(current))
    return chunks
