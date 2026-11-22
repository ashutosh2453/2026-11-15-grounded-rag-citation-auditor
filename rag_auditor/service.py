"""Application service that wires store, retriever, generator and auditor together."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .auditor import AuditReport, audit_answer
from .chunking import chunk_text
from .generation import ExtractiveGenerator, Generation, GenerationError, Generator
from .retrieval import HybridRetriever, ScoredChunk
from .store import Document, DocumentStore

log = logging.getLogger("rag_auditor")

MAX_DOCUMENT_CHARS = 2_000_000


class ValidationError(ValueError):
    pass


@dataclass
class AskResult:
    question: str
    answer: str
    model: str
    sources: list[ScoredChunk]
    audit: AuditReport
    latency_ms: float
    warnings: list[str]


class RAGService:
    def __init__(self, store: DocumentStore, generator: Generator | None = None,
                 retriever: HybridRetriever | None = None, chunk_words: int = 120, nli=None) -> None:
        self.store = store
        self.nli = nli  # optional rag_auditor.nli.NLIScorer
        self.generator = generator or ExtractiveGenerator()
        self.fallback = ExtractiveGenerator()
        self.retriever = retriever or HybridRetriever()
        self.chunk_words = chunk_words
        self.reindex()

    def reindex(self) -> None:
        self.retriever.build(self.store.all_chunks())

    def ingest(self, title: str, text: str, source: str | None = None) -> tuple[Document, bool]:
        """Chunk, persist and index a document. Returns ``(document, created)``."""
        title, text = title.strip(), text.strip()
        if not title:
            raise ValidationError("title must not be empty")
        if not text:
            raise ValidationError("document text must not be empty")
        if len(text) > MAX_DOCUMENT_CHARS:
            raise ValidationError(f"document exceeds {MAX_DOCUMENT_CHARS} characters")
        sha = self.store.fingerprint(text)
        existing = self.store.find_by_hash(sha)
        if existing:
            return existing, False
        chunks = chunk_text(text, max_words=self.chunk_words)
        doc = self.store.add_document(title, chunks, source, sha)
        self.reindex()
        log.info("ingested document id=%s chunks=%s", doc.id, len(chunks))
        return doc, True

    def delete(self, doc_id: int) -> bool:
        deleted = self.store.delete_document(doc_id)
        if deleted:
            self.reindex()
        return deleted

    def ask(self, question: str, top_k: int = 4) -> AskResult:
        question = question.strip()
        if not question:
            raise ValidationError("question must not be empty")
        if not 1 <= top_k <= 20:
            raise ValidationError("top_k must be between 1 and 20")
        started = time.perf_counter()
        contexts = self.retriever.search(question, top_k=top_k)
        warnings: list[str] = []
        try:
            generation: Generation = self.generator.generate(question, contexts)
        except GenerationError as exc:
            log.warning("generator failed, falling back to extractive: %s", exc)
            warnings.append(f"{exc}; used extractive fallback")
            generation = self.fallback.generate(question, contexts)
        report = audit_answer(generation.text, [c.chunk.text for c in contexts], nli=self.nli)
        if report.overall == "ungrounded":
            warnings.append("answer failed the groundedness audit - treat with caution")
        latency = (time.perf_counter() - started) * 1000
        return AskResult(question, generation.text, generation.model, contexts, report, round(latency, 2), warnings)

    def audit(self, answer: str, sources: list[str]) -> AuditReport:
        if not answer.strip():
            raise ValidationError("answer must not be empty")
        if not sources:
            raise ValidationError("at least one source is required")
        return audit_answer(answer, sources, nli=self.nli)
