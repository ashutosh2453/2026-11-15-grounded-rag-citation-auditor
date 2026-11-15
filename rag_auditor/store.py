"""SQLite persistence for documents and their chunks."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    source      TEXT,
    sha256      TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    UNIQUE (document_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    source: str | None
    sha256: str
    created_at: str
    chunk_count: int = 0


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    ord: int
    text: str
    document_title: str = ""


class DocumentStore:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)

    @staticmethod
    def fingerprint(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def find_by_hash(self, sha256: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count "
                "FROM documents d WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return _doc(row) if row else None

    def add_document(self, title: str, chunks: list[str], source: str | None, sha256: str) -> Document:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO documents (title, source, sha256, created_at) VALUES (?, ?, ?, ?)",
                (title, source, sha256, created),
            )
            doc_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO chunks (document_id, ord, text) VALUES (?, ?, ?)",
                [(doc_id, i, text) for i, text in enumerate(chunks)],
            )
        return Document(doc_id, title, source, sha256, created, len(chunks))

    def list_documents(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d "
                "LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.id"
            ).fetchall()
        return [_doc(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.document_id, c.ord, c.text, d.title FROM chunks c "
                "JOIN documents d ON d.id = c.document_id ORDER BY c.id"
            ).fetchall()
        return [Chunk(r["id"], r["document_id"], r["ord"], r["text"], r["title"]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _doc(row: sqlite3.Row) -> Document:
    return Document(row["id"], row["title"], row["source"], row["sha256"], row["created_at"], row["chunk_count"])
