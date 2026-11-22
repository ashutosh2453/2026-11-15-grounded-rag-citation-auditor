"""FastAPI HTTP layer."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .generation import Generator, generator_from_env
from .nli import nli_from_env
from .service import RAGService, ValidationError
from .store import DocumentStore

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}


class DocumentIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)
    source: str | None = Field(None, max_length=500)


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=20)


class AuditIn(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(..., min_length=1, max_length=50)


def _doc_json(doc, created: bool | None = None) -> dict:
    out = {"id": doc.id, "title": doc.title, "source": doc.source, "chunks": doc.chunk_count,
           "created_at": doc.created_at}
    if created is not None:
        out["created"] = created
    return out


def create_app(db_path: str | None = None, generator: Generator | None = None) -> FastAPI:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = DocumentStore(db_path or os.getenv("RAG_DB_PATH", "data/rag.db"))
        app.state.service = RAGService(store, generator or generator_from_env(), nli=nli_from_env())
        yield
        store.close()

    app = FastAPI(title="Grounded RAG Citation Auditor", version=__version__, lifespan=lifespan)

    @app.exception_handler(ValidationError)
    async def _validation(_: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def svc(request: Request) -> RAGService:
        return request.app.state.service

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request):
        s = svc(request)
        return {"status": "ok", "version": __version__, "generator": s.generator.name,
                "indexed_chunks": s.retriever.size}

    @app.post("/documents", status_code=201)
    def add_document(body: DocumentIn, request: Request):
        doc, created = svc(request).ingest(body.title, body.text, body.source)
        return _doc_json(doc, created)

    @app.post("/documents/upload", status_code=201)
    async def upload_document(request: Request, file: UploadFile = File(...), title: str | None = Form(None)):
        name = file.filename or "upload.txt"
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(415, f"unsupported file type; allowed: {sorted(ALLOWED_SUFFIXES)}")
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file exceeds 2 MB limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "file must be UTF-8 encoded text") from None
        doc, created = svc(request).ingest(title or Path(name).stem, text, name)
        return _doc_json(doc, created)

    @app.get("/documents")
    def list_documents(request: Request):
        return [_doc_json(d) for d in svc(request).store.list_documents()]

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: int, request: Request):
        if not svc(request).delete(doc_id):
            raise HTTPException(404, "document not found")

    @app.post("/ask")
    def ask(body: AskIn, request: Request):
        r = svc(request).ask(body.question, body.top_k)
        return {
            "question": r.question,
            "answer": r.answer,
            "model": r.model,
            "latency_ms": r.latency_ms,
            "warnings": r.warnings,
            "sources": [
                {"n": i, "chunk_id": s.chunk.id, "document_id": s.chunk.document_id,
                 "document_title": s.chunk.document_title, "text": s.chunk.text,
                 "score": s.score, "bm25": s.bm25, "cosine": s.cosine}
                for i, s in enumerate(r.sources, start=1)
            ],
            "audit": r.audit.to_dict(),
        }

    @app.post("/audit")
    def audit(body: AuditIn, request: Request):
        return svc(request).audit(body.answer, body.sources).to_dict()

    return app


app = create_app()
