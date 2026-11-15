from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_auditor.api import create_app
from rag_auditor.generation import Generation, GenerationError, OllamaGenerator
from rag_auditor.service import RAGService
from rag_auditor.store import DocumentStore

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class FailingGenerator:
    name = "failing"

    def generate(self, question, contexts):
        raise GenerationError("backend down")


class HallucinatingGenerator:
    name = "hallucinator"

    def generate(self, question, contexts):
        return Generation("The warranty lasts 99 years and covers water damage [1].", self.name)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(db_path=str(tmp_path / "api.db"))) as c:
        for f in sorted(SAMPLES.glob("*.md")):
            assert c.post("/documents", json={"title": f.stem, "text": f.read_text(encoding="utf-8")}).status_code == 201
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] > 0


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Citation Auditor" in r.text


def test_duplicate_document_is_not_reinserted(client):
    text = (SAMPLES / "acme_shipping_faq.md").read_text(encoding="utf-8")
    body = client.post("/documents", json={"title": "again", "text": text}).json()
    assert body["created"] is False
    assert len(client.get("/documents").json()) == 2


def test_ask_returns_grounded_cited_answer(client):
    body = client.post("/ask", json={"question": "How many months does the warranty last?"}).json()
    assert "24 months" in body["answer"]
    assert body["audit"]["overall"] == "grounded"
    assert body["sources"][0]["document_title"] == "acme_warranty_policy"


def test_ask_unknown_topic_abstains(client):
    body = client.post("/ask", json={"question": "What colour is the CEO's car?"}).json()
    assert body["audit"]["overall"] == "abstained"
    assert body["sources"] == []


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 99}).status_code == 422


def test_audit_endpoint_flags_hallucination(client):
    body = client.post("/audit", json={
        "answer": "Express shipping costs 499 rupees [1].",
        "sources": ["Express shipping delivers within 1 business day in metro cities for a fee of 199 rupees."],
    }).json()
    assert body["claims"][0]["verdict"] == "unsupported"
    assert body["overall"] == "ungrounded"


def test_upload_and_delete(client):
    r = client.post("/documents/upload", files={"file": ("notes.md", b"Acme was founded in 2015 in Pune.", "text/markdown")})
    assert r.status_code == 201
    doc_id = r.json()["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_upload_rejects_bad_files(client):
    assert client.post("/documents/upload", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 415
    assert client.post("/documents/upload", files={"file": ("x.txt", b"\xff\xfe\x00bad", "text/plain")}).status_code == 400


def test_generator_failure_falls_back(tmp_path):
    svc = RAGService(DocumentStore(tmp_path / "f.db"), FailingGenerator())
    svc.ingest("w", (SAMPLES / "acme_warranty_policy.md").read_text(encoding="utf-8"))
    r = svc.ask("warranty months")
    assert r.model == "extractive-mmr" and any("fallback" in w for w in r.warnings)


def test_hallucinating_generator_is_caught(tmp_path):
    svc = RAGService(DocumentStore(tmp_path / "h.db"), HallucinatingGenerator())
    svc.ingest("w", (SAMPLES / "acme_warranty_policy.md").read_text(encoding="utf-8"))
    r = svc.ask("how long is the warranty")
    assert r.audit.overall == "ungrounded"
    assert any("groundedness" in w for w in r.warnings)


def test_ollama_generator_uses_http(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json)
        return httpx.Response(200, json={"response": "Answer [1]."}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    gen = OllamaGenerator(model="tiny", base_url="http://ollama:11434/")
    assert gen.generate("q", []).text == "Answer [1]."
    assert captured["url"] == "http://ollama:11434/api/generate" and captured["json"]["model"] == "tiny"


def test_ollama_generator_raises_on_connection_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(GenerationError):
        OllamaGenerator().generate("q", [])
