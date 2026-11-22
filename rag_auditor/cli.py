"""Command-line interface: ``python -m rag_auditor.cli ingest|ask|audit``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generation import generator_from_env
from .nli import nli_from_env
from .service import RAGService, ValidationError
from .store import DocumentStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-auditor", description="Grounded RAG with claim-level citation auditing")
    parser.add_argument("--db", default="data/rag.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest .txt/.md files or directories")
    p_ing.add_argument("paths", nargs="+", type=Path)

    p_ask = sub.add_parser("ask", help="ask a question and audit the answer")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=4)
    p_ask.add_argument("--json", action="store_true", help="print the full JSON result")

    p_aud = sub.add_parser("audit", help="audit an answer against source files")
    p_aud.add_argument("answer")
    p_aud.add_argument("sources", nargs="+", type=Path)

    args = parser.parse_args(argv)
    service = RAGService(DocumentStore(args.db), generator_from_env(), nli=nli_from_env())
    try:
        if args.cmd == "ingest":
            files = [f for p in args.paths for f in (sorted(p.rglob("*")) if p.is_dir() else [p])
                     if f.suffix.lower() in {".txt", ".md", ".markdown"}]
            if not files:
                print("no .txt/.md files found", file=sys.stderr)
                return 1
            for f in files:
                doc, created = service.ingest(f.stem, f.read_text(encoding="utf-8"), str(f))
                print(f"{'added' if created else 'exists'}: #{doc.id} {doc.title} ({doc.chunk_count} chunks)")
        elif args.cmd == "ask":
            r = service.ask(args.question, args.top_k)
            if args.json:
                print(json.dumps({"answer": r.answer, "model": r.model, "audit": r.audit.to_dict()}, indent=2))
            else:
                print(f"\n{r.answer}\n")
                for i, s in enumerate(r.sources, start=1):
                    print(f"  [{i}] {s.chunk.document_title} (chunk {s.chunk.ord}, rrf={s.score})")
                a = r.audit
                print(f"\nAudit: {a.overall}  groundedness={a.groundedness}  citation_accuracy={a.citation_accuracy}")
                for c in a.claims:
                    print(f"  - {c.verdict:<20} {c.support_score:.2f}  {c.claim}")
                for w in r.warnings:
                    print(f"warning: {w}")
        else:
            report = service.audit(args.answer, [p.read_text(encoding="utf-8") for p in args.sources])
            print(json.dumps(report.to_dict(), indent=2))
    except (ValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        service.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
