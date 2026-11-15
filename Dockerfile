FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DB_PATH=/data/rag.db
WORKDIR /app

COPY pyproject.toml README.md ./
COPY rag_auditor ./rag_auditor
RUN pip install --no-cache-dir . && useradd --create-home app && mkdir /data && chown app /data

USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rag_auditor.api:app", "--host", "0.0.0.0", "--port", "8000"]
