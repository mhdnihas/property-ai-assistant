# syntax=docker/dockerfile:1.7
# Property AI Assistant — production image (Hugging Face Spaces compatible: port 7860, uid 1000)

# ---------- builder: install deps into a virtualenv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user \
    PORT=7860 \
    EMBEDDING_CACHE_DIR=/home/user/app/.cache/fastembed

# Hugging Face Spaces runs containers as uid 1000
RUN useradd --create-home --uid 1000 user \
    && install -d -o user -g user /home/user/app
WORKDIR /home/user/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=user:user app ./app
COPY --chown=user:user pipeline ./pipeline
COPY --chown=user:user data/index ./data/index

USER user

# Bake the query-embedding model (~70 MB) into the image so cold starts don't download it.
RUN python -c "from app.config import get_settings as g; from app.rag.embeddings import Embedder as E; s=g(); E(s.embedding_model, s.embedding_cache_dir).embed_query('warm up')"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
