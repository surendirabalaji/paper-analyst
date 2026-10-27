# Agentic AI Paper Analyst: Cloud Run image.
# Build:  docker build -t paper-analyst .        (Cloud Run needs linux/amd64; scripts/deploy.sh uses Cloud Build)
# Run:    docker run --rm -p 8080:8080 --env-file .env paper-analyst

FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ingest/chunking.py and OpenAIEmbeddings load the cl100k_base BPE at import time. Fetch it once here,
# so a cold start does not download it from the internet.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && ls /opt/tiktoken


FROM python:3.13-slim AS runtime

# Unbuffered logs; no .pyc writes at runtime. HOME points at a writable dir because the app user has none.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HOME=/tmp \
    CHROMA_PERSIST_DIR=/app/data/chroma_index \
    SQLITE_PATH=/app/data/papers.db

# Non-root user: fixed uid 10001, no home folder, no login shell.
RUN useradd --system --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Only the finished venv and the tokenizer cache come over: no pip cache, no build files.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken

# Code is root-owned and read-only for the app user; the app never writes here.
COPY app/ ./app/
COPY ingest/ ./ingest/
COPY data/papers.db ./data/papers.db

# The Chroma index is the one writable path: chromadb.PersistentClient opens chroma.sqlite3 read-write
# (migrations check, WAL/journal files), even for queries. Owning it avoids "attempt to write a readonly
# database" as uid 10001 without copying 105 MB into Cloud Run's in-memory /tmp on every cold start.
# Writes land in the container's ephemeral layer and never change the image or other instances.
COPY --chown=10001:10001 data/chroma_index/ ./data/chroma_index/

USER 10001:10001

# Documentation only; Cloud Run injects PORT.
EXPOSE 8080

# Shell form so ${PORT} expands; exec makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
