# syntax=docker/dockerfile:1.7
# Dockerfile multi-stage. Imagem unica que muda comportamento via $RUN_MODE.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: builder com toolchain para compilar wheels (psutil, pyarrow etc)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime enxuto
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    RUN_MODE=collector \
    DASHBOARD_PORT=8501 \
    DASHBOARD_HOST=0.0.0.0

WORKDIR /app

# curl serve apenas para HEALTHCHECK do dashboard
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Cria usuario nao-root
RUN useradd --create-home --shell /usr/sbin/nologin abrace
COPY app ./app
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data/raw /app/data/processed /app/data/models /app/reports/figures /app/reports/metrics \
    && chown -R abrace:abrace /app

USER abrace

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD test "$RUN_MODE" != "dashboard" || curl -fsS "http://localhost:${DASHBOARD_PORT}/_stcore/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
