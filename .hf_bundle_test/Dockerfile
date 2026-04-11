# PromptOptEnv Dockerfile for Hugging Face Spaces
# Runs the OpenEnv FastAPI app (server.app:app), which also mounts the custom UI
# routes from web_ui. This keeps /reset, /step, /state, /health compatible.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy environment package
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install all project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable || uv sync --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable || uv sync --no-editable; \
    else \
        uv sync --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/env/.venv jinja2 python-dotenv httpx

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PORT=7860

# Health check on OpenEnv server endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch OpenEnv server app; UI is mounted by server/app.py
CMD ["sh", "-c", "cd /app/env && python -c \"import sys; sys.path.insert(0, '.'); from server.app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=7860)\""]
