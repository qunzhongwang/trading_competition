# ---------- build stage ----------
FROM python:3.10-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Install CPU-only torch to keep image small (~1.2 GB vs ~5 GB with CUDA)
RUN uv venv .venv \
    && uv sync --no-dev --extra-index-url https://download.pytorch.org/whl/cpu

# ---------- runtime stage ----------
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["python", "main.py"]
# Override mode or config at runtime:
#   docker run trading-bot --mode live --config config/my_config.yaml
