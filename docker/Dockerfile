# ---------- build stage ----------
FROM python:3.11-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Install CPU-only torch to keep image small (~1.2 GB vs ~5 GB with CUDA)
RUN uv venv .venv \
    && uv sync --no-dev --extra-index-url https://download.pytorch.org/whl/cpu

# ---------- runtime stage ----------
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Create logs directory
RUN mkdir -p logs

EXPOSE 8080

ENTRYPOINT ["python", "main.py"]
# Competition mode:
#   docker run -e ROOSTOO_API_KEY=... -e ROOSTOO_API_SECRET=... trading-bot --mode roostoo
# Paper mode:
#   docker run trading-bot
# Live (Binance) mode:
#   docker run -e BINANCE_API_KEY=... -e BINANCE_API_SECRET=... trading-bot --mode live
