FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Install dependencies first (better Docker layer cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application files
COPY main.py README.md ./
COPY audio/.gitkeep audio/.gitkeep
COPY transcription/.gitkeep transcription/.gitkeep
COPY processed/.gitkeep processed/.gitkeep

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "main.py"]
