# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies (needed for some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY omnidoc/ ./omnidoc/
COPY main.py .

# Create data directories
RUN mkdir -p data/raw data/processed data/evaluation

EXPOSE 8000

CMD ["uvicorn", "omnidoc.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
