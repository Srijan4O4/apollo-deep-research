# ============================================================
# Stage 1: Builder — install dependencies
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# ============================================================
# Stage 2: Runtime — lean production image
# ============================================================
FROM python:3.12-slim AS runtime

LABEL maintainer="Apollo Team"
LABEL description="Apollo Deep Research Agent — API & Streamlit frontend"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY frontend/ ./frontend/

RUN useradd --create-home --shell /bin/bash apollo && \
    chown -R apollo:apollo /app

USER apollo

EXPOSE 8000
EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
