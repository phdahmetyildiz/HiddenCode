# === Evolution Simulator Docker Image ===
# Multi-stage build for smaller final image

# ---- Stage 1: Base Python image ----
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage 2: Dependencies ----
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 3: Application ----
FROM dependencies AS app

# Copy project source
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY main.py .

# Create output directory
RUN mkdir -p /app/runs

# Expose Streamlit default port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: launch Streamlit UI
# Override with: docker run ... python main.py --mode single --config config/default_config.json
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
