# OCI Anthropic Gateway Docker Image
# Python 3.12 slim image for minimal footprint
#
# If Docker Hub is slow, try using a mirror:
#   docker build --build-arg PYTHON_IMAGE=python:3.12-slim .

ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

LABEL maintainer="OCI Anthropic Gateway"
LABEL description="Translation layer for OCI GenAI models with Anthropic API compatibility"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for OCI SDK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies using Tsinghua mirror (faster in China)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy application code
COPY main.py .
COPY src/ ./src/
COPY web/ ./web/

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directory for debug dumps (if enabled)
RUN mkdir -p /app/debug_dumps

# Expose default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/debug/ || exit 1

# Run the application
ENTRYPOINT ["./entrypoint.sh"]
