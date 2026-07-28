# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir hatchling

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /app/wheels .


# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home appuser

# Copy wheels and install
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/*.whl && rm -rf /app/wheels

# Copy config
COPY config/ config/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
