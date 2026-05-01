# ── Stage 1: Build environment ─────────────────────────────────
# We use a slim Python image — smaller attack surface, faster pulls
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install uv (our package manager)
RUN pip install uv --no-cache-dir

# Copy only dependency files first (Docker caching optimization)
# If pyproject.toml hasn't changed, Docker won't re-install dependencies
COPY pyproject.toml uv.lock* ./

# Install dependencies into the virtual environment
RUN uv sync --frozen --no-dev

# ── Stage 2: Runtime image ─────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .
COPY .env.example .env

# The fine-tuned model must be provided at runtime (it's too large for git)
# Docker volume or pre-built image will provide models/empathy-model/

# Set environment to use our virtual environment's Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1   
# PYTHONUNBUFFERED ensures print() and logging output appears immediately in Docker logs

# Expose the Flask port
EXPOSE 5000

# Health check — Docker will ping this every 30 seconds
# If it fails 3 times in a row, Docker marks the container unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application with Gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "60", "app:app"]
# Flask's built-in server is single-threaded and not suitable for production
CMD ["python", "app.py"]