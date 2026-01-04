# Dockerfile for PolyB0T (optional deployment)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY polyb0t ./polyb0t
COPY tests ./tests

# Create non-root user
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POLYBOT_LOG_LEVEL=INFO

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["polyb0t", "run", "--paper"]

# Alternative commands:
# docker run polyb0t polyb0t api  # Run API server
# docker run polyb0t polyb0t report --today  # Generate report

