FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt /app/
COPY ingest/requirements.txt /app/ingest-requirements.txt
RUN python -m pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt -r /app/ingest-requirements.txt

COPY app /app

# Ensure git is available so pip can install directly from the GitHub repo
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install `aiq_aira` with --no-deps to avoid dependency resolution issues
# This installs only the package itself; runtime dependencies are handled separately
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir --no-deps \
        git+https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant.git@main#egg=aiq_aira

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "server:app", "--workers", "2"]
