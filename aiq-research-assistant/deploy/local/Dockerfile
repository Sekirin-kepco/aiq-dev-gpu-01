FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl gcc libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy minimal project files
COPY . /app

# Install pip and project dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt || true; fi
RUN pip install --no-cache-dir -e aira/src || true

EXPOSE 8080

CMD ["sh", "-c", "aira serve --host 0.0.0.0 --port 8080"]
