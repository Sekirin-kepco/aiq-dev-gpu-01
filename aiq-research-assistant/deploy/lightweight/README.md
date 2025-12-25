Lightweight smoke test for AI-Q Research Assistant

Purpose
- Provide a minimal, local smoke-test deployment that simulates the AI-Q backend with a tiny HTTP service.
- This is intentionally lightweight (no heavy LLM or GPU dependency). It's meant to verify Docker/Compose flow and API wiring on the EC2 host.

What is included
- `docker-compose.yml` : starts a single `aira-lite` service
- `Dockerfile` : builds a tiny Flask-based server
- `app/` : simple health and `/query` endpoints that return canned responses (placeholder for a real LLM)

How to run (on EC2)
1. cd into this directory

```bash
cd ~/aiq/aiq-research-assistant/deploy/lightweight
docker compose build
docker compose up -d
```

2. Smoke tests

```bash
# health
curl -s http://localhost:8080/health

# sample query
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' -d '{"q":"What is AI-Q?"}'
```

Notes
- This service is a placeholder. After smoke testing, replace the app implementation with real backend code (the `aira` package) and point the compose to the proper images or build context.
- If you want GPU-backed inference later, update `docker-compose.yml` with `deploy: resources` or use `--gpus` and a GPU-capable base image and model service.
