# anaML Roadmap

The single source of truth for build order. The `/roadmap` skill reads this file to figure out what's done, what's next, and whether work is in progress.

## How this file works

Each step has a heading and an optional `done_when:` predicate. The predicate is what the skill uses to detect completion. Predicates support two signal types, combined with `AND` / `OR`:

- `commit:<regex>` — matches against commit subject lines (case-insensitive). Use when the step naturally results in a commit with a recognizable subject.
- `file:<path>:<regex>` — checks whether the file at `<path>` (in the working tree, committed or not) contains a substring/regex match. Use when the marker is a function name, import, config key, etc.

If a predicate matches the **working tree** but no matching commit exists yet, the skill reports the step as "done locally, not committed" — so you don't have to commit before getting an accurate read.

If a step has no `done_when:`, the skill falls back to loose subject matching against the step title.

To add or reorder phases: edit this file. The skill picks up changes automatically — no skill edits needed.

---

## Phase 1: Core API ✅

### Step 1 — BaseEngine ABC (load, predict, stream)
done_when: file:shared/inference/base.py:class\s+BaseEngine OR commit:base.?engine|abc

### Step 2 — JoblibModel backend
done_when: file:shared/inference/backends/joblib_backend.py:class\s+JoblibModel OR commit:joblib.?backend

### Step 3 — Registry (string → class)
done_when: file:shared/inference/registry.py:BACKENDS OR commit:registry

### Step 4 — Database models + Pydantic schemas
done_when: file:shared/models/models.py:class\s+Model OR commit:db.?models|schemas

### Step 5 — Config + async DB setup
done_when: file:services/api/core/database.py:async OR commit:config|database

### Step 6 — Health router
done_when: file:services/api/routers/health.py: OR commit:health.?router

### Step 7 — Model router (CRUD + file upload)
done_when: file:services/api/routers/models.py:UploadFile OR commit:model.?router|crud

### Step 8 — Inference router (predict + stats)
done_when: file:services/api/routers/inference.py:predict OR commit:inference.?router

### Step 9 — main.py (lifespan, CORS, routers)
done_when: file:services/api/main.py:lifespan OR commit:main\.py|app.?init

## Phase 2: Infrastructure & CI/CD ✅

### Step 10 — Dockerfile (multi-stage, non-root)
done_when: file:services/api/Dockerfile:USER OR commit:dockerfile

### Step 11 — docker-compose.yml (Postgres healthcheck)
done_when: file:docker-compose.yml:healthcheck OR commit:docker.?compose

### Step 12 — train_model.py
done_when: file:scripts/train_model.py: OR commit:train.?model

### Step 13 — Test suite (conftest + workflow test)
done_when: file:tests/conftest.py:fixture OR commit:conftest|test.?suite

### Step 14 — pyproject.toml (ruff + pytest)
done_when: file:pyproject.toml:ruff OR commit:pyproject|ruff

### Step 15 — requirements-dev.txt
done_when: file:requirements-dev.txt:pytest OR commit:requirements.?dev

### Step 16 — .github/workflows/ci.yml
done_when: file:.github/workflows/ci.yml:on: OR commit:ci\.yml|github.?action

### Step 17 — Push feat/cicd → dev, PR, CI green
done_when: commit:cicd.*merge|merge.*cicd

## Phase 3: ONNX Backend + Model Metadata

### Step 18 — Implement onnx_backend.py
done_when: file:shared/inference/backends/onnx_backend.py:onnxruntime OR commit:onnx.?backend

### Step 19 — Register ONNX in registry.py
done_when: file:shared/inference/registry.py:onnx OR commit:register.?onnx

### Step 20 — Add test_onnx_workflow to test suite
done_when: file:tests/test_api.py:onnx OR commit:test.*onnx

### Step 21 — Train + export an ONNX model for testing
done_when: file:scripts/train_model.py:onnx OR commit:export.?onnx

### Step 22 — Add metadata() abstract method to BaseEngine
done_when: file:shared/inference/base.py:def\s+metadata OR commit:metadata.?method

### Step 23 — Implement metadata() in OnnxModel
done_when: file:shared/inference/backends/onnx_backend.py:def\s+metadata OR commit:onnx.?metadata

### Step 24 — Implement metadata() in JoblibModel
done_when: file:shared/inference/backends/joblib_backend.py:def\s+metadata OR commit:joblib.?metadata

### Step 25 — Add metadata JSON column, populate at registration
done_when: file:shared/models/models.py:metadata OR commit:metadata.?column|metadata.?registration

### Step 26 — Add metadata assertions to both test workflows
done_when: file:tests/test_api.py:assert.*metadata OR commit:test.*metadata

### Step 27 — Migrate InferenceLog input/output from Text to JSON
done_when: file:shared/models/models.py:JSON OR commit:inference.?log.*json

## Phase 4: Redis + Caching

### Step 28 — Redis container in docker-compose
done_when: file:docker-compose.yml:redis OR commit:redis.*compose|compose.*redis

### Step 29 — Redis connection utility (core/redis.py)
done_when: file:shared/core/redis.py: OR file:services/api/core/redis.py: OR commit:redis.?client|redis.?util

### Step 30 — Replace module-level model cache with Redis
done_when: commit:redis.?cache|cache.*redis

### Step 31 — Cache model metadata (skip Postgres on every predict)
done_when: commit:metadata.?cache|cache.*metadata

### Step 32 — Add Redis health to /readyz
done_when: file:services/api/routers/health.py:redis OR commit:readyz.*redis|health.*redis

## Phase 5: Async Inference

### Step 33 — POST /predict/async → publish to Redis Streams
done_when: file:services/api/routers/inference.py:predict.?async OR commit:async.?predict|predict.?async

### Step 34 — GET /jobs/{id} → poll for result
done_when: file:services/api/routers/inference.py:jobs OR commit:jobs.?endpoint|get.?job

### Step 35 — Worker service (consumes stream, runs predict)
done_when: file:services/worker/main.py: OR commit:worker.?service

### Step 36 — Worker Dockerfile + add to docker-compose
done_when: file:services/worker/Dockerfile: OR commit:worker.?dockerfile

## Phase 6: Cleanup + Restructure

### Step 37 — Extract shared model resolution helper
done_when: file:shared/core/resolve.py:resolve_model OR commit:resolve.?model

### Step 38 — Restructure into services/api, services/worker, shared/
done_when: file:services/api/main.py: AND file:services/worker:.* AND file:shared/inference/base.py:

### Step 39 — Add async inference endpoint tests
done_when: file:tests/test_async.py:test_get_completed_job OR (file:tests/test_async.py:test_async_predict AND file:tests/test_async.py:nonexistent AND file:tests/test_async.py:pending)

## Phase 7: Go Gateway

### Step 40 — Basic Go HTTP reverse proxy to FastAPI
done_when: file:services/gateway/main.go:ReverseProxy OR commit:gateway.*proxy|reverse.?proxy

### Step 41 — Rate limiting (token bucket + Redis counters)
done_when: file:services/gateway:rate.?limit OR commit:rate.?limit

### Step 42 — Auth middleware (API key validation)
done_when: file:services/gateway:api.?key OR commit:auth.?middleware|api.?key.?auth

### Step 43 — A/B routing between model versions
done_when: commit:a.?b.?routing|ab.?test

### Step 44 — Structured JSON logging
done_when: commit:json.?log|structured.?log

### Step 45 — Gateway Dockerfile + add to docker-compose
done_when: file:services/gateway/Dockerfile: OR commit:gateway.?dockerfile

## Phase 8: Remote Backend + Streaming

### Step 46 — remote_backend.py (HTTP to vLLM/TGI, SSE)
done_when: file:shared/inference/backends/remote_backend.py: OR commit:remote.?backend

### Step 47 — SSE streaming endpoint
done_when: file:services/api/routers/inference.py:StreamingResponse OR commit:sse|streaming.?endpoint

### Step 48 — vLLM container in docker-compose
done_when: file:docker-compose.yml:vllm OR commit:vllm.*compose

### Step 49 — test_remote_workflow
done_when: file:tests/test_api.py:remote OR commit:test.*remote

## Phase 9: Polish

### Step 50 — S3 model storage
done_when: commit:s3|model.?storage

### Step 51 — K8s manifests
done_when: file:k8s:.* OR commit:k8s|kubernetes.?manifest

### Step 52 — Helm chart
done_when: file:helm:.* OR commit:helm

### Step 53 — Prometheus + Grafana
done_when: commit:prometheus|grafana

### Step 54 — README + architecture diagram + demo
done_when: file:README.md:architecture OR commit:readme.*final|architecture.?diagram
