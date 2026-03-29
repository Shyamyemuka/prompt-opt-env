# TECH_STACK.md — Technology Stack
# Project: PromptOptEnv
# Cross-reference: BACKEND_STRUCTURE.md, IMPLEMENTATION_PLAN.md

---

## 1. Core Framework

| Technology | Exact Version | Purpose | Install command |
|---|---|---|---|
| openenv-core | 0.2.1 (latest as of March 2026) | Base classes: Environment, EnvClient, create_app | `pip install openenv-core` |
| Python | 3.11.x (3.11.9 recommended) | Runtime language | System install or pyenv |
| uv | 0.4.x (latest) | Fast package manager, replaces pip in dev | `pip install uv` |

Why Python 3.11 specifically: OpenEnv's pyproject.toml requires `python >= "3.10"`. 3.11 gives the best performance/compatibility balance. 3.12 has some async edge cases with FastAPI still being ironed out.

---

## 2. Web Server

| Technology | Exact Version | Purpose |
|---|---|---|
| FastAPI | 0.115.0 | HTTP + WebSocket server (required by OpenEnv) |
| Uvicorn | 0.30.6 | ASGI server that runs FastAPI |
| Pydantic | 2.7.0 | Data validation for Action/Observation models |
| websockets | 12.0 | WebSocket protocol library (used internally by EnvClient) |
| httpx | 0.27.0 | Async HTTP client for HF Inference API calls |

These versions are pinned in `pyproject.toml`. FastAPI 0.115 is required because OpenEnv uses the WebSocket lifespan pattern introduced there.

---

## 3. ML / NLP Libraries

| Technology | Exact Version | Purpose |
|---|---|---|
| rouge-score | 0.1.2 | ROUGE-L computation for grader |
| requests | 2.31.0 | Synchronous fallback for HF API calls |
| numpy | 1.26.4 | Numeric operations for reward clipping |

Why rouge-score 0.1.2: This is the Google-maintained package, not `rouge`. It supports `use_stemmer=True` which improves recall on varied phrasings. It is lightweight and has no heavy dependencies.

No HuggingFace transformers installed locally — all LLM inference goes through HF Inference API (serverless). This keeps the Docker image small (~300MB vs ~8GB with transformers + model weights).

---

## 4. Testing

| Technology | Exact Version | Purpose |
|---|---|---|
| pytest | 8.2.0 | Test runner |
| pytest-asyncio | 0.23.6 | Async test support for WebSocket tests |
| pytest-cov | 5.0.0 | Coverage reporting |

---

## 5. Containerisation

| Technology | Exact Version | Purpose |
|---|---|---|
| Docker | 25.x (host machine) | Container runtime |
| openenv-base image | latest | Base image provided by Meta/OpenEnv (includes Python 3.11, uv, FastAPI, uvicorn) |

The `openenv-base` image is pulled automatically when you run `openenv build`. You do NOT install Python or uv manually in the Dockerfile — the base image has them.

```dockerfile
# server/Dockerfile — exact base image line
FROM openenv-base:latest AS builder
```

---

## 6. Deployment

| Technology | Version/Detail | Purpose |
|---|---|---|
| Hugging Face Spaces | Docker Spaces (free tier) | Hosts the environment server |
| HF Inference API | Serverless (free, rate-limited) | Calls Mistral-7B for LLM output |
| openenv CLI | 0.2.1 | `openenv push` to deploy |

HF Spaces Docker tier runs on CPU by default. This is fine — we are not running model inference locally. The HF Inference API call is remote.

HF Inference API model used: `mistralai/Mistral-7B-Instruct-v0.2`
- Free with HF account
- Rate limit: ~300 requests/hour on free tier
- Max input tokens: 4096
- Max output tokens: capped at 200 in our API call to keep it fast

---

## 7. Development Tools

| Tool | Version | Purpose |
|---|---|---|
| Git | 2.x | Version control |
| GitHub | N/A | Repository hosting |
| VS Code | Latest | IDE (recommended) |
| Python extension for VS Code | Latest | Linting, autocomplete |
| Pylance | Latest | Type checking |

---

## 8. Environment Variables (all configuration)

Defined in `.env.example` at repo root. Never commit real values.

```bash
# .env.example
MAX_STEPS=5
DONE_THRESHOLD=0.85
GRADER=rouge
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
TASK_SEED=
ENABLE_WEB_INTERFACE=false
```

For local dev: copy to `.env` and fill in HF_TOKEN.
For Docker: pass via `docker run -e HF_TOKEN=...`
For HF Spaces: set in Spaces Settings → Repository secrets.

---

## 9. openenv.yaml — Manifest

```yaml
name: prompt-opt-env
version: 0.1.0
description: >
  RL environment where an agent learns to iteratively improve prompts
  to maximise LLM output quality. Agent takes editing actions on a
  prompt; reward is ROUGE-L score improvement of the resulting output.
author: Shyam
license: BSD-3-Clause
python: ">=3.11"
entry_point: server.app:app
port: 8000
env_vars:
  - name: MAX_STEPS
    default: "5"
    description: Maximum steps per episode
  - name: GRADER
    default: "rouge"
    description: "rouge or hf_api"
  - name: HF_TOKEN
    required: false
    secret: true
    description: HuggingFace token for Inference API grader
  - name: HF_MODEL
    default: "mistralai/Mistral-7B-Instruct-v0.2"
    description: Model to use for HF Inference API
tags:
  - nlp
  - prompt-engineering
  - rl
  - text
  - llm
```

---

## 10. pyproject.toml — Exact dependency pinning

```toml
[project]
name = "prompt-opt-env"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openenv-core==0.2.1",
    "fastapi==0.115.0",
    "uvicorn==0.30.6",
    "pydantic==2.7.0",
    "rouge-score==0.1.2",
    "httpx==0.27.0",
    "requests==2.31.0",
    "numpy==1.26.4",
    "websockets==12.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.2.0",
    "pytest-asyncio==0.23.6",
    "pytest-cov==5.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 11. What is NOT used and why

| Technology | Reason NOT used |
|---|---|
| HuggingFace transformers | Would make Docker image 8GB+. HF Inference API does the same job remotely for free. |
| LangChain | Overkill, adds complexity, not needed for simple API calls |
| Redis / any database | No persistence needed — episodes are stateless |
| Celery / task queue | Single-user env, no async task processing needed |
| React / any frontend | Not applicable — this is a server-side RL env |
| SQLite / PostgreSQL | No data to persist across episodes |
| sentence-transformers | Heavier than rouge-score, not needed for Round 1 grader |
| Kubernetes | Not needed for HF Spaces deployment |
