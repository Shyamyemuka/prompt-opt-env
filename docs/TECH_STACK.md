# TECH_STACK.md — Technology Stack
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization
# Cross-reference: BACKEND_STRUCTURE.md, IMPLEMENTATION_PLAN.md
# Version: FINAL
# Last updated: 2026-03-30

---

## 1. Core Framework

| Technology | Exact Version | Role | Why |
|---|---|---|---|
| Python | 3.11.9 | Runtime | OpenEnv requires ≥3.10; 3.11 best perf/stability |
| openenv-core | 0.2.1 | Base classes, CLI, create_app() | Latest stable; matches dashboard |
| uv | 0.4.x latest | Package manager | Replaces pip in dev; used by openenv run commands |

---

## 2. Web Server

| Technology | Exact Version | Role |
|---|---|---|
| FastAPI | 0.115.0 | HTTP + WebSocket server |
| Uvicorn | 0.30.6 | ASGI server |
| Pydantic | 2.7.0 | Typed Action/Observation models (v2 required by OpenEnv 0.2.1) |
| websockets | 12.0 | WebSocket protocol |

---

## 3. LLM Integration (MANDATORY: OpenAI client only)

| Technology | Exact Version | Role |
|---|---|---|
| openai | 1.30.1 | **MANDATORY** — only permitted method for LLM calls |

The hackathon dashboard states: *"Participants must use OpenAI Client for all LLM calls using the variables API_BASE_URL, MODEL_NAME, HF_TOKEN."*

HuggingFace Inference API has an OpenAI-compatible endpoint at `https://api-inference.huggingface.co/v1/`. The openai package points to it via `base_url`:

```python
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
response = client.chat.completions.create(model=MODEL_NAME, messages=[...])
```

**httpx and requests are NOT used for LLM calls. Do not add them.**

---

## 4. NLP / Scoring

| Technology | Exact Version | Role |
|---|---|---|
| rouge-score | 0.1.2 | ROUGE-L F1 computation for quality scoring |
| numpy | 1.26.4 | Reward clipping and arithmetic |

**Token counting:** `len(prompt.split())` — word-level approximation. No tiktoken needed. This requires zero additional dependencies, keeps Docker image small, and is consistent for relative comparisons. The trade-off (word tokens vs BPE tokens) is fine because we track *changes* in token count, not exact BPE counts.

---

## 5. Testing

| Technology | Exact Version | Role |
|---|---|---|
| pytest | 8.2.0 | Test runner |
| pytest-asyncio | 0.23.6 | Async test support |
| pytest-cov | 5.0.0 | Coverage reporting |

---

## 6. Containerisation

| Technology | Version | Role |
|---|---|---|
| Docker Desktop | 25.x (host) | Build and run container locally |
| openenv-base | latest | Base image (Python 3.11, uv, FastAPI pre-installed) |

`server/Dockerfile` uses `FROM openenv-base:latest`. Python and uv come from the base image — do not install them manually in the Dockerfile.

---

## 7. Deployment

| Technology | Detail | Role |
|---|---|---|
| Hugging Face Spaces | Docker Spaces (free tier) | Hosts the environment server |
| HF Inference API | Serverless, free | Remote LLM inference via OpenAI-compatible endpoint |
| openenv CLI | 0.2.1 | `openenv push` deploys to HF Spaces |

HF Inference API:
- Endpoint: `https://api-inference.huggingface.co/v1/`
- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Rate limit: ~300 req/hour (free tier)
- Max output tokens: capped at 200 per call
- Timeout: 30 seconds (set in openai client call)

---

## 8. Mandatory Environment Variables

These exact names are checked by the automated submission pipeline.

```bash
# .env.example — copy to .env, fill in real values, NEVER commit .env

# MANDATORY — submission checker verifies all three
API_BASE_URL=https://api-inference.huggingface.co/v1/
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# OPTIONAL — tuning
MAX_STEPS=7
DONE_THRESHOLD=0.85
TOKEN_PENALTY_ALPHA=0.02
GRADER=rouge
TASK_SEED=
ENABLE_WEB_INTERFACE=false
```

How each mandatory var is used:
- `API_BASE_URL` → `OpenAI(base_url=API_BASE_URL, ...)`
- `MODEL_NAME` → `client.chat.completions.create(model=MODEL_NAME, ...)`
- `HF_TOKEN` → `OpenAI(..., api_key=HF_TOKEN)`

---

## 9. pyproject.toml — Final Pinned Dependencies

```toml
[project]
name = "prompt-rl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openenv-core==0.2.1",
    "fastapi==0.115.0",
    "uvicorn==0.30.6",
    "pydantic==2.7.0",
    "openai==1.30.1",
    "rouge-score==0.1.2",
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

## 10. server/requirements.txt (Docker)

```
openenv-core==0.2.1
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.7.0
openai==1.30.1
rouge-score==0.1.2
numpy==1.26.4
websockets==12.0
```

Mirrors pyproject.toml dependencies. Used by `RUN pip install -r` in Dockerfile.

---

## 11. openenv.yaml — Final Manifest

```yaml
name: prompt-rl
version: 0.1.0
description: >
  PromptRL is a cost-aware Reinforcement Learning environment for task-adaptive
  prompt optimisation. Unlike existing tools (DSPy, TextGrad, OPRO) that maximise
  quality with unlimited token budgets, PromptRL trains agents to maximise
  output quality per token spent. The reward is shaped by two competing objectives:
  ROUGE-L quality improvement minus a token cost penalty. Each task has a hard token
  budget; exceeding it terminates the episode with a penalty. A STOP action lets the
  agent voluntarily end the episode when quality is good enough for the tokens spent.
  This models real production constraints — every token costs money at inference scale.
  Includes 15 tasks (summarisation, QA, instruction-following, code explanation) with
  explicit easy/medium/hard difficulty and decreasing token budgets.
author: Shyam
license: BSD-3-Clause
python: ">=3.11"
entry_point: server.app:app
port: 8000
env_vars:
  - name: API_BASE_URL
    required: true
    secret: false
    description: OpenAI-compatible API endpoint (e.g. https://api-inference.huggingface.co/v1/)
  - name: MODEL_NAME
    required: true
    secret: false
    description: Model identifier (e.g. mistralai/Mistral-7B-Instruct-v0.2)
  - name: HF_TOKEN
    required: true
    secret: true
    description: HuggingFace API token
  - name: MAX_STEPS
    required: false
    default: "7"
    description: Maximum steps per episode (includes STOP action)
  - name: DONE_THRESHOLD
    required: false
    default: "0.85"
    description: ROUGE-L above which episode ends with success bonus
  - name: TOKEN_PENALTY_ALPHA
    required: false
    default: "0.02"
    description: Cost penalty coefficient alpha in reward = quality_delta - alpha * token_overhead
  - name: GRADER
    required: false
    default: "rouge"
    description: "rouge (no API, deterministic) or openai_client (real LLM)"
  - name: TASK_SEED
    required: false
    default: ""
    description: Integer 0-14. Fixes task selection for reproducibility.
tags:
  - nlp
  - prompt-engineering
  - reinforcement-learning
  - cost-aware
  - token-efficiency
  - llm
  - openenv
  - constrained-rl
```

---

## 12. What Is NOT Used and Why

| Technology | Reason |
|---|---|
| httpx | Replaced by openai package — hackathon mandates OpenAI client for all LLM calls |
| requests | Same reason |
| tiktoken | Word-split token counting is sufficient for relative comparisons; avoids heavy dependency |
| HuggingFace transformers | Remote API via openai client does the same job; keeps image ~400 MB |
| LangChain | Unnecessary complexity |
| Redis / any database | No persistence needed — stateless RL env |
| React / any frontend | Server-side environment only |
| Kubernetes | HF Spaces handles deployment |

---

## 13. Infrastructure Constraints Compliance

| Constraint | Value | How Complied |
|---|---|---|
| CPU | ≤2 vCPU | Single-threaded; no multi-processing |
| RAM | ≤8 GB | No local model; image ~400 MB |
| Inference runtime | <20 min | 3 tasks × 7 steps × 1 API call = 21 calls max; each ≤30s → max 10.5 min |
| Docker | Must build automatically | Single `docker build -f server/Dockerfile .` |
