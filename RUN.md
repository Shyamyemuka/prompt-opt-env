# RUN.md - Meta x Scaler OpenEnv Hackathon Guide

This runbook is updated to reflect the current implementation and Docker-validated workflow for `prompt-opt-env`.

## 1. Prerequisites

- Python `3.11+`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop (running)
- HuggingFace account + token (`HF_TOKEN`)
- OpenEnv CLI (`openenv`)

## 2. Fresh Install (Using uv)

From repository root:

```bash
git clone <your-repo-url>
cd prompt-opt-env
uv venv
uv pip install -e "./prompt_opt_env[dev]"
```

## 3. Configure Environment Variables

Create `.env.local` in repo root:

```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1/
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TOKEN_PENALTY_ALPHA=0.02
MAX_STEPS=7
DONE_THRESHOLD=0.85
LLM_TIMEOUT_SECONDS=10
```

PowerShell example:

```powershell
$env:HF_TOKEN="hf_your_token_here"
$env:API_BASE_URL="https://router.huggingface.co/v1/"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

## 4. Run Locally (Non-Docker)

### A) Web UI (Prompt Optimizer)

```bash
python web_app.py
```

Open: `http://localhost:5000`

### B) OpenEnv API server

```bash
cd prompt_opt_env
uv run server
```

OpenAPI health:

```bash
curl http://localhost:8000/health
```

### C) Baseline inference script (mandatory file)

From repo root:

```bash
python inference.py
```

## 5. Docker Workflow (Validated)

This is the workflow currently used to validate container behavior.

### A) Build image

From repo root:

```bash
docker build -t prompt-opt-env-web:latest -f prompt_opt_env/server/Dockerfile prompt_opt_env
```

### B) Run container

```powershell
docker run -d --name prompt-opt-env-web-run -p 8000:8000 ^
  -e API_BASE_URL=https://router.huggingface.co/v1/ ^
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct ^
  -e HF_TOKEN=hf_your_token_here ^
  -e GRADER=rouge ^
  prompt-opt-env-web:latest
```

### C) Verify container

PowerShell:

```powershell
docker ps --filter "name=prompt-opt-env-web-run"
docker logs --tail 40 prompt-opt-env-web-run
(Invoke-WebRequest -UseBasicParsing http://localhost:8000/health).Content
(Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://localhost:8000/reset -ContentType "application/json" -Body "{}").Content
```

Expected:
- `/health` returns JSON with healthy status.
- `/reset` returns a valid observation payload.

### D) Stop and clean up

```bash
docker rm -f prompt-opt-env-web-run
```

## 6. API Endpoint Modes

Use HuggingFace Router (recommended):

```env
API_BASE_URL=https://router.huggingface.co/v1/
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_...
```

Use local OpenAI-compatible endpoint (vLLM/Ollama proxy/etc.):

```env
API_BASE_URL=http://localhost:8000/v1/
MODEL_NAME=<your-local-model-name>
HF_TOKEN=dummy_or_local_key
```

Notes:
- Endpoint must support OpenAI Chat Completions (`/v1/chat/completions`).
- `GRADER=rouge` avoids external grader calls and is deterministic.
- Web UI now has a fallback output path if upstream LLM calls fail.
- `web_app.py` accepts key aliases:
  - token: `HF_TOKEN` (or `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`)
  - base URL: `API_BASE_URL` (or `OPENAI_BASE_URL`)
  - model: `MODEL_NAME` (or `OPENAI_MODEL`)
- `web_app.py` forces direct API calls (`trust_env=False`) to avoid broken global proxy env vars.

## 7. OpenEnv Validate and Deploy

From `prompt_opt_env/`:

```bash
cd prompt_opt_env
openenv validate
openenv push --repo-id <hf-username>/prompt-opt-env
```

Set HuggingFace Space secrets:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Post-deploy check:

```bash
curl https://<hf-username>-prompt-opt-env.hf.space/health
```

## 8. Submission Sanity Checklist

- `openenv validate` passes.
- Docker image builds successfully.
- Docker `/health` and `/reset` respond.
- `python inference.py` runs from repo root.
- `inference.py` remains at repo root (not moved).
- Space secrets are configured.

## 9. Dependency Notes

Current compatibility pins for container build stability:
- `fastapi==0.115.12`
- `pydantic==2.7.4`

## 10. Fallback Troubleshooting (Web UI)

If you see: `API connection failed, so offline fallback outputs were used for this run.`

1. Confirm the runtime diagnostics shown below that message:
- Token detected should be `yes`.
- Base URL should be your intended endpoint.
- Model should be valid for the endpoint.

2. Quick key check in PowerShell:

```powershell
Get-Content .env.local | Select-String "HF_TOKEN|API_BASE_URL|MODEL_NAME"
```

3. Restart app after changing keys:

```bash
python web_app.py
```
