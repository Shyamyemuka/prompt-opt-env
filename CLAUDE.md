# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

**PromptRL** is a cost-aware Reinforcement Learning environment for task-adaptive prompt optimisation, built on Meta's OpenEnv framework. It trains RL agents to maximise LLM output quality *per token spent* — not just raw quality. Every prompt edit has a token cost. The agent must learn which edits are worth it and when to stop.

**Submitted to:** Meta x Scaler OpenEnv Hackathon 2026 (Round 1, Solo)
**Deadline:** April 7, 2026, 11:59 PM IST
**HF Spaces URL:** `https://{username}-prompt-rl.hf.space`

---

## Architecture

### Repository Structure

```
prompt-rl/                              ← repo root
│
├── inference.py                        ← MANDATORY: baseline script (must be here)
├── .env.example                        ← API_BASE_URL, MODEL_NAME, HF_TOKEN template
├── .gitignore
├── README.md
├── CLAUDE.md                           ← this file
│
└── prompt_rl/                          ← OpenEnv environment package
    ├── __init__.py                     ← exports: PromptAction, PromptObservation, PromptRLEnv
    ├── models.py                       ← Pydantic v2 models
    ├── client.py                       ← WebSocket client (EnvClient subclass)
    ├── openenv.yaml                    ← OpenEnv manifest
    ├── pyproject.toml
    │
    ├── server/
    │   ├── app.py                      ← FastAPI app via create_app()
    │   ├── prompt_rl_environment.py    ← core RL logic
    │   ├── actions.py                  ← 6 actions + count_tokens()
    │   ├── grader.py                   ← ROUGE-L + OpenAI client
    │   ├── task_bank.py                ← 15 tasks with token_budget
    │   ├── requirements.txt
    │   └── Dockerfile
    │
    └── tests/
        ├── test_actions.py
        ├── test_grader.py
        └── test_environment.py
```

---

## Core Concepts

### The Differentiating Idea
Every existing prompt optimiser (DSPy, TextGrad, OPRO) maximises quality with no token constraints. PromptRL frames this as **constrained RL**: the agent learns *which prompt improvements are worth their token cost* and *when to stop*. Token cost is a first-class citizen in the reward.

### Action Space (6 actions, IDs 0–5)

| ID | Name | Token Effect | Quality Effect |
|---|---|---|---|
| 0 | ADD_CONTEXT | +10–15 tokens | +medium |
| 1 | SHORTEN | −5–12 tokens | neutral/−small |
| 2 | ADD_EXAMPLE | +12–20 tokens | +medium-high |
| 3 | REPHRASE | ±0 tokens | +small (free) |
| 4 | ADD_CONSTRAINT | +8–12 tokens | +small-medium |
| 5 | STOP | 0 tokens | — (voluntary end) |

STOP is the key mechanic: `reward = current_score × 1.5`. Agent learns to stop when quality/cost ratio is good enough.

### Reward Formula
```python
# Editing actions (0–4):
token_overhead = count_tokens(new_prompt) - count_tokens(old_prompt)
quality_delta  = new_rouge_l - old_rouge_l
reward         = clip(quality_delta - TOKEN_PENALTY_ALPHA * token_overhead, -2.0, +2.0)

# STOP (action 5):
reward = clip(current_score * 1.5, 0.0, +2.0)

# Penalties:
# No-op (prompt unchanged)  → reward = -0.1, episode continues
# Stuck (same action ×3)    → reward = -0.5, done = True
# Budget exceeded            → reward = -0.5, done = True, prompt REVERTS
# Success (score > 0.85)    → reward += +1.0 bonus, done = True
```

### Token Counting
`count_tokens(text: str) -> int` = `len(text.split())` — word-level approximation. No external library. Consistent for relative comparisons. Never use tiktoken here — adds unnecessary dependency.

### Token Budgets per Difficulty
- Easy tasks: 80 tokens
- Medium tasks: 65 tokens
- Hard tasks: 55 tokens (tighter = requires more concise language)

---

## Configuration (Environment Variables)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `API_BASE_URL` | YES | — | OpenAI-compatible endpoint. For HF: `https://api-inference.huggingface.co/v1/` |
| `MODEL_NAME` | YES | — | e.g. `mistralai/Mistral-7B-Instruct-v0.2` |
| `HF_TOKEN` | YES | — | HuggingFace token |
| `MAX_STEPS` | No | `7` | Max steps per episode |
| `DONE_THRESHOLD` | No | `0.85` | ROUGE-L for success bonus |
| `TOKEN_PENALTY_ALPHA` | No | `0.02` | Cost penalty α in reward formula |
| `GRADER` | No | `rouge` | `rouge` (no API) or `openai_client` |
| `TASK_SEED` | No | random | Int 0–14, fixes task for reproducibility |

**All three mandatory vars must be set.** The submission checker verifies them at startup.

---

## Common Commands

### Setup
```bash
cd prompt_rl
pip install uv
uv pip install -e ".[dev]"

# Verify
python -c "import openenv; print('openenv OK')"
python -c "from openai import OpenAI; print('openai OK')"
python -c "from rouge_score import rouge_scorer; print('rouge OK')"
```

### Run server locally
```bash
cd prompt_rl
GRADER=rouge uv run server
# Server at http://localhost:8000

# With real LLM:
API_BASE_URL=https://api-inference.huggingface.co/v1/ \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
HF_TOKEN=hf_your_token \
GRADER=openai_client \
uv run server
```

### Run tests
```bash
cd prompt_rl
python -m pytest tests/ -v
python -m pytest tests/test_actions.py -v
python -m pytest tests/test_environment.py -v
python -m pytest tests/ --cov=prompt_rl -v
```

### Run baseline inference script
```bash
# From repo root (not from inside prompt_rl/)
export API_BASE_URL=https://api-inference.huggingface.co/v1/
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
export HF_TOKEN=hf_your_token

python inference.py
# Prints cost-aware summary with efficiency column
# Must complete in under 20 minutes
```

### Docker build and test
```bash
# Build
docker build -t prompt-rl:latest -f prompt_rl/server/Dockerfile .

# Run
docker run -d -p 8000:8000 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1/ \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
  -e HF_TOKEN=hf_your_token \
  -e GRADER=rouge \
  prompt-rl:latest

# Verify
curl http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"
```

### OpenEnv validate and deploy
```bash
cd prompt_rl
openenv validate         # Must pass before deploying
openenv push --repo-id {username}/prompt-rl
```

---

## Code Patterns

### Adding a New Action
1. Add function in `server/actions.py` (pure string transform, no LLM calls)
2. Update `ACTION_NAMES` dict in `server/actions.py`
3. Update `apply_action()` dispatcher (handle only 0–4; 5=STOP stays in env)
4. Update `PromptAction` docstring in `models.py`
5. Add tests in `tests/test_actions.py` — must test no-duplicate, token delta, and no-op cases
6. Update action space table in `README.md`

### LLM Calls — MANDATORY PATTERN
All LLM calls MUST use the OpenAI Python client. Never use httpx or requests to call LLMs.

```python
# CORRECT — always this pattern
from openai import OpenAI
client = OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=os.getenv("HF_TOKEN"))
response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200,
    temperature=0.1,
    timeout=30,
)

# WRONG — never this
import httpx
response = httpx.post("https://api-inference.huggingface.co/...", ...)
```

This is a hackathon rule. The submission checker verifies OpenAI client usage.

### Environment Variables — Read Pattern
```python
# MANDATORY vars — use os.environ (raises KeyError if missing, which is correct)
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

# OPTIONAL vars — use os.getenv with defaults
MAX_STEPS = int(os.getenv("MAX_STEPS", "7"))
TOKEN_PENALTY_ALPHA = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
```

### Writing Tests
Tests use `pytest-asyncio` (asyncio_mode = auto in pyproject.toml). Async tests need no decorator.

```python
# Environment tests need a running server OR use the env class directly:
from prompt_rl.server.prompt_rl_environment import PromptRLEnvironment
from prompt_rl.models import PromptAction

def test_reset():
    env = PromptRLEnvironment()
    obs = env.reset()
    assert obs.token_budget > 0

# For WebSocket client tests (require running server):
async def test_client_episode():
    async with PromptRLEnv(base_url="ws://localhost:8000") as env:
        result = await env.reset()
        assert result.observation.step_count == 0
```

### Observation Fields — Always All Present
The programmatic checker accesses every field by name. Never make a field Optional that might be missing. The full list must always be in every response:
`task_description, current_prompt, previous_prompt, current_score, previous_score, current_token_count, previous_token_count, token_budget, tokens_remaining, token_overhead, reward, done, step_count, reference_answer, info`

And `info` must always contain: `grader_used, action_applied, stuck_count, termination_reason, llm_output_preview, no_op`

---

## Hackathon Submission Checklist

Before every commit that might be the final submission:

```
[ ] openenv validate passes (no errors)
[ ] docker build succeeds from repo root
[ ] docker run starts, /health returns {"status": "ok"}
[ ] reset() returns PromptObservation with ALL fields including all token fields
[ ] step(action_id=5) returns done=True, reward=current_score*1.5
[ ] Budget exceeded path returns done=True, reward=-0.5, prompt reverts
[ ] inference.py at repo root, not inside prompt_rl/
[ ] python inference.py prints cost-aware table with efficiency column, exits 0
[ ] inference.py uses os.environ["API_BASE_URL"], ["MODEL_NAME"], ["HF_TOKEN"]
[ ] All LLM calls use openai.OpenAI — no httpx/requests
[ ] HF Spaces URL returns 200 and responds to reset()
[ ] API_BASE_URL, MODEL_NAME, HF_TOKEN set as HF Spaces secrets
[ ] GitHub repo is public
[ ] README differentiation paragraph is the first section after title
[ ] .env is NOT committed to GitHub
```

**Hard deadline: April 7, 2026, 11:59 PM IST**

---

## Known Constraints

- Docker image must build and run on 2 vCPU / 8 GB RAM — no local model weights
- inference.py must complete in under 20 minutes — max 21 API calls (3 tasks × 7 steps)
- Token counting uses word-split only — do NOT add tiktoken to Round 1
- Grader fallback is mandatory — if OpenAI client call fails, fall back to DUMMY_OUTPUTS silently
- STOP action (action_id=5) is handled directly in `prompt_rl_environment.py`, NOT in `actions.py`

---

## Dependencies

```
openenv-core==0.2.1
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.7.0
openai==1.30.1
rouge-score==0.1.2
numpy==1.26.4
websockets==12.0

# dev only
pytest==8.2.0
pytest-asyncio==0.23.6
pytest-cov==5.0.0
```

Python: >=3.11
