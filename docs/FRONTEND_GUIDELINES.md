# FRONTEND_GUIDELINES.md — Interface & Code Standards
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization
# Cross-reference: PRD.md Section 4, BACKEND_STRUCTURE.md
# Version: FINAL
# Last updated: 2026-03-30

---

## 1. No Frontend — What This Document Covers

PromptRL has no browser interface. "Frontend" here means every human-facing output: terminal logs, JSON response shape, README standards, code style rules, and openenv.yaml description quality. These are what judges read during both programmatic and LLM-based evaluation.

---

## 2. Terminal Log Standards

All print() statements in the environment server follow this format.

### Format
```
[PromptRL] [{LEVEL}] {message}
```

### Examples
```
[PromptRL] [BOOT] grader=rouge alpha=0.02 max_steps=7 api_base=https://api-inference.huggingface.co/v1/
[PromptRL] [RESET] episode=a1b2c3 task_id=11 task=Git merge conflict tokens=4/55 score=0.0500
[PromptRL] [STEP 1] action=ADD_CONTEXT score=0.1500 tokens=14/55 overhead=+10 reward=+0.0500
[PromptRL] [STEP 2] action=SHORTEN score=0.1700 tokens=10/55 overhead=-4 reward=+0.1000
[PromptRL] [LLM] Called OpenAI client model=Mistral-7B time=1.4s status=ok
[PromptRL] [LLM] Fallback to dummy output (error: timeout)
[PromptRL] [DONE] reason=voluntary_stop final_score=0.6200 tokens=22/55 efficiency=0.0282
[PromptRL] [WARN] Budget exceeded — action rejected (would be 58/55 tokens)
[PromptRL] [WARN] No-op detected — prompt unchanged
[PromptRL] [WARN] Stuck — same action 3× (penalty applied)
```

### Colour codes (only when stdout is a TTY)
```python
import sys
USE_COLOR = sys.stdout.isatty()
GREEN  = "\033[92m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RED    = "\033[91m" if USE_COLOR else ""
CYAN   = "\033[96m" if USE_COLOR else ""
RESET  = "\033[0m"  if USE_COLOR else ""
```

---

## 3. Python Code Style Standards

Meta engineers will review the code. All code must follow these rules.

### Naming
```python
# Classes: PascalCase
class PromptRLEnvironment(Environment): ...
class PromptAction(BaseModel): ...
class PromptObservation(BaseModel): ...

# Functions/methods: snake_case
def reset(self) -> PromptObservation: ...
def _compute_reward(self, quality_delta: float, token_overhead: int) -> float: ...

# Constants: UPPER_SNAKE_CASE
MAX_STEPS: int = 7
DONE_THRESHOLD: float = 0.85
TOKEN_PENALTY_ALPHA: float = 0.02
TASK_BANK: list[Task] = [...]
DUMMY_OUTPUTS: dict[int, str] = {...}
```

### Type hints — mandatory on every function
```python
def score(self, prompt: str, reference: str, task_id: int) -> tuple[float, str]: ...
def count_tokens(text: str) -> int: ...
def compute_reward(self, quality_delta: float, token_overhead: int) -> float: ...
```

### Docstrings — mandatory on every class and public method
```python
class PromptRLEnvironment(Environment):
    """
    OpenEnv RL environment for cost-aware prompt optimisation.

    The agent edits a prompt using 6 actions (5 editing + STOP) to maximise
    output quality while respecting a token budget. Reward = quality_delta
    minus alpha * token_overhead. Exceeding the token budget terminates the
    episode with a penalty, teaching agents to plan token usage.

    Configuration via environment variables:
        GRADER             — 'rouge' or 'openai_client'. Default: 'rouge'.
        MAX_STEPS          — Max steps per episode. Default: 7.
        DONE_THRESHOLD     — ROUGE-L for success. Default: 0.85.
        TOKEN_PENALTY_ALPHA — Cost penalty coefficient. Default: 0.02.
        API_BASE_URL       — OpenAI-compatible endpoint.
        MODEL_NAME         — Model identifier.
        HF_TOKEN           — API key.
    """
```

### Style rules
- Maximum line length: 100 characters
- Indentation: 4 spaces — no tabs
- Two blank lines between top-level definitions
- One blank line between methods inside a class
- Descriptive error messages:

```python
# Wrong:
raise ValueError("bad action")

# Right:
raise ValueError(
    f"Invalid action_id={action_id}. Must be integer in {{0,1,2,3,4,5}}. "
    f"Action 5 = STOP (voluntary episode end). "
    f"Received type={type(action_id).__name__}, value={action_id!r}."
)
```

### Import order
```python
# Standard library
import os, sys, random
from uuid import uuid4
from dataclasses import dataclass

# Third-party
from openai import OpenAI
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
import numpy as np

# OpenEnv
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Internal
from ..models import PromptAction, PromptObservation
from .task_bank import TASK_BANK
```

---

## 4. JSON Response Standards

Every `step()` and `reset()` response must contain ALL fields below. The programmatic checker accesses fields by name — a missing field is immediate disqualification.

### Required JSON shape
```json
{
  "observation": {
    "task_description": "Describe the steps to resolve a Git merge conflict",
    "current_prompt": "Resolve merge conflict.\nContext: A Git merge conflict...",
    "previous_prompt": "Resolve merge conflict.",
    "current_score": 0.2800,
    "previous_score": 0.1500,
    "current_token_count": 14,
    "previous_token_count": 4,
    "token_budget": 55,
    "tokens_remaining": 41,
    "token_overhead": 10,
    "reward": 0.0500,
    "done": false,
    "step_count": 1,
    "reference_answer": "1. Run git merge... 2. Open conflicted file...",
    "info": {
      "grader_used": "rouge",
      "action_applied": "ADD_CONTEXT",
      "stuck_count": 0,
      "termination_reason": null,
      "llm_output_preview": "To resolve a git merge conflict you need to...",
      "no_op": false
    }
  },
  "reward": 0.0500,
  "done": false
}
```

### Field rules
- `current_score`, `previous_score`: floats in [0.0, 1.0], rounded to 4 decimal places
- `reward`: float in [−2.0, +2.0], rounded to 4 decimal places
- `current_token_count`, `previous_token_count`, `token_budget`, `tokens_remaining`, `token_overhead`: all integers
- `done`: always boolean — never null or string
- `step_count`: integer starting at 0 after reset, 1 after first step
- `info["grader_used"]`: one of `"rouge"`, `"openai_client"`, `"rouge_fallback"`
- `info["action_applied"]`: one of `"ADD_CONTEXT"`, `"SHORTEN"`, `"ADD_EXAMPLE"`, `"REPHRASE"`, `"ADD_CONSTRAINT"`, `"STOP"`, or `null` on reset
- `info["termination_reason"]`: one of `"success"`, `"max_steps"`, `"stuck"`, `"budget_exceeded"`, `"voluntary_stop"`, or `null` if not done
- `info["no_op"]`: boolean

---

## 5. README Standards (judge-facing document)

### Required sections in order
1. Project title + one-line description
2. The differentiation (why existing tools fail at this; what makes this novel)
3. Quick start (copy-paste ready commands)
4. Action space table (all 6 actions, effect on quality AND tokens)
5. Observation space table (all fields with types)
6. Reward function (formula as code block, all special cases, worked example)
7. Token budget mechanics (what happens when budget exceeded)
8. Task bank (table with all 15 tasks including token_budget column)
9. Configuration table (required/optional, with defaults)
10. Baseline scores (table with quality, token count, efficiency per task)
11. Example training loop
12. Running tests
13. Deployment to HF Spaces
14. Round 2 roadmap

### Writing rules
- Lead with the differentiation. The first paragraph after the title must explain what makes this different from DSPy/OPRO/TextGrad.
- Precision: "combined reward = ROUGE-L delta − α × token overhead" not "a combined score"
- Show the worked example in the README — it is the clearest way to explain the cost-aware mechanic
- All terminal commands copy-paste ready (no `<placeholder>` that isn't explained)

---

## 6. openenv.yaml Description Quality

The `description` field is read by the LLM scoring judge. It must be a complete, differentiated paragraph. Do not just say "an RL environment." Explain why it is novel.

The full description is in TECH_STACK.md Section 11. Do not shorten it.

---

## 7. inference.py Output Standards

```
============================================================
PromptRL — Cost-Aware Baseline Inference Script
Model    : mistralai/Mistral-7B-Instruct-v0.2
Endpoint : https://api-inference.huggingface.co/v1/
Alpha    : 0.02
============================================================

  Task (easy): Explain what machine learning is to a 10-year-old
  Token budget  : 80
  Initial tokens: 3   initial_prompt: 'explain machine learning'
  Initial score : 0.1200
  Step 1: action=ADD_CONTEXT  score=0.1500 tokens=13/80 overhead=+10 reward=+0.0500
  Step 2: action=REPHRASE     score=0.1700 tokens=13/80 overhead= +0 reward=+0.0200
  Step 3: action=SHORTEN      score=0.1600 tokens=10/80 overhead= -3 reward=+0.0300
  Step 4: action=STOP         score=0.1600 stop_bonus=0.2400        reward=+0.2400
  [DONE] reason=voluntary_stop

  Task (medium): ...
  [same format]

  Task (hard): ...
  [same format]

============================================================
BASELINE SCORES SUMMARY
============================================================
Difficulty  Score  Tokens  Budget  Efficiency  Reward  Steps
------------------------------------------------------------------
easy        0.4800     22      80      0.0218  0.6300      5
medium      0.3500     35      65      0.0100  0.3100      6
hard        0.2400     28      55      0.0073  0.2200      7
------------------------------------------------------------------
Average     0.3567
Efficiency = final_score / final_token_count (higher = better)

All scores are ROUGE-L F1 in [0.0, 1.0]. Script complete.
```

Rules: all scores 4 decimal places, all tokens integers, exits code 0 always, no input() calls, all exceptions caught.
