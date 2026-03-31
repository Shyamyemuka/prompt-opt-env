---
title: Prompt Opt Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# PromptOptEnv: Cost-Aware Task-Adaptive Prompt Optimization via RL

A Reinforcement Learning environment where an agent learns to improve prompts while staying within a token budget — maximising output quality **per token spent**, not just raw quality.

Built for the **Meta x Scaler OpenEnv Hackathon 2026** on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch and Hugging Face.

---

## Why This Is Different

Every existing prompt optimisation tool — DSPy, TextGrad, OPRO — maximises output quality with no token constraints. They add context, add examples, add chain-of-thought, and prompts grow indefinitely.

In production, this fails. **Prompt tokens cost money at inference time.** Meta serves billions of LLaMA queries daily — a 10% reduction in average prompt length at that scale saves millions in compute costs. Every enterprise LLM deployment has a cost budget.

PromptOptEnv frames this as **constrained RL**: the agent must learn *which prompt improvements are worth their token cost* and *when to stop adding tokens*. The reward signal explicitly balances two competing objectives:

```
reward = quality_improvement - alpha * token_overhead
```

A STOP action lets the agent voluntarily end the episode when quality is good enough for the tokens spent — teaching efficiency, not just quality maximisation.

---

## Quick Start

### Option 1: Interactive Prompt Optimizer (Recommended for Users)

```bash
# 1. Install dependencies
pip install openai rouge-score

# 2. Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# 3. Run interactive optimizer
python optimize.py

# Or see demo examples
python demo_examples.py
```

**Interactive mode lets you:**
- Enter your own prompts
- See before/after comparison with metrics
- View token cost analysis
- Save results to file

### Option 2: Baseline Inference Script

```bash
# 1. Install
cd prompt_opt_env/
pip install -e ".[dev]"

# 2. Set mandatory environment variables
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here

# 3. Run baseline inference script
cd ..
python inference.py
```

---

## Live Environment (HF Spaces)

```python
from prompt_opt_env import PromptOptEnvEnv, PromptAction

async with PromptOptEnvEnv(base_url="wss://{username}-prompt-opt-env.hf.space") as env:
    result = await env.reset()
    # run episode...
```

---

## Action Space

6 actions. The key innovation: each action has a different token cost profile that the agent must learn to balance against its quality benefit.

| ID | Name | Quality Effect | Token Effect | Description |
|---|---|---|---|---|
| 0 | `ADD_CONTEXT` | +medium | +10–15 tokens | Appends domain context sentence |
| 1 | `SHORTEN` | −small or neutral | −5–12 tokens | Removes filler phrases via regex — the only token-reducing action |
| 2 | `ADD_EXAMPLE` | +medium-high | +12–20 tokens | Appends example output format |
| 3 | `REPHRASE` | +small | ±0 tokens | Converts questions to direct imperatives — free quality improvement |
| 4 | `ADD_CONSTRAINT` | +small-medium | +8–12 tokens | Appends output constraint |
| 5 | `STOP` | — | 0 tokens | Voluntarily end episode. Reward = current_score × 1.5 |

The STOP action is the core differentiator: it forces the agent to learn when it has done enough, rather than always adding more context until steps run out.

---

## Observation Space

Every `reset()` and `step()` returns a `PromptObservation`. All fields always present.

| Field | Type | Description |
|---|---|---|
| `task_description` | `str` | What the prompt should accomplish |
| `current_prompt` | `str` | Prompt after this step |
| `previous_prompt` | `str` | Prompt before this step |
| `current_score` | `float` [0,1] | ROUGE-L F1 of current prompt output |
| `previous_score` | `float` [0,1] | ROUGE-L F1 before this step |
| `current_token_count` | `int` | Word-level token count of current prompt |
| `previous_token_count` | `int` | Token count before this step |
| `token_budget` | `int` | Hard ceiling for this task (easy=80, medium=65, hard=55) |
| `tokens_remaining` | `int` | token_budget − current_token_count |
| `token_overhead` | `int` | Tokens added this step (negative if SHORTEN applied) |
| `reward` | `float` | Combined reward, clipped to [−2.0, +2.0] |
| `done` | `bool` | True if episode ended |
| `step_count` | `int` | Steps taken this episode |
| `reference_answer` | `str` | Gold-standard answer for grader |
| `info` | `dict` | grader_used, action_applied, stuck_count, termination_reason, llm_output_preview, no_op |

---

## Reward Function

```python
# At each editing step (actions 0–4):
token_overhead = current_token_count - previous_token_count  # negative if SHORTEN
quality_delta  = current_score - previous_score
raw_reward     = quality_delta - alpha * token_overhead       # alpha = TOKEN_PENALTY_ALPHA = 0.02
reward         = clip(raw_reward, -2.0, +2.0)

# STOP action (action_id = 5):
reward = clip(current_score × 1.5, 0.0, +2.0)
done   = True

# Special cases:
# No-op (action has no effect):          reward = -0.1,  episode continues
# Stuck (same action 3× in a row):       reward = -0.5,  done = True
# Budget exceeded (tokens > budget):     reward = -0.5,  done = True, prompt reverts
# Success (ROUGE_L > 0.85):              reward += +1.0 bonus, done = True
# Max steps (step_count = MAX_STEPS=7):   done = True
```

**Why alpha = 0.02**: Adding 10 tokens costs 0.2 reward. A typical quality improvement of +0.10 ROUGE-L is worth +0.10. So adding more than 5 tokens is only justified if quality improves by more than 0.10 — a calibrated trade-off that creates genuine learning tension.

### Worked Example

```
reset() — task: "Summarise Romeo and Juliet in exactly 2 sentences"
  initial_prompt: "tell me about romeo and juliet"  [7 tokens, budget=80]
  initial_score:  0.12

step(REPHRASE)   →  "Summarise the plot of Romeo and Juliet."  [7 tokens, overhead=0]
  quality_delta=+0.16, reward = 0.16 - 0.02×0 = +0.16   ✓ free improvement

step(ADD_EXAMPLE) → adds 14 tokens
  quality_delta=+0.23, reward = 0.23 - 0.02×14 = -0.05  ✗ too expensive

step(ADD_CONSTRAINT) → adds 9 tokens
  quality_delta=+0.21, reward = 0.21 - 0.02×9 = +0.03   ~ marginal

step(SHORTEN)    → removes 8 tokens
  quality_delta=+0.03, reward = 0.03 - 0.02×(-8) = +0.19  ✓ negative overhead = bonus!

step(STOP)       → current_score=0.75
  stop_bonus = 0.75 × 1.5 = +1.125                        ✓ agent decides this is enough

Total reward: 0.16 - 0.05 + 0.03 + 0.19 + 1.125 = 1.455
Token efficiency: 0.75 quality at 22 tokens (vs budget of 80)
```

---

## Token Budget

Each task has a hard token ceiling. If any action would cause the prompt to exceed the budget, that action is rejected, the prompt reverts, and the episode ends with a penalty (−0.5). This teaches the agent to plan token usage across the full episode, not just locally.

| Difficulty | Token Budget | Why |
|---|---|---|
| Easy | 80 | More slack — agent explores freely |
| Medium | 65 | Meaningful constraint — must choose actions wisely |
| Hard | 55 | Tight — requires efficient language from the start |

---

## Task Bank

15 tasks across 4 categories with explicit difficulty and token budget per task.

| ID | Category | Description | Difficulty | Budget |
|---|---|---|---|---|
| 0 | Summarisation | Climate change article → 3 bullets | Easy | 80 |
| 1 | Summarisation | Romeo and Juliet → 2 sentences | Easy | 80 |
| 2 | Summarisation | Crypto risks → under 60 words | Medium | 65 |
| 3 | Summarisation | French Revolution → chronological bullets | Medium | 65 |
| 4 | Summarisation | Machine learning → explain to 10-year-old | Easy | 80 |
| 5 | QA | Binary search time complexity and why? | Medium | 65 |
| 6 | QA | What causes inflation, how does central bank control it? | Medium | 65 |
| 7 | QA | Difference between RAM and ROM? | Easy | 80 |
| 8 | QA | Why blue sky by day, red at sunset? | Easy | 80 |
| 9 | Instruction | Steps to make a cup of tea | Easy | 80 |
| 10 | Instruction | Set up Python venv on Windows | Medium | 65 |
| 11 | Instruction | Resolve a Git merge conflict | Hard | 55 |
| 12 | Code | Python list comprehension with example | Medium | 65 |
| 13 | Code | Big O notation with code example | Medium | 65 |
| 14 | Code | What is recursion with Python example | Easy | 80 |

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | **Yes** | — | OpenAI-compatible endpoint. HF: `https://router.huggingface.co/v1/` |
| `MODEL_NAME` | **Yes** | — | E.g. `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | **Yes** | — | HuggingFace token |
| `MAX_STEPS` | No | `7` | Max steps per episode |
| `DONE_THRESHOLD` | No | `0.85` | ROUGE-L for success bonus |
| `TOKEN_PENALTY_ALPHA` | No | `0.02` | Cost penalty alpha in reward formula |
| `GRADER` | No | `rouge` | `rouge` (no API) or `openai_client` |
| `TASK_SEED` | No | random | Fix task (0–14) for reproducibility |

---

## Baseline Scores

From running `python inference.py` with a random agent (7 steps max, with STOP).
Efficiency = final_score / final_token_count (higher = better quality per token).

| Difficulty | Score | Tokens | Budget | Efficiency | Reward | Steps |
|---|---|---|---|---|---|---|
| Easy | 0.4800 | 22 | 80 | 0.0218 | 0.6300 | 5 |
| Medium | 0.3500 | 35 | 65 | 0.0100 | 0.3100 | 6 |
| Hard | 0.2400 | 28 | 55 | 0.0073 | 0.2200 | 7 |
| **Average** | **0.3567** | | | | | |

*Run `python inference.py` to get your exact reproducible scores.*

---

## Example Training Loop

```python
import asyncio
from prompt_opt_env import PromptOptEnvEnv, PromptAction

# Standard GRPO / TRL / torchforge integration loop
async def collect_episode(env_url: str) -> list[dict]:
    """Collect one episode. Reward includes token cost penalty."""
    trajectory = []
    async with PromptOptEnvEnv(base_url=env_url) as env:
        result = await env.reset()
        while not result.done:
            # Your policy observes quality score, token count, and budget
            obs = result.observation
            action_id = your_policy(obs)  # e.g. learns to STOP when efficient
            result = await env.step(PromptAction(action_id=action_id))
            trajectory.append({
                "observation": result.observation.model_dump(),
                "reward": result.reward,           # cost-aware combined reward
                "done": result.done,
                "token_efficiency": (
                    result.observation.current_score /
                    max(1, result.observation.current_token_count)
                ),
            })
    return trajectory
```

---

## Running Tests

```bash
cd prompt_opt_env
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Deployment

```bash
huggingface-cli login
cd prompt_opt_env
openenv push --repo-id {username}/prompt-opt-env

# Set in HF Spaces → Settings → Repository secrets:
#   API_BASE_URL, MODEL_NAME, HF_TOKEN
```

---

## Project Structure

```
prompt-opt-env/
├── inference.py                      # Mandatory baseline script
├── .env.example
└── prompt_opt_env/
    ├── __init__.py                   # Exports: PromptAction, PromptObservation, PromptOptEnvEnv
    ├── models.py                     # Pydantic models (all token fields)
    ├── client.py                     # PromptOptEnvEnv WebSocket client
    ├── openenv.yaml                  # OpenEnv manifest
    ├── pyproject.toml                # Dependencies
    └── server/
        ├── app.py                    # FastAPI app
        ├── prompt_opt_env_environment.py  # Core RL logic
        ├── actions.py                # 5 edit functions + count_tokens()
        ├── grader.py                 # ROUGE + OpenAI client
        ├── task_bank.py              # 15 tasks with token_budget
        └── Dockerfile
```

---

## Round 2 Roadmap (Bangalore, April 25–26)

- Exact BPE token counting via tiktoken (replaces word-split)
- Adaptive alpha: cost penalty increases as tokens_remaining decreases
- LLM-as-judge grader replacing ROUGE-L
- Multi-objective Pareto front tracking (quality vs. cost per task category)
- ADD_COT action with explicit high token cost (+15 tokens)
- Multi-task episode: 3 tasks, shared total token budget across all 3

---

## Author

Shyam — B.Tech Data Science, Annamacharya University, Rajampet, Andhra Pradesh, India
Solo submission — Meta x Scaler OpenEnv Hackathon 2026

---

## License

BSD-3-Clause (same as OpenEnv)
