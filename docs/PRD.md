# PRD.md — Product Requirements Document
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization via RL
# Hackathon: Meta x Scaler OpenEnv Hackathon 2026
# Submission: Round 1 | Solo — Shyam | Annamacharya University
# Version: FINAL
# Last updated: 2026-03-30

---

## 1. One-Line Summary

PromptRL is a Reinforcement Learning environment where an agent learns to improve prompts for diverse real-world tasks while staying within a token budget — maximising output quality per token spent, not just raw quality.

---

## 2. Problem Statement and Differentiation

### Why existing tools are not enough

Every existing prompt optimisation tool — DSPy, TextGrad, OPRO, APE — treats prompt engineering as a single-objective problem: maximise output quality with no constraints. They add context, add examples, add chain-of-thought, and the prompt grows indefinitely.

In production, this fails. Prompt tokens cost money at inference time. Meta serves billions of LLaMA queries daily — a 10% reduction in average prompt token count at that scale saves millions in compute costs. Every enterprise LLM deployment has a cost budget.

### What PromptRL does differently

PromptRL frames prompt optimisation as **constrained RL**: the agent must learn *which prompt improvements are worth their token cost* and *when to stop adding tokens*. The reward signal is explicitly shaped by two competing objectives:

```
reward = quality_improvement - α × token_overhead
```

Where:
- `quality_improvement` = delta in ROUGE-L score after the action
- `token_overhead` = tokens added to the prompt by this action (can be negative if SHORTEN was applied)
- `α` = cost penalty coefficient (configurable, default 0.02)

This creates a genuine trade-off: `ADD_CONTEXT` may improve ROUGE-L by +0.08 but cost 15 tokens. `SHORTEN` may reduce ROUGE-L by −0.02 but save 12 tokens. The agent must learn which choice is better given the current budget.

Additionally, each task has a `token_budget` — a hard ceiling on prompt length. Exceeding it terminates the episode with a budget penalty, teaching the agent to be efficient from the start.

### Why judges at Meta will care

Meta's real infrastructure challenge is serving high-quality outputs at minimum compute cost. An RL environment that trains agents to optimise this trade-off is directly relevant to their production systems. No team at this hackathon will be framing it this way. This is the differentiation.

---

## 3. Hackathon Rule Compliance

### 3.1 Official Requirements Checklist

| Requirement | How PromptRL Satisfies It | Status |
|---|---|---|
| Real-world task (not games/toys) | Cost-aware prompt optimisation is a genuine daily production engineering problem | PASS |
| Full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml | Fully implemented per RFC 002 | PASS |
| Min 3 tasks, easy→medium→hard, graders scoring 0.0–1.0 | 15 tasks across 4 categories, 3 difficulties, ROUGE-L returns floats | PASS |
| Meaningful reward with partial progress signals | Dense reward at every step: quality delta minus token penalty | PASS |
| Baseline inference script `inference.py` at repo root | Complete, uses OpenAI client, <20 min | PASS |
| Deploy to HF Spaces + working Dockerfile | Both implemented | PASS |
| README with description, action/obs spaces, tasks, setup, baseline scores | All sections present | PASS |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` defined | All three used everywhere | PASS |
| OpenAI Python client for ALL LLM calls | `openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)` | PASS |
| `inference.py` at repo root, named exactly | Yes | PASS |
| ≤2 vCPU, ≤8 GB RAM, inference <20 min | No local model; image ~400 MB; 15 API calls total | PASS |
| Pass pre-submission validation script | Included in Phase 8 | PASS |

### 3.2 Evaluation Criteria Mapping

| Criterion | Weight | How We Score Well |
|---|---|---|
| Real-world utility | 30% | Cost-aware optimisation is a genuine unsolved production problem; Meta runs billions of LLaMA queries — token efficiency is infrastructure-critical |
| Task & grader quality | 25% | 15 tasks; explicit easy/medium/hard; ROUGE-L is objective and reproducible; each task has a distinct token_budget |
| Reward function design | ~25% | Two-objective reward (quality delta − token penalty); hard budget termination; success bonus; no-op and stuck penalties |
| Runtime correctness | Pass/fail | All checks listed in IMPLEMENTATION_PLAN.md Phase 9 |
| Interface compliance | Pass/fail | Full OpenEnv spec, valid openenv.yaml, typed Pydantic v2 models |

---

## 4. Core Concept: Token-Aware Reward

### 4.1 Token counting

Token count is computed as `len(prompt.split())` — word-level approximation. This requires no external library (no tiktoken), keeps Docker image small, and is consistent across all comparisons. The approximation is sufficient because we care about relative change, not exact BPE count.

### 4.2 Reward formula

```
token_overhead(t) = token_count(prompt_t) - token_count(prompt_{t-1})

quality_delta(t)  = ROUGE_L(output_t, reference) - ROUGE_L(output_{t-1}, reference)

raw_reward(t)     = quality_delta(t) - α × token_overhead(t)

reward(t)         = clip(raw_reward(t), -2.0, +2.0)

where α = TOKEN_PENALTY_ALPHA (default: 0.02)
```

### 4.3 Why α = 0.02 as default

At α=0.02, adding 10 tokens costs 0.2 reward units. A typical quality improvement of +0.10 ROUGE-L is worth 0.10 reward units. This means adding more than 5 tokens is only justified if it improves quality by more than 0.10 — a reasonable trade-off that creates genuine tension without making token cost dominate. α is configurable so researchers can study different cost regimes.

### 4.4 Token budget enforcement

Each task has a `token_budget` (hard ceiling). If `token_count(new_prompt) > task.token_budget`, the action is rejected: the prompt reverts, reward = −0.5 (budget violation penalty), done = True. This teaches the agent to plan its token usage across the episode, not just locally.

---

## 5. All Features — Round 1

### F1: reset()
- Randomly selects one of 15 tasks (or fixed by TASK_SEED)
- Sets `current_prompt` = task's deliberately bad initial prompt
- Computes initial ROUGE-L score and initial token count
- Returns `PromptObservation` with all fields, `step_count=0`, `done=False`, `reward=0.0`

### F2: Action Space — 6 actions (5 editing + 1 terminal)

| ID | Name | Effect on Quality | Effect on Tokens | Description |
|---|---|---|---|---|
| 0 | ADD_CONTEXT | +medium | +10–15 tokens | Appends domain context sentence from task bank |
| 1 | SHORTEN | −small or neutral | −5–12 tokens | Removes filler phrases via regex; the only token-reducing action |
| 2 | ADD_EXAMPLE | +medium-high | +12–20 tokens | Appends example output format from task bank |
| 3 | REPHRASE | +small | ±0 tokens | Converts question phrasing to direct imperative (net-neutral token cost) |
| 4 | ADD_CONSTRAINT | +small-medium | +8–12 tokens | Appends output format constraint from task bank |
| 5 | STOP | none | 0 tokens | Agent voluntarily ends episode; receives current_score as final bonus |

Action 5 (STOP) is the key new mechanic. It lets the agent decide "I have done enough improvement for the tokens spent." If the agent STOPs with a good quality/cost ratio, it earns a stop bonus. This teaches the agent the value of efficiency.

### F3: STOP Action Reward

```
If action_id == 5 (STOP):
    efficiency_score = current_score / max(1, tokens_used_total)
    stop_bonus = current_score × 1.5   (reward quality at stop time)
    reward = stop_bonus
    done = True
```

### F4: Grader — Two modes

**`GRADER=rouge` (default)**
Uses DUMMY_OUTPUTS[task_id] — pre-written canned responses. No API call. Always works. Used by programmatic checker.

**`GRADER=openai_client`**
Calls LLM via `openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN).chat.completions.create(model=MODEL_NAME, ...)`. Falls back to dummy on any error.

### F5: Observation — PromptObservation (all fields always present)

| Field | Type | Description |
|---|---|---|
| task_description | str | What the prompt should accomplish |
| current_prompt | str | Prompt string after this step |
| previous_prompt | str | Prompt string before this step |
| current_score | float 0–1 | ROUGE-L F1 of current prompt output |
| previous_score | float 0–1 | ROUGE-L F1 of previous step |
| current_token_count | int | Word-level token count of current prompt |
| previous_token_count | int | Word-level token count of previous prompt |
| token_budget | int | Max allowed token count for this task |
| tokens_remaining | int | token_budget − current_token_count |
| token_overhead | int | Tokens added this step (negative if SHORTEN applied) |
| reward | float | Clipped combined reward |
| done | bool | True if episode ended |
| step_count | int | Steps taken this episode |
| reference_answer | str | Gold-standard answer used by grader |
| info | dict | grader_used, action_applied, stuck_count, termination_reason, llm_output_preview |

### F6: State — state()
Returns OpenEnv `State` with: `episode_id` (UUID), `step_count` (int)

### F7: Task Bank — 15 Tasks

Each task has: `task_id`, `category`, `difficulty`, `task_description`, `initial_bad_prompt`, `reference_answer`, `example_output`, `context_sentence`, `constraint_sentence`, `token_budget`

Token budgets are set per difficulty:
- Easy tasks: token_budget = 80
- Medium tasks: token_budget = 65
- Hard tasks: token_budget = 55

Tighter budgets on harder tasks because they require more precise language, rewarding agents that learn conciseness.

| ID | Category | Description | Difficulty | Token Budget |
|---|---|---|---|---|
| 0 | Summarisation | Climate change article → 3 bullet points | Easy | 80 |
| 1 | Summarisation | Romeo and Juliet → exactly 2 sentences | Easy | 80 |
| 2 | Summarisation | Crypto investment risks → under 60 words | Medium | 65 |
| 3 | Summarisation | French Revolution → chronological bullets | Medium | 65 |
| 4 | Summarisation | Machine learning → explain to a 10-year-old | Easy | 80 |
| 5 | QA | Time complexity of binary search and why? | Medium | 65 |
| 6 | QA | What causes inflation and how does central bank control it? | Medium | 65 |
| 7 | QA | Difference between RAM and ROM? | Easy | 80 |
| 8 | QA | Why is sky blue by day and red at sunset? | Easy | 80 |
| 9 | Instruction | Step-by-step instructions to make tea | Easy | 80 |
| 10 | Instruction | Set up Python virtual environment on Windows | Medium | 65 |
| 11 | Instruction | Resolve a Git merge conflict | Hard | 55 |
| 12 | Code | Python list comprehension with example | Medium | 65 |
| 13 | Code | Big O notation with code example | Medium | 65 |
| 14 | Code | What is recursion with Python example | Easy | 80 |

### F8: Termination Conditions

| Condition | Reward | done |
|---|---|---|
| STOP action (voluntary) | current_score × 1.5 | True |
| Budget exceeded (token_count > token_budget) | −0.5 | True |
| Stuck (same action 3 times in a row) | −0.5 | True |
| Max steps reached (step_count = MAX_STEPS = 7) | normal reward | True |
| Success (ROUGE_L > 0.85) | reward += +1.0 bonus | True |
| No-op (action has no effect) | −0.1 | False |

Note: MAX_STEPS is 7 (not 5) in PromptRL. The STOP action means episodes may be shorter, but agents need more steps to explore the quality/cost trade-off space.

### F9: inference.py (Mandatory)
- At repo root, named exactly `inference.py`
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from `os.environ`
- Uses `openai.OpenAI` client exclusively
- Runs a random agent on 3 tasks (easy, medium, hard), 7 steps each
- Tracks and prints quality score, token count, and combined reward per step
- Prints final summary with quality/cost efficiency ratio
- Exits cleanly (code 0), handles all exceptions
- Completes in under 20 minutes

### F10: Docker
- `server/Dockerfile` uses `openenv-base:latest`
- Image size ~400 MB (no local model weights)
- Builds with `docker build` from repo root with no extra steps
- All env vars passed at runtime

---

## 6. Out of Scope — Round 1

- Training an actual RL agent (env is the training ground, agents are external)
- Frontend UI or dashboard of any kind
- Database or persistent storage
- Multi-agent setup
- Local model inference (no transformers, no llama.cpp)
- Paid LLM APIs (no GPT-4, Claude, Gemini)
- Exact BPE tokenisation (word-split approximation is sufficient)
- Round 2 features

---

## 7. Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | YES | — | OpenAI-compatible endpoint. HF: `https://api-inference.huggingface.co/v1/` |
| `MODEL_NAME` | YES | — | Model ID. E.g. `mistralai/Mistral-7B-Instruct-v0.2` |
| `HF_TOKEN` | YES | — | HuggingFace token |
| `MAX_STEPS` | No | `7` | Max steps per episode |
| `DONE_THRESHOLD` | No | `0.85` | ROUGE-L above which episode ends with success bonus |
| `TOKEN_PENALTY_ALPHA` | No | `0.02` | Cost penalty coefficient α in reward formula |
| `GRADER` | No | `rouge` | `rouge` or `openai_client` |
| `TASK_SEED` | No | random | Integer 0–14, fixes task for reproducibility |
| `ENABLE_WEB_INTERFACE` | No | `false` | Gradio debug UI at /web |

---

## 8. Hard Rules (Non-Negotiable)

1. `inference.py` lives at repo root — never move it
2. All LLM calls use `openai.OpenAI` — no raw httpx/requests to LLM endpoints
3. `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` set as HF Spaces secrets
4. All grader scores in [0.0, 1.0] — enforced by clipping
5. inference.py completes in under 20 minutes
6. Docker must build from `docker build` with no extra manual steps
7. **Submission deadline: April 7, 2026, 11:59 PM IST**

---

## 9. Round 2 Extensions (Out of Scope Now)

- Multi-objective Pareto front tracking (quality vs. cost frontier per task category)
- Exact token counting via tiktoken (replaces word-split approximation)
- Adaptive α: cost penalty increases as tokens_remaining decreases (dynamic budget pressure)
- LLM-as-judge grader replacing ROUGE-L
- ADD_COT action (appends "Let's think step by step:") with its high token cost made explicit
- Per-task difficulty progression: agent unlocks harder tasks as it achieves good efficiency ratios
- Multi-task episode: 3 tasks, shared token budget across all 3
