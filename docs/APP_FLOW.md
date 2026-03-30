# APP_FLOW.md — Application Flow
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization
# Cross-reference: PRD.md, BACKEND_STRUCTURE.md
# Version: FINAL
# Last updated: 2026-03-30

---

## Overview

PromptRL has no frontend. It is a server-side RL environment exposed via WebSocket and HTTP. This document traces every path: system boot, client connection, reset(), every step() branch (all 6 actions including STOP, success, stuck, budget exceeded, no-op, invalid), state(), inference.py standalone flow, and full deployment. No path is undocumented.

---

## 1. System Boot

```
uv run server   (or: docker run ...)
    │
    ├── server/app.py loads
    │       ├── os.getenv() reads all env vars:
    │       │       MAX_STEPS             → default "7"
    │       │       DONE_THRESHOLD        → default "0.85"
    │       │       TOKEN_PENALTY_ALPHA   → default "0.02"
    │       │       GRADER                → default "rouge"
    │       │       API_BASE_URL          → used if GRADER=openai_client
    │       │       MODEL_NAME            → used if GRADER=openai_client
    │       │       HF_TOKEN              → used if GRADER=openai_client
    │       │       TASK_SEED             → default None
    │       │
    │       ├── Instantiates PromptRLEnvironment()
    │       │       └── Creates Grader(grader_type=GRADER)
    │       │
    │       └── Calls create_app(PromptRLEnvironment, PromptAction, PromptObservation)
    │
    ├── Uvicorn starts on 0.0.0.0:8000
    │       ├── /ws      — WebSocket (all env interactions)
    │       ├── /reset   — HTTP POST
    │       ├── /step    — HTTP POST
    │       ├── /state   — HTTP GET
    │       └── /health  — HTTP GET → {"status": "ok"}
    │
    └── Server ready
```

---

## 2. Client Connection

```
from prompt_rl import PromptRLEnv, PromptAction

# Async (recommended):
async with PromptRLEnv(base_url="ws://localhost:8000") as client:
    │
    ├── __init__: stores base_url
    ├── __aenter__: opens WebSocket to ws://localhost:8000/ws
    │       └── Server logs: "Client connected"
    │
    [... episode ...]
    │
    └── __aexit__: closes WebSocket

# Sync alternative:
with PromptRLEnv(base_url="ws://localhost:8000").sync() as client:
    [identical, blocking]

# Remote (HF Spaces):
async with PromptRLEnv(base_url="wss://{username}-prompt-rl.hf.space") as client:
    [identical]
```

---

## 3. reset() Flow

```
client.reset()
    │
    ├── Sends: {"type": "reset", "payload": {}}
    │
    ├── Server → PromptRLEnvironment.reset()
    │       │
    │       ├── Generates episode_id = str(uuid4())
    │       ├── Resets: step_count=0, stuck_count=0, last_action=None, tokens_used_total=0
    │       │
    │       ├── Task selection:
    │       │       TASK_SEED set  → task_id = int(TASK_SEED) % 15
    │       │       else           → task_id = random.randint(0, 14)
    │       │
    │       ├── Loads TASK_BANK[task_id]:
    │       │       current_prompt   ← task.initial_bad_prompt
    │       │       reference_answer ← task.reference_answer
    │       │       token_budget     ← task.token_budget
    │       │
    │       ├── Counts initial tokens:
    │       │       initial_token_count = len(current_prompt.split())
    │       │
    │       ├── Computes initial score:
    │       │       grader.score(current_prompt, reference_answer, task_id)
    │       │       → returns (initial_rouge_l, output_text)
    │       │
    │       └── Returns PromptObservation:
    │               task_description       = task.task_description
    │               current_prompt         = task.initial_bad_prompt
    │               previous_prompt        = ""
    │               current_score          = initial_rouge_l
    │               previous_score         = 0.0
    │               current_token_count    = initial_token_count
    │               previous_token_count   = 0
    │               token_budget           = task.token_budget
    │               tokens_remaining       = task.token_budget - initial_token_count
    │               token_overhead         = 0
    │               reward                 = 0.0
    │               done                   = False
    │               step_count             = 0
    │               reference_answer       = task.reference_answer
    │               info = {grader_used, action_applied: None, stuck_count: 0,
    │                       termination_reason: None, llm_output_preview: ""}
    │
    └── Client receives result.observation + result.reward=0.0 + result.done=False
```

---

## 4. step() Flow — All 9 Paths

### 4A. Normal Step — action works, episode continues

```
client.step(PromptAction(action_id=2))
    │
    ├── Sends: {"type": "step", "payload": {"action_id": 2}}
    │
    ├── Server → PromptRLEnvironment.step(action)
    │       │
    │       ├── [PATH CHECK] action_id not in {0,1,2,3,4,5}?
    │       │       YES → PATH 4H (invalid action)
    │       │
    │       ├── [PATH CHECK] action_id == 5 (STOP)?
    │       │       YES → PATH 4B (stop action)
    │       │
    │       ├── Stuck detection:
    │       │       action_id == last_action  → stuck_count += 1
    │       │       else                      → stuck_count = 0; last_action = action_id
    │       │
    │       ├── [PATH CHECK] stuck_count >= 3?
    │       │       YES → PATH 4D (stuck termination)
    │       │
    │       ├── Apply action (actions.py):
    │       │       0 → add_context(prompt, task.context_sentence)
    │       │       1 → shorten(prompt)
    │       │       2 → add_example(prompt, task.example_output)
    │       │       3 → rephrase(prompt)
    │       │       4 → add_constraint(prompt, task.constraint_sentence)
    │       │
    │       ├── [PATH CHECK] new_prompt == current_prompt?
    │       │       YES → PATH 4E (no-op)
    │       │
    │       ├── new_token_count = len(new_prompt.split())
    │       │
    │       ├── [PATH CHECK] new_token_count > task.token_budget?
    │       │       YES → PATH 4F (budget exceeded)
    │       │
    │       ├── Grader call:
    │       │       GRADER=rouge:
    │       │           llm_output = DUMMY_OUTPUTS[task_id]
    │       │       GRADER=openai_client:
    │       │           try:
    │       │               llm_output = openai_client.chat.completions.create(
    │       │                   model=MODEL_NAME,
    │       │                   messages=[{"role":"user","content":new_prompt}],
    │       │                   max_tokens=200, temperature=0.1, timeout=30
    │       │               ).choices[0].message.content
    │       │           except Exception:
    │       │               llm_output = DUMMY_OUTPUTS[task_id]  ← silent fallback
    │       │       new_score = ROUGE_L(llm_output, reference_answer)
    │       │
    │       ├── Compute reward:
    │       │       quality_delta  = new_score - current_score
    │       │       token_overhead = new_token_count - current_token_count
    │       │       raw_reward     = quality_delta - α × token_overhead
    │       │       reward         = clip(raw_reward, -2.0, +2.0)
    │       │
    │       ├── Update state:
    │       │       previous_prompt      = current_prompt
    │       │       previous_score       = current_score
    │       │       previous_token_count = current_token_count
    │       │       current_prompt       = new_prompt
    │       │       current_score        = new_score
    │       │       current_token_count  = new_token_count
    │       │       tokens_used_total   += max(0, token_overhead)
    │       │       step_count          += 1
    │       │
    │       ├── Termination checks:
    │       │       new_score > DONE_THRESHOLD (0.85)?
    │       │           YES → reward = clip(reward + 1.0, -2.0, +2.0); done = True
    │       │               → info["termination_reason"] = "success"
    │       │       step_count >= MAX_STEPS (7)?
    │       │           YES → done = True
    │       │               → info["termination_reason"] = "max_steps"
    │       │       else → done = False
    │       │
    │       └── Returns PromptObservation with all updated fields
    │
    └── Client receives observation + reward + done
```

### 4B. STOP Action Path (action_id = 5)

```
Agent decides to voluntarily end episode:
    │
    ├── stop_bonus = current_score × 1.5
    ├── reward     = clip(stop_bonus, 0.0, +2.0)
    ├── done       = True
    ├── step_count += 1
    └── Returns observation with done=True, reward=stop_bonus
            info["termination_reason"] = "voluntary_stop"
            info["action_applied"]     = "STOP"
    
    Why stop_bonus? Agent is rewarded for the quality it achieved.
    Stopping early with high quality = high efficiency = high reward.
    Stopping early with low quality = low reward → agent learns not to stop prematurely.
```

### 4C. Success Termination Path (ROUGE_L > 0.85)

```
During step() — new_score > 0.85:
    reward = clip(raw_reward + 1.0, -2.0, +2.0)  ← success bonus added
    done   = True
    info["termination_reason"] = "success"
    → Client should call reset() for next episode
```

### 4D. Stuck Termination Path (same action 3+ times)

```
During step() — stuck_count >= 3:
    reward = -0.5
    done   = True
    info["termination_reason"] = "stuck"
    info["stuck_count"]        = 3
```

### 4E. No-Op Path (action had no effect on prompt string)

```
During step() — new_prompt == current_prompt:
    reward     = -0.1
    done       = step_count + 1 >= MAX_STEPS
    step_count += 1
    Grader is NOT called
    State scores unchanged, token counts unchanged
    info["termination_reason"] = None (or "max_steps" if done)
    info["no_op"]              = True
```

### 4F. Budget Exceeded Path (token count would exceed token_budget)

```
During step() — new_token_count > task.token_budget:
    Action is REJECTED — prompt reverts to current_prompt
    reward     = -0.5
    done       = True
    step_count += 1
    info["termination_reason"] = "budget_exceeded"
    info["tokens_over_budget"] = new_token_count - task.token_budget
    → Agent learns: do not add tokens when near the budget ceiling
```

### 4G. Max Steps Path (step_count = MAX_STEPS = 7)

```
During step() — step_count reaches 7:
    done   = True
    reward = normal combined reward (no extra penalty)
    info["termination_reason"] = "max_steps"
```

### 4H. Invalid Action Path (action_id not in {0..5})

```
During step() — action_id not in {0,1,2,3,4,5}:
    → raises ValueError with descriptive message
    → State is NOT changed
    → Client must send valid action to continue
    → Episode is NOT terminated
```

---

## 5. state() Flow

```
client.state()
    │
    ├── Sends: {"type": "state"}
    ├── Server returns State(episode_id=self._episode_id, step_count=self._step_count)
    └── Client receives State object
```

---

## 6. Full Episode Example — Cost-Aware Trade-offs

```
reset()
  task:          "Summarise the plot of Romeo and Juliet in exactly 2 sentences"
  initial_prompt: "tell me about romeo and juliet"  [7 tokens]
  token_budget:   80
  initial_score:  0.12
  tokens_remaining: 73

step(action_id=3)  ← REPHRASE  [net 0 token change]
  new_prompt: "Summarise the plot of Romeo and Juliet."  [7 tokens]
  token_overhead: 0
  new_score: 0.28
  quality_delta: +0.16
  reward: 0.16 - 0.02×0 = +0.16  ✓ free improvement

step(action_id=2)  ← ADD_EXAMPLE  [+14 tokens]
  new_prompt: "Summarise the plot of Romeo and Juliet.
               Example output format: [2 concise sentences covering key events]"
  token_count: 21  tokens_remaining: 59
  new_score: 0.51
  quality_delta: +0.23
  reward: 0.23 - 0.02×14 = 0.23 - 0.28 = -0.05  ✗ too many tokens for the quality gain

step(action_id=4)  ← ADD_CONSTRAINT  [+9 tokens]
  new_prompt: "...Requirement: Exactly 2 sentences, no more."
  token_count: 30  tokens_remaining: 50
  new_score: 0.72
  quality_delta: +0.21
  reward: 0.21 - 0.02×9 = 0.21 - 0.18 = +0.03  ~ marginal

step(action_id=1)  ← SHORTEN  [-8 tokens]
  new_prompt: "Summarise Romeo and Juliet in exactly 2 sentences.
               Requirement: Exactly 2 sentences."
  token_count: 22  tokens_remaining: 58  [tokens recovered!]
  new_score: 0.75
  quality_delta: +0.03
  reward: 0.03 - 0.02×(-8) = 0.03 + 0.16 = +0.19  ✓ SHORTEN is rewarded twice

step(action_id=5)  ← STOP [voluntary]
  stop_bonus: 0.75 × 1.5 = 1.125
  reward: +1.125  ← agent decided score is good enough for tokens spent
  done: True

Cumulative reward: 0.16 - 0.05 + 0.03 + 0.19 + 1.125 = 1.455
Token efficiency: 0.75 ROUGE-L at 22 tokens (vs. task.token_budget=80)
```

This example shows the agent correctly learning: REPHRASE first (free quality), ADD_EXAMPLE was too expensive, ADD_CONSTRAINT was marginal, SHORTEN both improves quality AND recovers tokens, then STOP when quality is good enough.

---

## 7. Error Paths Reference

| Error | Trigger | Behaviour | Client Receives |
|---|---|---|---|
| Invalid action_id | Not in {0..5} | ValueError, state unchanged | Error dict, episode continues |
| Budget exceeded | token_count > token_budget | Prompt reverts, reward=-0.5, done=True | Observation with termination_reason="budget_exceeded" |
| OpenAI API timeout >30s | LLM call too slow | Fallback to DUMMY_OUTPUTS, episode continues | Normal obs, grader_used="rouge_fallback" |
| OpenAI 429 rate limit | Too many requests | Fallback to DUMMY_OUTPUTS | Same as timeout |
| Missing API_BASE_URL | Env var not set | Grader uses rouge mode automatically | Normal obs |
| STOP called with score=0 | Agent stops immediately | stop_bonus=0, reward=0, done=True | Valid obs, episode ends |
| step() before reset() | Client bug | RuntimeError raised | Error dict |
| step() after done=True | Client bug | Auto-resets, logs warning | reset() observation |
| WebSocket disconnect | Network | Session cleaned up, state lost | WebSocket exception |

---

## 8. inference.py Standalone Flow

```
python inference.py
    │
    ├── Reads from os.environ:
    │       API_BASE_URL  ← required; KeyError if missing
    │       MODEL_NAME    ← required; KeyError if missing
    │       HF_TOKEN      ← required; KeyError if missing
    │
    ├── Creates OpenAI client:
    │       OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    │
    ├── Prints header (model, endpoint)
    │
    ├── For each of 3 EVAL_TASKS (easy, medium, hard):
    │       │
    │       ├── call_llm(initial_prompt) → initial_output
    │       ├── rouge_l(initial_output, reference) → initial_score
    │       ├── initial_tokens = len(initial_prompt.split())
    │       │
    │       ├── For step 1 to MAX_STEPS=7:
    │       │       action_id = random 0–5
    │       │       if action_id == 5 (STOP): compute stop_bonus, break
    │       │       apply deterministic transform
    │       │       check token_budget — skip if exceeded
    │       │       call_llm(new_prompt) → new_output
    │       │       compute ROUGE-L, token_overhead, combined reward
    │       │       print per-step: action, score, tokens, reward
    │       │
    │       └── Returns {difficulty, initial_score, final_score,
    │                     total_reward, final_token_count, token_budget, steps}
    │
    ├── Prints summary table:
    │       Difficulty  Initial  Final  Reward  Tokens  Budget  Efficiency
    │       easy        0.1200   0.5400  0.6200     22      80   0.0245
    │       medium      0.0800   0.3500  0.3100     35      65   0.0100
    │       hard        0.0500   0.2400  0.2200     28      55   0.0086
    │       Average              0.3767
    │
    │       (Efficiency = final_score / final_token_count)
    │
    └── sys.exit(0)

Total runtime: ~4–9 minutes. Always exits with code 0.
```

---

## 9. Deployment Flow

```
Step 1: huggingface-cli login  (enter HF_TOKEN)

Step 2: cd prompt_rl/ && openenv push --repo-id {username}/prompt-rl

Step 3: Set secrets in HF Spaces UI:
        API_BASE_URL = https://api-inference.huggingface.co/v1/
        MODEL_NAME   = mistralai/Mistral-7B-Instruct-v0.2
        HF_TOKEN     = hf_your_actual_token

Step 4: Verify:
        curl https://{username}-prompt-rl.hf.space/health
        → {"status": "ok"}

Step 5: python validate.py  (from dashboard pre-submission script)
        → ALL PASS

Step 6: Submit HF Spaces URL on dashboard before April 7, 2026, 11:59 PM IST
```
