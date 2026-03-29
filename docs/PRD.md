# PRD.md — Product Requirements Document
# Project: PromptOptEnv — LLM Prompt Optimizer RL Environment
# Hackathon: Meta x Scaler OpenEnv Hackathon 2026
# Author: Shyam | Solo submission
# Last updated: 2026-03-29

---

## 1. Overview

PromptOptEnv is a Reinforcement Learning environment built on Meta's OpenEnv framework. It creates a training ground where an AI agent learns to iteratively improve prompts sent to a language model, maximising the quality of the model's output. The agent is rewarded based on how much better the output becomes after each prompt-editing action. This is submitted as a Mini-RL environment for Round 1 of the Meta x Scaler hackathon.

The core idea: give an agent a bad prompt, let it take editing actions, call a real LLM with the improved prompt, score the output, give reward. Repeat. The agent learns prompt engineering.

---

## 2. Problem Statement

Prompt engineering is one of the most impactful yet manual and unintuitive tasks in applied AI. Small changes in phrasing produce dramatically different LLM outputs. This environment formalises that process as an RL problem so that an agent can be trained to do it automatically, and more effectively than a human doing it manually.

This is directly relevant to Meta's own post-training research: training models to improve their own instructions and reasoning traces is a core open problem in the agentic AI space.

---

## 3. Goals

### Round 1 Goals (by April 8, 2026)
- G1: Build a fully functional, OpenEnv-compliant RL environment
- G2: Implement all 5 prompt-editing action types
- G3: Implement a deterministic local grader (ROUGE-based) that works without any external API dependency
- G4: Implement an optional HuggingFace Inference API grader for richer reward signal
- G5: Containerise the environment with Docker so it passes programmatic checks
- G6: Deploy to Hugging Face Spaces via `openenv push`
- G7: Write thorough README and openenv.yaml that score well on LLM evaluation

### Round 2 Goals (April 25–26, Bangalore)
- G8: Add LLM-as-judge grader (replace ROUGE with a judge LLM call)
- G9: Add multi-task episode support (agent faces different task types)
- G10: Add prompt chaining across multiple turns
- G11: Add meta-learning across prompt styles (few-shot vs zero-shot vs chain-of-thought)
- G12: Add curriculum difficulty: easy tasks → hard tasks based on agent performance

---

## 4. Non-Goals (OUT OF SCOPE for Round 1)

- NO training an actual RL agent inside the environment (the env is the training ground; the agent is external)
- NO frontend UI or dashboard
- NO authentication or user accounts
- NO persistent storage or database
- NO multi-agent setup
- NO support for closed-source models (GPT-4, Claude, Gemini) — free HF Inference API only
- NO render() method (not required by OpenEnv spec RFC 002)
- NO seed() method (not required for Round 1)
- NO Kubernetes deployment
- NO streaming output from LLM
- NO custom tokenizer integration
- NO web scraping or external data fetching during episodes

---

## 5. Users / Stakeholders

| User | Role | What they do with this env |
|---|---|---|
| RL researcher / framework | Primary user | Plugs env into GRPO / TRL / torchforge training loop |
| Hackathon judge (programmatic) | Automated evaluator | Runs reset(), step(), checks reward is numeric, checks done flag |
| Hackathon judge (LLM scoring) | Human/LLM evaluator | Reads README, openenv.yaml, code to assess quality of env design |
| Shyam (developer) | Builder + presenter | Submits Round 1, presents in Round 2 |

---

## 6. Core Features — Round 1

### F1: Environment Reset
- `reset()` returns an initial `PromptObservation`
- Initial observation contains: original task description, current prompt (starts as a deliberately bad/vague prompt), current score (0.0), step count (0), episode ID, done=False
- Episode is randomly sampled from a bank of 15 pre-written task-prompt pairs (see Section 8)
- Seed parameter optionally fixes the task for reproducibility

### F2: Action Space (5 actions)
All actions operate on the current prompt string and return a new prompt string deterministically. No LLM call happens inside actions — they are pure string transformations.

| Action ID | Action Name | What it does |
|---|---|---|
| 0 | ADD_CONTEXT | Appends a sentence providing domain context to the prompt |
| 1 | SHORTEN | Removes filler words and redundant clauses, makes the prompt more concise |
| 2 | ADD_EXAMPLE | Appends a concrete example of the desired output format |
| 3 | REPHRASE | Rewrites the prompt in more direct, imperative language |
| 4 | ADD_CONSTRAINT | Appends an explicit constraint on output format (e.g. "in under 50 words") |

Each action is applied by a deterministic Python function — no randomness, no external calls.

### F3: Grader (Reward Computation)
Two graders, selectable via environment variable:

**Grader A: ROUGE (default, no API needed)**
- Takes the LLM output produced from the current prompt
- Compares to a reference answer stored per task
- Computes ROUGE-L F1 score using the `rouge-score` Python library
- Reward = ROUGE-L(current) − ROUGE-L(previous step)
- First step: reward = ROUGE-L(current) − ROUGE-L(initial bad prompt output)
- Reward is clipped to range [−1.0, +1.0]

**Grader B: HF Inference API + ROUGE hybrid (optional, requires HF_TOKEN env var)**
- Actually calls a small HF-hosted model (Mistral-7B-Instruct via serverless API) with the current prompt
- Scores output with ROUGE-L against reference answer
- If API call fails (rate limit, timeout), falls back to Grader A automatically

### F4: Episode Termination
An episode ends (done=True) when:
- Agent has taken MAX_STEPS = 5 actions (configurable via env var), OR
- Current ROUGE score exceeds DONE_THRESHOLD = 0.85 (early success termination), OR
- The same action is taken 3 times in a row (stuck detection, terminates with penalty reward −0.5)

### F5: Observation Structure
Every call to `reset()` and `step()` returns a `PromptObservation` containing:
- `task_description` (str): What the prompt should accomplish
- `current_prompt` (str): The prompt string after this step's action
- `previous_prompt` (str): The prompt string before this step's action
- `current_score` (float): ROUGE-L score of current prompt's output
- `previous_score` (float): ROUGE-L score before this step
- `reward` (float): current_score − previous_score, clipped [−1, +1]
- `done` (bool): whether episode is over
- `step_count` (int): how many steps taken so far
- `reference_answer` (str): the gold-standard answer for this task (for grader transparency)
- `info` (dict): extra metadata — grader_used, action_applied, stuck_count

### F6: State
- `episode_id` (str): UUID for this episode
- `step_count` (int): current step number
- `task_id` (int): which task from the bank is active

### F7: Task Bank
15 pre-written tasks. Each task has:
- `task_id` (int)
- `task_description` (str): English description of what the prompt should produce
- `initial_bad_prompt` (str): a deliberately vague or poorly-worded starting prompt
- `reference_answer` (str): a gold-standard answer used by the grader
- `difficulty` (str): easy / medium / hard

Task categories: summarisation (5 tasks), question answering (4 tasks), instruction following (3 tasks), code explanation (3 tasks).

### F8: OpenEnv Compliance
- Implements `reset()`, `step()`, `state()` exactly as per RFC 002 spec
- Uses `openenv.core.env_server.interfaces.Environment` as base class
- Uses `openenv.core.env_server` `create_app()` to expose FastAPI server
- Packaged with `openenv.yaml` manifest
- Deployable with `openenv push` to Hugging Face Spaces
- Client class `PromptOptEnv` extends `EnvClient` for WebSocket interaction

---

## 7. Reward Design — Detailed

This is the most important part of the env design and what judges will scrutinise most.

```
reward(step t) = score(t) - score(t-1)

where:
  score(t) = ROUGE-L F1 between reference_answer and LLM_output(current_prompt_at_step_t)
  score(0) = ROUGE-L F1 of initial_bad_prompt's output (computed at reset time)

Special cases:
  - If action causes prompt to become identical to previous prompt: reward = -0.1 (no-op penalty)
  - If stuck (same action 3 times): reward = -0.5, done = True
  - If score(t) > 0.85: reward = +1.0 bonus added, done = True (success bonus)

Reward is always returned as a float in observation.reward
```

Why this design is good:
- Dense reward (every step has signal) — better for training than sparse end-of-episode reward
- Differentiates between good and bad actions clearly
- Punishes repetition and no-ops to encourage exploration
- The success bonus creates a clear incentive to reach high quality efficiently

---

## 8. Task Bank (all 15 tasks specified)

| ID | Category | Description | Difficulty |
|---|---|---|---|
| 0 | Summarisation | Summarise a 200-word news article about climate change in 3 bullet points | Easy |
| 1 | Summarisation | Summarise the plot of Romeo and Juliet in exactly 2 sentences | Easy |
| 2 | Summarisation | Summarise the key risks of investing in crypto in under 60 words | Medium |
| 3 | Summarisation | Summarise the French Revolution timeline in chronological bullet points | Medium |
| 4 | Summarisation | Summarise what machine learning is for a 10-year-old | Easy |
| 5 | QA | Answer: What is the time complexity of binary search and why? | Medium |
| 6 | QA | Answer: What causes inflation and how does the central bank control it? | Medium |
| 7 | QA | Answer: What is the difference between RAM and ROM? | Easy |
| 8 | QA | Answer: Why does the sky appear blue during the day and red at sunset? | Easy |
| 9 | Instruction | Write step-by-step instructions to make a cup of tea | Easy |
| 10 | Instruction | Explain how to set up a Python virtual environment on Windows | Medium |
| 11 | Instruction | Describe the steps to resolve a Git merge conflict | Hard |
| 12 | Code | Explain what the following Python function does (provided as context) | Medium |
| 13 | Code | Explain Big O notation using a simple code example | Medium |
| 14 | Code | Explain what recursion is with a simple Python example | Easy |

---

## 9. Configuration (all via environment variables)

| Variable | Default | Description |
|---|---|---|
| `MAX_STEPS` | `5` | Max steps per episode |
| `DONE_THRESHOLD` | `0.85` | ROUGE score at which episode ends successfully |
| `GRADER` | `rouge` | `rouge` or `hf_api` |
| `HF_TOKEN` | `""` | HuggingFace token (required for grader=hf_api) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Model for HF Inference API calls |
| `TASK_SEED` | `None` | Fix task selection for reproducibility |
| `ENABLE_WEB_INTERFACE` | `false` | Enable Gradio web UI (for debugging) |

---

## 10. Success Criteria for Round 1 Submission

- [ ] `openenv init prompt_opt_env` scaffolded and files filled
- [ ] `reset()` returns valid `PromptObservation` with all fields
- [ ] `step(action)` for all 5 action types returns valid observation with numeric reward
- [ ] `state()` returns `State` with episode_id and step_count
- [ ] Episode terminates correctly (max steps, success, stuck)
- [ ] ROUGE grader computes and returns a float reward
- [ ] Docker image builds with `openenv build`
- [ ] Server starts with `uv run server`
- [ ] `openenv validate` passes
- [ ] `openenv push` deploys to HF Spaces successfully
- [ ] README.md is thorough and explains env design clearly
- [ ] `openenv.yaml` manifest is complete
- [ ] Unit tests pass for all action transformations and grader logic
