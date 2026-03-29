# PromptOptEnv

An OpenEnv Reinforcement Learning environment where an agent learns to iteratively improve prompts to maximise the quality of an LLM's output.

Built for the **Meta x Scaler OpenEnv Hackathon 2026** using the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework by Meta PyTorch and Hugging Face.

---

## What This Environment Does

The agent is given a task description (e.g. "Summarise Romeo and Juliet in 2 sentences") and a deliberately bad starting prompt (e.g. "tell me about romeo and juliet"). Over up to 5 steps, it takes editing actions to improve the prompt. After each action, the environment scores the improved prompt's output against a reference answer using ROUGE-L. The score delta becomes the reward signal.

This directly addresses a real problem in LLM post-training: teaching models to automatically engineer better prompts for a given task — without a human in the loop.

---

## Quick Start

```bash
# 1. Install
pip install openenv-core
git clone https://github.com/{username}/prompt-opt-env.git
cd prompt-opt-env/prompt_opt_env
pip install -e .

# 2. Run server locally
uv run server
# Server starts at http://localhost:8000

# 3. Connect and run an episode
python -c "
import asyncio
from prompt_opt_env import PromptOptEnv, PromptAction
import random

async def main():
    async with PromptOptEnv(base_url='ws://localhost:8000') as env:
        result = await env.reset()
        print('Task:', result.observation.task_description)
        while not result.done:
            result = await env.step(PromptAction(action_id=random.randint(0,4)))
            print(f'Step {result.observation.step_count} | reward={result.reward:.3f} | done={result.done}')

asyncio.run(main())
"
```

---

## Or Use the Live HF Spaces Deployment

```python
from prompt_opt_env import PromptOptEnv, PromptAction

async with PromptOptEnv(base_url="wss://{username}-prompt-opt-env.hf.space") as env:
    result = await env.reset()
```

---

## Action Space

The agent can take one of 5 deterministic prompt-editing actions per step. All actions are pure string transformations — no LLM is called inside the action itself.

| Action ID | Name | What it does |
|---|---|---|
| 0 | `ADD_CONTEXT` | Appends a domain-relevant context sentence to the prompt |
| 1 | `SHORTEN` | Removes filler phrases ("please", "could you") and redundant clauses |
| 2 | `ADD_EXAMPLE` | Appends an example of the desired output format |
| 3 | `REPHRASE` | Rewrites questions and passive phrasing as direct imperatives |
| 4 | `ADD_CONSTRAINT` | Appends an explicit constraint on the output (format, length, style) |

---

## Observation Space

Every call to `reset()` and `step()` returns a `PromptObservation` with all of the following fields:

| Field | Type | Description |
|---|---|---|
| `task_description` | `str` | English description of what the prompt should accomplish |
| `current_prompt` | `str` | The prompt string after this step's action |
| `previous_prompt` | `str` | The prompt string before this step's action |
| `current_score` | `float` | ROUGE-L F1 score of current prompt's output vs reference (0.0–1.0) |
| `previous_score` | `float` | ROUGE-L F1 score before this step |
| `reward` | `float` | `current_score − previous_score`, clipped to [−1.0, +2.0] |
| `done` | `bool` | `True` if episode has ended |
| `step_count` | `int` | Number of steps taken so far |
| `reference_answer` | `str` | Gold-standard answer used by the grader |
| `info` | `dict` | Extra info: `grader_used`, `action_applied`, `stuck_count`, `llm_output_preview` |

---

## Reward Function

```
reward(step t) = clip(score(t) − score(t−1), −1.0, +2.0)

where:
  score(t)   = ROUGE-L F1 between reference_answer and LLM_output(current_prompt)
  score(t−1) = ROUGE-L F1 from the previous step (or initial bad prompt at t=0)

Special cases:
  No-op (action has no effect on prompt):  reward = −0.1
  Stuck (same action 3 times in a row):    reward = −0.5, done = True
  Success (score > 0.85):                  reward += +1.0 bonus, done = True
  Max steps reached (step_count = 5):      done = True, no extra penalty
```

The reward is dense — the agent gets a signal at every step, not just at the end. This makes the environment suitable for standard policy gradient methods (PPO, GRPO).

---

## Task Bank

15 pre-written tasks across 4 categories. Each task has a deliberately bad initial prompt, a gold-standard reference answer, and pre-written context/example/constraint strings used by the action functions.

| Category | Count | Example |
|---|---|---|
| Summarisation | 5 | "Summarise the French Revolution timeline in chronological bullet points" |
| Question Answering | 4 | "What is the time complexity of binary search and why?" |
| Instruction Following | 3 | "Write step-by-step instructions to make a cup of tea" |
| Code Explanation | 3 | "Explain what recursion is with a simple Python example" |

Task is randomly selected at each `reset()`. Set `TASK_SEED` environment variable to fix the task for reproducibility.

---

## Configuration

All configuration is via environment variables. No code changes needed.

| Variable | Default | Description |
|---|---|---|
| `MAX_STEPS` | `5` | Maximum steps per episode |
| `DONE_THRESHOLD` | `0.85` | ROUGE-L score above which episode terminates with success bonus |
| `GRADER` | `rouge` | `rouge` (deterministic) or `hf_api` (calls real LLM) |
| `HF_TOKEN` | `""` | HuggingFace token (required for `hf_api` grader) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Model for HF Inference API calls |
| `TASK_SEED` | `""` | Fix task selection (integer 0–14) |
| `ENABLE_WEB_INTERFACE` | `false` | Enable Gradio debug UI at `/web` |

---

## Example Training Loop (GRPO / TRL compatible)

```python
import asyncio
from prompt_opt_env import PromptOptEnv, PromptAction

# This is the standard loop an RL trainer (TRL, torchforge, SkyRL) would use.
# The env returns rewards that can drive policy gradient updates.

async def collect_episode(env_url: str) -> list[dict]:
    """Collect one episode of (observation, action, reward) tuples."""
    trajectory = []
    async with PromptOptEnv(base_url=env_url) as env:
        result = await env.reset()
        while not result.done:
            # Replace with your policy's action selection
            action = PromptAction(action_id=your_policy(result.observation))
            result = await env.step(action)
            trajectory.append({
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
            })
    return trajectory

# Run with: asyncio.run(collect_episode("wss://your-space.hf.space"))
```

---

## OpenEnv API

This environment implements the full OpenEnv specification (RFC 002):

```
reset()  → PromptObservation   # Start new episode
step(action) → PromptObservation  # Apply action, get reward
state()  → State               # Get episode metadata
```

WebSocket-based (persistent session). Also supports HTTP endpoints at `/reset`, `/step`, `/state`.

---

## Project Structure

```
prompt_opt_env/
├── __init__.py              # Public exports
├── models.py                # PromptAction, PromptObservation, PromptState
├── client.py                # PromptOptEnv WebSocket client
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
└── server/
    ├── app.py               # FastAPI app
    ├── prompt_opt_environment.py  # Core RL logic
    ├── actions.py           # 5 prompt editing actions
    ├── grader.py            # ROUGE + HF API grader
    ├── task_bank.py         # 15 tasks with reference answers
    └── Dockerfile
```

---

## Running Tests

```bash
cd prompt_opt_env
pip install -e ".[dev]"
python -m pytest ../tests/ -v
```

---

## Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --repo-id {your_username}/prompt-opt-env
```

Then set `HF_TOKEN` as a secret in your Space settings.

---

## Roadmap (Round 2 — Bangalore Finals)

- LLM-as-judge grader (replace ROUGE with an LLM quality scorer)
- Chain-of-thought action (`ADD_COT`: appends "Let's think step by step:")
- Multi-task episodes (agent optimises 3 prompts per episode)
- Curriculum difficulty (easy → medium → hard tasks based on agent performance)
- Multi-turn optimisation (agent sees actual LLM output before next action)

---

## Author

Shyam — B.Tech Data Science, Annamacharya University, Rajampet, AP, India
Submitted solo to the Meta x Scaler OpenEnv Hackathon 2026.

---

## License

BSD-3-Clause (same as OpenEnv)
