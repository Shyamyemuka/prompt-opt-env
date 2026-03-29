# FRONTEND_GUIDELINES.md — Interface Guidelines
# Project: PromptOptEnv
# Cross-reference: PRD.md Section 4 (Non-Goals), TECH_STACK.md

---

## Important Note

PromptOptEnv has NO traditional frontend. It is a server-side RL environment. However, OpenEnv provides an optional built-in web interface (Gradio-based) that is exposed at `/web` when `ENABLE_WEB_INTERFACE=true`. This document defines how that interface should look and behave, and also defines the standards for all text output that judges will see (README, docstrings, logs).

---

## 1. OpenEnv Web Interface (Gradio, optional debug tool)

This is NOT built by us — it is generated automatically by `create_app()` when `ENABLE_WEB_INTERFACE=true`. However, understanding it helps during debugging.

When enabled, the interface at `http://localhost:8000/web` shows:
- Left pane: Human Agent — input fields for action selection, submit button
- Right pane: State observation — current prompt, score, step count, reward

To enable during development only:
```bash
ENABLE_WEB_INTERFACE=true uv run server
```

Do NOT enable in production HF Spaces deployment — it adds Gradio dependency and slows the container.

---

## 2. Terminal / Log Output Standards

All `print()` and logging calls must follow this format so that during demo/debugging, output is readable.

### Log format
```
[PromptOptEnv] [RESET] episode_id=abc123 task_id=3 initial_score=0.12
[PromptOptEnv] [STEP 1] action=REPHRASE new_score=0.28 reward=+0.16 done=False
[PromptOptEnv] [STEP 2] action=ADD_EXAMPLE new_score=0.51 reward=+0.23 done=False
[PromptOptEnv] [GRADER] called HF API — model=Mistral-7B status=200 time=1.2s
[PromptOptEnv] [GRADER] ROUGE-L computed — score=0.51
[PromptOptEnv] [DONE] reason=max_steps total_reward=0.71 final_score=0.83
```

### Colours in terminal (use only if output is a TTY)
```python
import sys
USE_COLOR = sys.stdout.isatty()

GREEN  = "\033[92m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RED    = "\033[91m" if USE_COLOR else ""
RESET  = "\033[0m"  if USE_COLOR else ""

# Usage:
print(f"{GREEN}[PromptOptEnv] [RESET]{RESET} episode started")
print(f"{YELLOW}[PromptOptEnv] [GRADER]{RESET} calling HF API...")
print(f"{RED}[PromptOptEnv] [WARN]{RESET} HF API failed, falling back to rouge")
```

---

## 3. README.md Presentation Standards

The README is a first-class deliverable because the LLM scoring judge reads it. It must be clear, well-structured, and professional.

### Required sections in README (in order):
1. Title + one-line description
2. Environment overview (what the agent does, what the reward is)
3. Quick start (exact commands to install and run)
4. Action space table (all 5 actions with descriptions)
5. Observation space table (all fields with types)
6. Reward function explanation (formula + special cases)
7. Task bank overview (categories + count)
8. Configuration (all env vars with defaults)
9. Example training loop (Python code snippet)
10. Deployment (how to push to HF Spaces)
11. Round 2 roadmap (what will be added)

### Writing style for README:
- Short sentences. One idea per sentence.
- Active voice. "The agent takes an action" not "An action is taken by the agent."
- Use tables for structured data (action space, obs space, config).
- Use code blocks for all commands and code snippets.
- Do NOT use vague terms like "state-of-the-art" or "powerful."
- Be precise: "ROUGE-L F1 score" not "a score."

---

## 4. Code Style Standards

Because code quality is evaluated by Meta engineers, all Python code must follow these rules:

### Naming conventions
```python
# Classes: PascalCase
class PromptOptEnvironment(Environment):
class PromptAction(Action):
class PromptObservation(Observation):

# Functions and methods: snake_case
def reset(self) -> PromptObservation:
def _apply_action(self, action_id: int, prompt: str) -> str:

# Constants: UPPER_SNAKE_CASE
MAX_STEPS = 5
DONE_THRESHOLD = 0.85
TASK_BANK: list[dict] = [...]

# Variables: snake_case
current_score: float
task_description: str
```

### Type hints — mandatory on all functions
```python
# Every function must have full type hints
def score(self, prompt: str, reference: str) -> float:
def add_context(prompt: str, task_description: str) -> str:
def clip_reward(reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
```

### Docstrings — mandatory on all classes and public methods
```python
class PromptOptEnvironment(Environment):
    """
    OpenEnv RL environment for prompt optimization.

    The agent iteratively edits a prompt to maximize the quality
    of an LLM's output, measured by ROUGE-L against a reference answer.

    Args:
        max_steps: Maximum actions per episode. Default: 5.
        grader: Grading strategy. One of 'rouge', 'hf_api'. Default: 'rouge'.
        hf_token: HuggingFace token for hf_api grader. Optional.
    """
```

### Maximum line length: 100 characters
### Indentation: 4 spaces (no tabs)
### Blank lines: 2 between top-level definitions, 1 between methods

---

## 5. Error Messages — User-facing strings

All error messages must be descriptive. No bare `raise ValueError("bad")`.

```python
# Bad
raise ValueError("bad action")

# Good
raise ValueError(
    f"Invalid action_id={action_id}. "
    f"Must be an integer in {{0, 1, 2, 3, 4}}. "
    f"Received type={type(action_id).__name__}."
)
```

---

## 6. openenv.yaml — Presentation

The manifest is also read by the LLM judge. The `description` field must be a clear, complete English paragraph:

```yaml
description: >
  PromptOptEnv is a Reinforcement Learning environment built on Meta's OpenEnv framework.
  The agent observes a task description and a current prompt string, then takes one of five
  deterministic editing actions (add context, shorten, add example, rephrase, add constraint)
  to improve the prompt. After each action, the environment scores the improved prompt's output
  using ROUGE-L against a reference answer and returns the score delta as the reward signal.
  Episodes run for a maximum of 5 steps or terminate early on success (score > 0.85).
  This environment is designed to train agents to perform automatic prompt engineering,
  directly relevant to LLM post-training and RLHF research.
```

---

## 7. Observation JSON — What judges see

When the programmatic checker calls your env, the JSON it receives must be clean and complete. Example of what a valid step response looks like:

```json
{
  "observation": {
    "task_description": "Summarise the plot of Romeo and Juliet in exactly 2 sentences",
    "current_prompt": "Summarise Romeo and Juliet. Example output format: [2 concise sentences]. Requirement: Exactly 2 sentences.",
    "previous_prompt": "Summarise Romeo and Juliet. Example output format: [2 concise sentences].",
    "current_score": 0.72,
    "previous_score": 0.51,
    "reward": 0.21,
    "done": false,
    "step_count": 3,
    "reference_answer": "Romeo and Juliet is a tragedy about two young lovers from feuding families who die for each other. Their deaths ultimately reconcile their families.",
    "info": {
      "grader_used": "rouge",
      "action_applied": "ADD_CONSTRAINT",
      "stuck_count": 0,
      "llm_output_preview": "Romeo and Juliet follows two young lovers from rival families..."
    }
  },
  "reward": 0.21,
  "done": false
}
```

Every field must always be present — no optional fields that might be missing. Judges' programmatic checks will access these fields by name.
