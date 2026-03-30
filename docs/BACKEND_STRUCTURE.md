# BACKEND_STRUCTURE.md — Backend Structure & Complete Code
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization
# Cross-reference: APP_FLOW.md, TECH_STACK.md, IMPLEMENTATION_PLAN.md
# Version: FINAL
# Last updated: 2026-03-30

---

## 1. Repository Structure

```
prompt-rl/                              ← GitHub repo root
│
├── inference.py                        ← MANDATORY: baseline script at repo root
├── .env.example                        ← API_BASE_URL, MODEL_NAME, HF_TOKEN template
├── .gitignore
├── README.md
│
└── prompt_rl/                          ← OpenEnv environment package
    │
    ├── __init__.py                     ← Exports: PromptAction, PromptObservation, PromptRLEnv
    ├── models.py                       ← Pydantic v2 models
    ├── client.py                       ← PromptRLEnv WebSocket client
    ├── openenv.yaml                    ← OpenEnv manifest
    ├── pyproject.toml                  ← Dependencies
    │
    ├── server/
    │   ├── __init__.py
    │   ├── app.py                      ← FastAPI app via create_app()
    │   ├── prompt_rl_environment.py    ← Core environment logic
    │   ├── actions.py                  ← 6 action functions (5 edit + STOP)
    │   ├── grader.py                   ← ROUGE + OpenAI client grader
    │   ├── task_bank.py                ← 15 tasks with token_budget
    │   ├── requirements.txt            ← Docker dependencies
    │   └── Dockerfile
    │
    └── tests/
        ├── __init__.py
        ├── test_actions.py
        ├── test_grader.py
        └── test_environment.py
```

**`inference.py` must be at the repo root — never inside `prompt_rl/`.**

---

## 2. Complete Code — Every File

### 2.1 models.py

```python
"""
Typed Pydantic v2 models for PromptRL.
All fields always present — no Optional fields that might be missing.
The programmatic checker accesses fields by name.
"""
from typing import Any
from pydantic import BaseModel, Field


class PromptAction(BaseModel):
    """
    Action taken by the agent.

    action_id:
      0: ADD_CONTEXT    — append context sentence (adds ~10-15 tokens)
      1: SHORTEN        — remove filler phrases (reduces ~5-12 tokens)
      2: ADD_EXAMPLE    — append example output (adds ~12-20 tokens)
      3: REPHRASE       — convert to direct imperative (net 0 token change)
      4: ADD_CONSTRAINT — append constraint (adds ~8-12 tokens)
      5: STOP           — voluntarily end episode (reward = current_score × 1.5)
    """
    action_id: int = Field(
        ..., ge=0, le=5,
        description="Integer 0–5. Action 5 = STOP (voluntary episode end)."
    )


class PromptObservation(BaseModel):
    """
    Full observation returned by reset() and step().
    Every field is always present. Never null.
    """
    task_description: str = Field(..., description="What the prompt should accomplish")
    current_prompt: str = Field(..., description="Prompt after this step's action")
    previous_prompt: str = Field(..., description="Prompt before this step (empty at reset)")
    current_score: float = Field(..., ge=0.0, le=1.0, description="ROUGE-L F1 of current prompt output")
    previous_score: float = Field(..., ge=0.0, le=1.0, description="ROUGE-L F1 before this step")
    current_token_count: int = Field(..., ge=0, description="Word-level token count of current prompt")
    previous_token_count: int = Field(..., ge=0, description="Word-level token count before this step")
    token_budget: int = Field(..., ge=1, description="Hard ceiling on prompt token count for this task")
    tokens_remaining: int = Field(..., description="token_budget - current_token_count")
    token_overhead: int = Field(..., description="Tokens added this step (negative if SHORTEN applied)")
    reward: float = Field(..., description="Combined reward: quality_delta - alpha*token_overhead, clipped [-2, +2]")
    done: bool = Field(..., description="True if episode ended")
    step_count: int = Field(..., ge=0, description="Steps taken this episode")
    reference_answer: str = Field(..., description="Gold-standard answer for grader")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="grader_used, action_applied, stuck_count, termination_reason, llm_output_preview, no_op"
    )
```

---

### 2.2 server/task_bank.py

```python
"""
15 tasks across 4 categories with token budgets.
Token budgets: easy=80, medium=65, hard=55.
Tighter budgets for harder tasks — requires more concise language.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Task:
    task_id: int
    category: Literal["summarisation", "qa", "instruction", "code"]
    difficulty: Literal["easy", "medium", "hard"]
    task_description: str
    initial_bad_prompt: str
    reference_answer: str
    example_output: str
    context_sentence: str
    constraint_sentence: str
    token_budget: int  # hard ceiling on prompt word count


TASK_BANK: list[Task] = [
    # ── SUMMARISATION ────────────────────────────────────────────────────────
    Task(
        task_id=0, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Summarise a news article about climate change in 3 bullet points",
        initial_bad_prompt="talk about climate",
        reference_answer=(
            "- Global temperatures have risen 1.1°C since pre-industrial times.\n"
            "- Extreme weather events are becoming more frequent and severe.\n"
            "- Nations must cut emissions by 45% by 2030 to limit warming to 1.5°C."
        ),
        example_output="- Key fact in under 20 words.\n- Impact point.\n- Action needed.",
        context_sentence="Climate change refers to long-term shifts in global temperatures caused primarily by human activities since the 1800s.",
        constraint_sentence="Exactly 3 bullet points, each under 20 words.",
    ),
    Task(
        task_id=1, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Summarise the plot of Romeo and Juliet in exactly 2 sentences",
        initial_bad_prompt="tell me about romeo and juliet",
        reference_answer=(
            "Romeo and Juliet is a tragedy about two young lovers from feuding families in Verona who secretly marry. "
            "Their deaths by suicide ultimately reconcile their families, ending the feud."
        ),
        example_output="[First sentence: setup and conflict]. [Second sentence: resolution].",
        context_sentence="Romeo and Juliet is a Shakespeare tragedy set in Verona, Italy, about feuding families.",
        constraint_sentence="Exactly 2 sentences. No more, no fewer.",
    ),
    Task(
        task_id=2, category="summarisation", difficulty="medium", token_budget=65,
        task_description="Summarise the key risks of investing in cryptocurrency in under 60 words",
        initial_bad_prompt="crypto risks",
        reference_answer=(
            "Cryptocurrency investments carry extreme price volatility, potential total loss, regulatory uncertainty, "
            "security risks from hacks, and illiquidity. Unlike traditional assets, crypto is uninsured and largely "
            "unregulated, making it unsuitable for risk-averse investors."
        ),
        example_output="Crypto risks include: [risk 1], [risk 2], and [risk 3]. Note: [key warning].",
        context_sentence="Cryptocurrency is a digital currency secured by cryptography, with Bitcoin and Ethereum as major examples.",
        constraint_sentence="Under 60 words total.",
    ),
    Task(
        task_id=3, category="summarisation", difficulty="medium", token_budget=65,
        task_description="Summarise the French Revolution timeline in chronological bullet points",
        initial_bad_prompt="french revolution summary",
        reference_answer=(
            "- 1789: Bastille stormed; National Assembly formed.\n"
            "- 1791: Constitutional monarchy established.\n"
            "- 1792: Republic declared.\n"
            "- 1793–1794: Reign of Terror.\n"
            "- 1799: Napoleon seizes power."
        ),
        example_output="- [Year]: [Key event in one line].",
        context_sentence="The French Revolution (1789–1799) was a period of radical political transformation in France.",
        constraint_sentence="Bullet points with years. At least 5 events in chronological order.",
    ),
    Task(
        task_id=4, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Explain what machine learning is to a 10-year-old",
        initial_bad_prompt="explain machine learning",
        reference_answer=(
            "Machine learning is when computers learn from examples, just like how you learned to recognise cats "
            "by seeing many cats. The computer looks at lots of data and figures out patterns by itself."
        ),
        example_output="Machine learning is like [simple child analogy]. The computer [what it does simply].",
        context_sentence="Machine learning is a type of AI where computers learn patterns from data without being explicitly programmed.",
        constraint_sentence="Simple words only. No jargon. Write for a 10-year-old.",
    ),
    # ── QA ───────────────────────────────────────────────────────────────────
    Task(
        task_id=5, category="qa", difficulty="medium", token_budget=65,
        task_description="Answer: What is the time complexity of binary search and why?",
        initial_bad_prompt="binary search complexity",
        reference_answer=(
            "Binary search has O(log n) time complexity. With each comparison it eliminates half the remaining elements. "
            "After k steps n/2^k = 1, so k = log₂(n) steps in the worst case."
        ),
        example_output="Binary search is O([notation]) because [explanation in 2 sentences].",
        context_sentence="Binary search finds a target in a sorted array by repeatedly halving the search space.",
        constraint_sentence="Include the Big O notation and explain why that complexity holds.",
    ),
    Task(
        task_id=6, category="qa", difficulty="medium", token_budget=65,
        task_description="Answer: What causes inflation and how does the central bank control it?",
        initial_bad_prompt="inflation",
        reference_answer=(
            "Inflation occurs when money supply grows faster than output or when demand exceeds supply. "
            "Central banks control it by raising interest rates, which reduces borrowing and slows price increases."
        ),
        example_output="Inflation is caused by [cause]. Central banks respond by [mechanism].",
        context_sentence="Inflation is the rate at which general price levels rise, eroding purchasing power over time.",
        constraint_sentence="Cover both causes and the central bank's primary control tool.",
    ),
    Task(
        task_id=7, category="qa", difficulty="easy", token_budget=80,
        task_description="Answer: What is the difference between RAM and ROM?",
        initial_bad_prompt="RAM ROM difference",
        reference_answer=(
            "RAM is volatile memory that temporarily stores data the computer is currently using and is erased when power is lost. "
            "ROM is non-volatile memory that permanently stores firmware and retains data without power."
        ),
        example_output="RAM is [description]. ROM is [description]. Key difference: [one sentence].",
        context_sentence="RAM and ROM are both types of computer memory serving different purposes in a system.",
        constraint_sentence="Define both clearly. State the single most important difference.",
    ),
    Task(
        task_id=8, category="qa", difficulty="easy", token_budget=80,
        task_description="Answer: Why does the sky appear blue during the day and red at sunset?",
        initial_bad_prompt="sky color why",
        reference_answer=(
            "Sunlight contains all colours. Earth's atmosphere scatters shorter blue wavelengths in all directions "
            "(Rayleigh scattering), making the sky appear blue. At sunset, sunlight travels through more atmosphere, "
            "scattering blue away and leaving longer red and orange wavelengths."
        ),
        example_output="Blue sky: [reason]. Red sunset: [reason].",
        context_sentence="This is explained by Rayleigh scattering, where atmospheric particles scatter light wavelengths differently.",
        constraint_sentence="Explain both daytime blue and sunset red in one coherent answer.",
    ),
    # ── INSTRUCTION ──────────────────────────────────────────────────────────
    Task(
        task_id=9, category="instruction", difficulty="easy", token_budget=80,
        task_description="Write step-by-step instructions to make a cup of tea",
        initial_bad_prompt="how to make tea",
        reference_answer=(
            "1. Boil water in a kettle.\n2. Place a tea bag in your cup.\n"
            "3. Pour hot water over the tea bag.\n4. Steep 3–5 minutes.\n"
            "5. Remove the tea bag.\n6. Add milk or sugar to taste."
        ),
        example_output="1. [Action verb]. 2. [Action verb]. (continue...)",
        context_sentence="Making tea involves boiling water and steeping a tea bag for the correct amount of time.",
        constraint_sentence="Numbered steps. Include timing. Start each step with an action verb.",
    ),
    Task(
        task_id=10, category="instruction", difficulty="medium", token_budget=65,
        task_description="Explain how to set up a Python virtual environment on Windows",
        initial_bad_prompt="python venv windows",
        reference_answer=(
            "1. Open Command Prompt.\n2. Navigate to project: cd path\\to\\project\n"
            "3. Create venv: python -m venv venv\n4. Activate: venv\\Scripts\\activate\n"
            "5. Install packages: pip install package\n6. Deactivate: deactivate"
        ),
        example_output="Step 1: Open [tool]. Step 2: Run `[command]`. (continue...)",
        context_sentence="A Python virtual environment isolates packages from your system Python installation.",
        constraint_sentence="Include exact commands in code format. Cover creation, activation, and deactivation.",
    ),
    Task(
        task_id=11, category="instruction", difficulty="hard", token_budget=55,
        task_description="Describe the steps to resolve a Git merge conflict",
        initial_bad_prompt="git merge conflict fix",
        reference_answer=(
            "1. Run git merge to trigger the conflict.\n"
            "2. Open conflicted file — Git marks sections with <<<<<<, =======, >>>>>>>.\n"
            "3. Edit file to keep correct code and remove markers.\n"
            "4. Stage resolved file: git add filename\n"
            "5. Commit: git commit\n6. Push: git push"
        ),
        example_output="1. [Trigger]. 2. [Find markers <<<<<<, =======, >>>>>>>]. 3. [Edit]. 4. [git add]. 5. [git commit].",
        context_sentence="A Git merge conflict occurs when two branches changed the same lines and Git cannot auto-resolve.",
        constraint_sentence="Include the conflict markers (<<<<<<, =======, >>>>>>>). Cover all steps through push.",
    ),
    # ── CODE ─────────────────────────────────────────────────────────────────
    Task(
        task_id=12, category="code", difficulty="medium", token_budget=65,
        task_description="Explain what a Python list comprehension does, with an example",
        initial_bad_prompt="list comprehension",
        reference_answer=(
            "A list comprehension creates a list concisely: [expression for item in iterable if condition]. "
            "Example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81."
        ),
        example_output="A list comprehension [definition]. Example: `[x**2 for x in range(10)]` produces [result].",
        context_sentence="List comprehensions are a Python feature offering a concise alternative to for loops for creating lists.",
        constraint_sentence="Include a runnable code example. Explain what the example produces.",
    ),
    Task(
        task_id=13, category="code", difficulty="medium", token_budget=65,
        task_description="Explain Big O notation using a simple code example",
        initial_bad_prompt="big o notation",
        reference_answer=(
            "Big O notation describes how runtime grows with input size. "
            "O(1) = constant (dict lookup). O(n) = linear (single loop). O(n²) = quadratic (nested loops)."
        ),
        example_output="Big O [definition]. Example: `[code]` is O([notation]) because [reason].",
        context_sentence="Big O notation describes the upper bound of an algorithm's time or space complexity.",
        constraint_sentence="Give 2+ examples with different complexities. Include code for each.",
    ),
    Task(
        task_id=14, category="code", difficulty="easy", token_budget=80,
        task_description="Explain what recursion is with a simple Python example",
        initial_bad_prompt="what is recursion",
        reference_answer=(
            "Recursion is when a function calls itself to solve a smaller version of the same problem. "
            "Example: def factorial(n): return 1 if n==0 else n*factorial(n-1). factorial(5) = 120."
        ),
        example_output="Recursion is [definition]. Example: `[code]` — base case: [base], recursive case: [recursive].",
        context_sentence="Recursion solves problems by breaking them into smaller instances of the same problem.",
        constraint_sentence="Show the base case and recursive case explicitly. Show sample input/output.",
    ),
]
```

---

### 2.3 server/actions.py

```python
"""
Six action functions for PromptRL.
Actions 0–4: deterministic prompt edits (no LLM calls, no randomness).
Action 5: STOP — handled in environment, not here.
"""
import re
from .task_bank import Task


ACTION_NAMES: dict[int, str] = {
    0: "ADD_CONTEXT",
    1: "SHORTEN",
    2: "ADD_EXAMPLE",
    3: "REPHRASE",
    4: "ADD_CONSTRAINT",
    5: "STOP",
}

_FILLER_PATTERNS: list[str] = [
    r"\bplease\b",
    r"\bcould you\b",
    r"\bi would like you to\b",
    r"\bi want you to\b",
    r"\bi need you to\b",
    r"\bcan you\b",
    r"\bwould you\b",
    r"\bkindly\b",
    r"\bif possible\b",
]


def count_tokens(text: str) -> int:
    """Word-level token count. Simple, no dependencies, consistent for relative comparison."""
    return len(text.split())


def apply_action(action_id: int, current_prompt: str, task: Task) -> str:
    """
    Apply one of actions 0–4 to the prompt. Action 5 (STOP) is handled
    by PromptRLEnvironment directly.

    Returns new prompt string. May equal current_prompt (no-op) — caller detects this.

    Raises:
        ValueError: If action_id not in {0,1,2,3,4}.
    """
    if action_id == 0:
        return add_context(current_prompt, task.context_sentence)
    elif action_id == 1:
        return shorten(current_prompt)
    elif action_id == 2:
        return add_example(current_prompt, task.example_output)
    elif action_id == 3:
        return rephrase(current_prompt)
    elif action_id == 4:
        return add_constraint(current_prompt, task.constraint_sentence)
    else:
        raise ValueError(
            f"Invalid action_id={action_id} passed to apply_action. "
            f"Must be in {{0,1,2,3,4}}. Action 5 (STOP) is handled by the environment."
        )


def add_context(prompt: str, context_sentence: str) -> str:
    """Append domain context if not already present."""
    if context_sentence[:30].lower() in prompt.lower():
        return prompt  # no-op
    return f"{prompt}\nContext: {context_sentence}"


def shorten(prompt: str) -> str:
    """Remove filler phrases. The only action that reduces token count."""
    result = prompt
    for pattern in _FILLER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result if result != prompt else prompt


def add_example(prompt: str, example_output: str) -> str:
    """Append example output format if not already present."""
    if "Example output format:" in prompt:
        return prompt  # no-op
    return f"{prompt}\nExample output format: {example_output}"


def rephrase(prompt: str) -> str:
    """Convert question phrasing to direct imperative. Net-neutral token change."""
    result = prompt
    result = re.sub(
        r"^(?:can you|could you|would you|please)\s+(.+?)[\?\.]*$",
        lambda m: m.group(1).strip().capitalize() + ".",
        result, flags=re.IGNORECASE | re.MULTILINE,
    )
    result = re.sub(
        r"^i (?:want|need|would like) you to\s+",
        "", result, flags=re.IGNORECASE | re.MULTILINE,
    )
    result = result.strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result if result != prompt else prompt


def add_constraint(prompt: str, constraint_sentence: str) -> str:
    """Append output constraint if not already present."""
    if "Requirement:" in prompt:
        return prompt  # no-op
    return f"{prompt}\nRequirement: {constraint_sentence}"
```

---

### 2.4 server/grader.py

```python
"""
Grading logic for PromptRL.
ALL LLM calls use the OpenAI Python client — required by hackathon rules.
Never use httpx or requests for LLM calls.

Modes:
  'rouge'         — uses DUMMY_OUTPUTS[task_id], no API, always works
  'openai_client' — calls LLM via OpenAI client, falls back to dummy on error
"""
import os
import sys
from openai import OpenAI
from rouge_score import rouge_scorer


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Pre-written canned outputs per task_id for ROUGE mode.
# Quality is intentionally moderate — represents what a model
# would return for the initial bad prompt, giving room for improvement.
DUMMY_OUTPUTS: dict[int, str] = {
    0: (
        "- Global temperatures are rising due to human emissions.\n"
        "- Extreme weather is becoming more common.\n"
        "- Countries need to cut emissions urgently."
    ),
    1: (
        "Romeo and Juliet are young lovers from rival families who secretly marry. "
        "Their tragic deaths reconcile their feuding families."
    ),
    2: (
        "Crypto risks include extreme volatility, potential total loss, "
        "regulatory uncertainty, and security vulnerabilities."
    ),
    3: (
        "- 1789: Bastille stormed.\n- 1791: Constitutional monarchy.\n"
        "- 1792: Republic declared.\n- 1793: Reign of Terror.\n"
        "- 1799: Napoleon takes power."
    ),
    4: (
        "Machine learning is when computers learn patterns from data, "
        "like learning to recognise images by seeing many examples."
    ),
    5: (
        "Binary search has O(log n) complexity because it halves the "
        "search space with each comparison."
    ),
    6: (
        "Inflation is caused by excess money supply or high demand. "
        "Central banks raise interest rates to slow price increases."
    ),
    7: (
        "RAM is volatile temporary storage erased on power loss. "
        "ROM is permanent non-volatile memory storing firmware."
    ),
    8: (
        "The sky is blue due to Rayleigh scattering of short wavelengths. "
        "At sunset, longer red wavelengths dominate after more atmosphere."
    ),
    9: (
        "1. Boil water. 2. Add tea bag. 3. Pour water. "
        "4. Steep 3-5 minutes. 5. Remove bag. 6. Add milk or sugar."
    ),
    10: (
        "Run python -m venv venv to create. Activate with "
        "venv\\Scripts\\activate. Install with pip. Type deactivate to exit."
    ),
    11: (
        "1. Run git merge. 2. Open file with <<<<<<, =======, >>>>>>> markers. "
        "3. Edit to resolve. 4. git add. 5. git commit. 6. git push."
    ),
    12: (
        "A list comprehension creates lists concisely: "
        "[x**2 for x in range(10)] produces [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]."
    ),
    13: (
        "Big O describes runtime growth. O(n) is a single loop. "
        "O(n²) is nested loops. Example: for i in range(n) is O(n)."
    ),
    14: (
        "Recursion is when a function calls itself. "
        "def factorial(n): return 1 if n==0 else n*factorial(n-1). "
        "factorial(5) = 120."
    ),
}


def _make_client() -> OpenAI:
    """Create OpenAI client from environment variables."""
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
        api_key=os.getenv("HF_TOKEN", ""),
    )


class Grader:
    """
    Reward computation for PromptRL.
    Uses OpenAI Python client for all LLM calls (hackathon requirement).

    Args:
        grader_type: 'rouge' (no API) or 'openai_client' (real LLM).
    """

    def __init__(self, grader_type: str = "rouge") -> None:
        self.grader_type = grader_type
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    def score(self, prompt: str, reference_answer: str, task_id: int) -> tuple[float, str]:
        """
        Score a prompt against the reference answer.

        Returns:
            (rouge_l_f1, llm_output) — score is float in [0.0, 1.0].
        """
        llm_output = self._get_output(prompt, task_id)
        rouge_l = self._compute_rouge(llm_output, reference_answer)
        return rouge_l, llm_output

    def _get_output(self, prompt: str, task_id: int) -> str:
        """Get LLM output. Uses OpenAI client if grader_type='openai_client', else dummy."""
        if self.grader_type == "openai_client":
            try:
                return self._call_llm(prompt)
            except Exception as e:
                print(f"[PromptRL] [LLM] Fallback to dummy (error: {e})", file=sys.stderr)
                return DUMMY_OUTPUTS.get(task_id, "")
        return DUMMY_OUTPUTS.get(task_id, "")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using the OpenAI Python client.
        MANDATORY approach — no raw HTTP to LLM endpoints.
        """
        client = _make_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
            timeout=30,
        )
        return response.choices[0].message.content or ""

    def _compute_rouge(self, hypothesis: str, reference: str) -> float:
        """ROUGE-L F1 between hypothesis and reference. Returns float in [0.0, 1.0]."""
        if not hypothesis or not reference:
            return 0.0
        scores = _ROUGE.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)

    @staticmethod
    def clip_reward(reward: float) -> float:
        """Clip to [-2.0, +2.0]. Wider range than pure quality env to accommodate STOP bonus."""
        return round(max(-2.0, min(2.0, reward)), 4)
```

---

### 2.5 server/prompt_rl_environment.py

```python
"""
Core RL environment for PromptRL: cost-aware prompt optimisation.
Reward = quality_delta - alpha * token_overhead.
Token budget enforcement terminates episode on breach.
STOP action (action_id=5) lets agent voluntarily end with quality bonus.
"""
import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import PromptAction, PromptObservation
from .actions import apply_action, count_tokens, ACTION_NAMES
from .grader import Grader
from .task_bank import TASK_BANK, Task


# ── Configuration ─────────────────────────────────────────────────────────────
MAX_STEPS: int           = int(os.getenv("MAX_STEPS", "7"))
DONE_THRESHOLD: float    = float(os.getenv("DONE_THRESHOLD", "0.85"))
TOKEN_PENALTY_ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
GRADER_TYPE: str         = os.getenv("GRADER", "rouge")
_TASK_SEED: str | None   = os.getenv("TASK_SEED", None)


class PromptRLEnvironment(Environment):
    """
    Cost-aware prompt optimisation RL environment.

    The agent edits a prompt using 6 actions (5 editing + STOP) to maximise
    output quality while respecting a hard token budget per task.

    Reward formula:
        reward = clip(quality_delta - alpha * token_overhead, -2.0, +2.0)

    Special termination conditions:
        - Budget exceeded:    reward=-0.5, done=True
        - Stuck (3× same):   reward=-0.5, done=True
        - STOP action:       reward=current_score × 1.5, done=True
        - Success (>0.85):   reward+=1.0 bonus, done=True
        - Max steps:         done=True
        - No-op:             reward=-0.1, episode continues
    """

    def __init__(self) -> None:
        self._grader = Grader(grader_type=GRADER_TYPE)
        self._episode_id: str = ""
        self._step_count: int = 0
        self._task: Task | None = None
        self._current_prompt: str = ""
        self._previous_prompt: str = ""
        self._current_score: float = 0.0
        self._previous_score: float = 0.0
        self._current_token_count: int = 0
        self._previous_token_count: int = 0
        self._tokens_used_total: int = 0
        self._last_action: int | None = None
        self._stuck_count: int = 0

    def reset(self) -> PromptObservation:
        """Start new episode. Select task, compute baseline score and token count."""
        if _TASK_SEED is not None:
            task_id = int(_TASK_SEED) % len(TASK_BANK)
        else:
            task_id = random.randint(0, len(TASK_BANK) - 1)

        self._task = TASK_BANK[task_id]
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._last_action = None
        self._stuck_count = 0
        self._tokens_used_total = 0
        self._current_prompt = self._task.initial_bad_prompt
        self._previous_prompt = ""

        self._current_token_count = count_tokens(self._current_prompt)
        self._previous_token_count = 0

        initial_score, _ = self._grader.score(
            self._current_prompt, self._task.reference_answer, task_id
        )
        self._current_score = initial_score
        self._previous_score = 0.0

        return PromptObservation(
            task_description=self._task.task_description,
            current_prompt=self._current_prompt,
            previous_prompt="",
            current_score=self._current_score,
            previous_score=0.0,
            current_token_count=self._current_token_count,
            previous_token_count=0,
            token_budget=self._task.token_budget,
            tokens_remaining=self._task.token_budget - self._current_token_count,
            token_overhead=0,
            reward=0.0,
            done=False,
            step_count=0,
            reference_answer=self._task.reference_answer,
            info={
                "grader_used": GRADER_TYPE,
                "action_applied": None,
                "stuck_count": 0,
                "termination_reason": None,
                "llm_output_preview": "",
                "no_op": False,
            },
        )

    def step(self, action: PromptAction) -> PromptObservation:
        """
        Apply action, compute cost-aware reward, return updated observation.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._task is None:
            raise RuntimeError("reset() must be called before step().")

        task_id = self._task.task_id

        # ── STOP action ──────────────────────────────────────────────────────
        if action.action_id == 5:
            stop_bonus = Grader.clip_reward(self._current_score * 1.5)
            self._step_count += 1
            return PromptObservation(
                task_description=self._task.task_description,
                current_prompt=self._current_prompt,
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                current_token_count=self._current_token_count,
                previous_token_count=self._previous_token_count,
                token_budget=self._task.token_budget,
                tokens_remaining=self._task.token_budget - self._current_token_count,
                token_overhead=0,
                reward=stop_bonus,
                done=True,
                step_count=self._step_count,
                reference_answer=self._task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": "STOP",
                    "stuck_count": self._stuck_count,
                    "termination_reason": "voluntary_stop",
                    "llm_output_preview": "",
                    "no_op": False,
                },
            )

        # ── Stuck detection ───────────────────────────────────────────────────
        if action.action_id == self._last_action:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._last_action = action.action_id

        if self._stuck_count >= 3:
            self._step_count += 1
            return PromptObservation(
                task_description=self._task.task_description,
                current_prompt=self._current_prompt,
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                current_token_count=self._current_token_count,
                previous_token_count=self._previous_token_count,
                token_budget=self._task.token_budget,
                tokens_remaining=self._task.token_budget - self._current_token_count,
                token_overhead=0,
                reward=-0.5,
                done=True,
                step_count=self._step_count,
                reference_answer=self._task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": ACTION_NAMES[action.action_id],
                    "stuck_count": self._stuck_count,
                    "termination_reason": "stuck",
                    "llm_output_preview": "",
                    "no_op": False,
                },
            )

        # ── Apply action ──────────────────────────────────────────────────────
        new_prompt = apply_action(action.action_id, self._current_prompt, self._task)

        # ── No-op detection ───────────────────────────────────────────────────
        if new_prompt == self._current_prompt:
            self._step_count += 1
            done = self._step_count >= MAX_STEPS
            return PromptObservation(
                task_description=self._task.task_description,
                current_prompt=self._current_prompt,
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                current_token_count=self._current_token_count,
                previous_token_count=self._previous_token_count,
                token_budget=self._task.token_budget,
                tokens_remaining=self._task.token_budget - self._current_token_count,
                token_overhead=0,
                reward=-0.1,
                done=done,
                step_count=self._step_count,
                reference_answer=self._task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": ACTION_NAMES[action.action_id],
                    "stuck_count": self._stuck_count,
                    "termination_reason": "max_steps" if done else None,
                    "llm_output_preview": "",
                    "no_op": True,
                },
            )

        # ── Budget check ──────────────────────────────────────────────────────
        new_token_count = count_tokens(new_prompt)
        if new_token_count > self._task.token_budget:
            self._step_count += 1
            return PromptObservation(
                task_description=self._task.task_description,
                current_prompt=self._current_prompt,  # prompt REVERTS
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                current_token_count=self._current_token_count,
                previous_token_count=self._previous_token_count,
                token_budget=self._task.token_budget,
                tokens_remaining=self._task.token_budget - self._current_token_count,
                token_overhead=0,
                reward=-0.5,
                done=True,
                step_count=self._step_count,
                reference_answer=self._task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": ACTION_NAMES[action.action_id],
                    "stuck_count": self._stuck_count,
                    "termination_reason": "budget_exceeded",
                    "llm_output_preview": "",
                    "no_op": False,
                    "tokens_over_budget": new_token_count - self._task.token_budget,
                },
            )

        # ── Score and reward ──────────────────────────────────────────────────
        new_score, llm_output = self._grader.score(
            new_prompt, self._task.reference_answer, task_id
        )
        token_overhead = new_token_count - self._current_token_count
        quality_delta = new_score - self._current_score
        raw_reward = quality_delta - TOKEN_PENALTY_ALPHA * token_overhead
        reward = Grader.clip_reward(raw_reward)

        # ── Update state ──────────────────────────────────────────────────────
        self._previous_prompt = self._current_prompt
        self._previous_score = self._current_score
        self._previous_token_count = self._current_token_count
        self._current_prompt = new_prompt
        self._current_score = new_score
        self._current_token_count = new_token_count
        if token_overhead > 0:
            self._tokens_used_total += token_overhead
        self._step_count += 1

        # ── Termination ───────────────────────────────────────────────────────
        done = False
        termination_reason: str | None = None
        if new_score > DONE_THRESHOLD:
            reward = Grader.clip_reward(reward + 1.0)
            done = True
            termination_reason = "success"
        elif self._step_count >= MAX_STEPS:
            done = True
            termination_reason = "max_steps"

        return PromptObservation(
            task_description=self._task.task_description,
            current_prompt=self._current_prompt,
            previous_prompt=self._previous_prompt,
            current_score=self._current_score,
            previous_score=self._previous_score,
            current_token_count=self._current_token_count,
            previous_token_count=self._previous_token_count,
            token_budget=self._task.token_budget,
            tokens_remaining=self._task.token_budget - self._current_token_count,
            token_overhead=token_overhead,
            reward=reward,
            done=done,
            step_count=self._step_count,
            reference_answer=self._task.reference_answer,
            info={
                "grader_used": GRADER_TYPE,
                "action_applied": ACTION_NAMES[action.action_id],
                "stuck_count": self._stuck_count,
                "termination_reason": termination_reason,
                "llm_output_preview": llm_output[:100] if llm_output else "",
                "no_op": False,
            },
        )

    @property
    def state(self) -> State:
        """Return current episode metadata."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )
```

---

### 2.6 server/app.py

```python
"""FastAPI app — three lines. All endpoints generated by create_app()."""
from openenv.core.env_server import create_app
from ..models import PromptAction, PromptObservation
from .prompt_rl_environment import PromptRLEnvironment

app = create_app(PromptRLEnvironment, PromptAction, PromptObservation)
```

---

### 2.7 server/Dockerfile

```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "prompt_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 2.8 server/requirements.txt

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

---

### 2.9 prompt_rl/__init__.py

```python
from .models import PromptAction, PromptObservation
from .client import PromptRLEnv

__all__ = ["PromptAction", "PromptObservation", "PromptRLEnv"]
```

---

### 2.10 inference.py (MANDATORY — repo root)

```python
"""
Cost-aware baseline inference script.

MANDATORY:
  - Name: inference.py (do not rename)
  - Location: repo root (not inside prompt_rl/)
  - Uses: openai.OpenAI client for all LLM calls
  - Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from os.environ
  - Covers: 3 tasks (easy, medium, hard), 7 steps each
  - Prints: per-step score, tokens, reward + summary with efficiency metric
  - Runtime: under 20 minutes on 2 vCPU / 8 GB RAM
  - Exit: always code 0

Usage:
    export API_BASE_URL=https://api-inference.huggingface.co/v1/
    export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
    export HF_TOKEN=hf_your_token
    python inference.py
"""
import os
import sys
import re
import random

from openai import OpenAI
from rouge_score import rouge_scorer

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str   = os.environ["MODEL_NAME"]
HF_TOKEN: str     = os.environ["HF_TOKEN"]
ALPHA: float      = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))

_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
_ROUGE  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def count_tokens(text: str) -> int:
    return len(text.split())


EVAL_TASKS = [
    {
        "difficulty": "easy", "token_budget": 80,
        "task_description": "Explain what machine learning is to a 10-year-old",
        "initial_prompt": "explain machine learning",
        "reference": (
            "Machine learning is when computers learn from examples, like how you "
            "learned to recognise cats by seeing many cats. The computer looks at data "
            "and figures out patterns by itself."
        ),
        "context": "Machine learning is a type of AI where computers learn patterns from data.",
        "example": "Machine learning is like [child analogy]. The computer [simple description].",
        "constraint": "Simple words only. No jargon. For a 10-year-old.",
    },
    {
        "difficulty": "medium", "token_budget": 65,
        "task_description": "Answer: What is the time complexity of binary search and why?",
        "initial_prompt": "binary search complexity",
        "reference": (
            "Binary search has O(log n) time complexity. With each comparison it eliminates "
            "half the remaining elements. After k steps n/2^k = 1, so k = log2(n) steps."
        ),
        "context": "Binary search finds a target in a sorted array by halving the search space.",
        "example": "Binary search is O([notation]) because [explanation].",
        "constraint": "Include Big O notation and explain why that complexity holds.",
    },
    {
        "difficulty": "hard", "token_budget": 55,
        "task_description": "Describe the steps to resolve a Git merge conflict",
        "initial_prompt": "git merge conflict fix",
        "reference": (
            "1. Run git merge. 2. Open conflicted file with <<<<<<, =======, >>>>>>> markers. "
            "3. Edit to keep correct code and remove markers. "
            "4. git add filename. 5. git commit. 6. git push."
        ),
        "context": "A Git merge conflict occurs when two branches changed the same lines differently.",
        "example": "1. [Trigger]. 2. [Markers: <<<<<<, =======, >>>>>>>]. 3. [Edit]. 4. [git add]. 5. [Commit].",
        "constraint": "Include the conflict markers (<<<<<<, =======, >>>>>>>). Cover all steps to push.",
    },
]

FILLER = [r"\bplease\b", r"\bcould you\b", r"\bcan you\b", r"\bi want you to\b"]


def apply(action_id: int, prompt: str, task: dict) -> str:
    if action_id == 0:
        if task["context"][:20].lower() in prompt.lower(): return prompt
        return f"{prompt}\nContext: {task['context']}"
    elif action_id == 1:
        r = prompt
        for f in FILLER: r = re.sub(f, "", r, flags=re.IGNORECASE)
        r = re.sub(r"  +", " ", r).strip()
        return r if r != prompt else prompt
    elif action_id == 2:
        if "Example output format:" in prompt: return prompt
        return f"{prompt}\nExample output format: {task['example']}"
    elif action_id == 3:
        r = re.sub(r"^(?:can you|could you|please)\s+(.+?)[\?\.]*$",
                   lambda m: m.group(1).strip().capitalize() + ".",
                   prompt, flags=re.IGNORECASE | re.MULTILINE)
        return r if r != prompt else prompt
    elif action_id == 4:
        if "Requirement:" in prompt: return prompt
        return f"{prompt}\nRequirement: {task['constraint']}"
    return prompt


def call_llm(prompt: str) -> str:
    try:
        r = _CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.1, timeout=30,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}", file=sys.stderr)
        return ""


def rouge_l(hyp: str, ref: str) -> float:
    if not hyp: return 0.0
    return round(_ROUGE.score(ref, hyp)["rougeL"].fmeasure, 4)


def run_episode(task: dict, max_steps: int = 7) -> dict:
    prompt = task["initial_prompt"]
    init_out = call_llm(prompt)
    init_score = rouge_l(init_out, task["reference"])
    init_tokens = count_tokens(prompt)

    print(f"\n  Task ({task['difficulty']}): {task['task_description']}")
    print(f"  Token budget  : {task['token_budget']}")
    print(f"  Initial tokens: {init_tokens}   prompt: '{prompt}'")
    print(f"  Initial score : {init_score:.4f}")

    current_score = init_score
    current_tokens = init_tokens
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action_id = random.randint(0, 5)

        if action_id == 5:
            stop_bonus = round(current_score * 1.5, 4)
            print(f"  Step {step+1}: action=STOP  score={current_score:.4f} stop_bonus={stop_bonus:.4f}  reward={stop_bonus:+.4f}")
            total_reward += stop_bonus
            steps += 1
            break

        new_prompt = apply(action_id, prompt, task)
        if new_prompt == prompt:
            print(f"  Step {step+1}: action={action_id} [NO-OP]  reward=-0.1000")
            total_reward += -0.1
            steps += 1
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > task["token_budget"]:
            print(f"  Step {step+1}: action={action_id} [BUDGET EXCEEDED {new_tokens}>{task['token_budget']}]  reward=-0.5000")
            total_reward += -0.5
            steps += 1
            break

        new_out = call_llm(new_prompt)
        new_score = rouge_l(new_out, task["reference"])
        overhead = new_tokens - current_tokens
        reward = round(new_score - current_score - ALPHA * overhead, 4)
        total_reward += reward
        steps += 1

        sign = "+" if overhead >= 0 else ""
        print(f"  Step {step+1}: action={action_id} score={new_score:.4f} tokens={new_tokens}/{task['token_budget']} overhead={sign}{overhead} reward={reward:+.4f}")

        prompt = new_prompt
        current_score = new_score
        current_tokens = new_tokens

        if current_score > 0.85:
            total_reward += 1.0
            print(f"  [SUCCESS] score exceeded 0.85 — bonus +1.0 added")
            break

    efficiency = round(current_score / max(1, current_tokens), 4)
    return {
        "difficulty": task["difficulty"],
        "initial_score": init_score,
        "final_score": current_score,
        "total_reward": round(total_reward, 4),
        "final_token_count": current_tokens,
        "token_budget": task["token_budget"],
        "efficiency": efficiency,
        "steps": steps,
    }


def main() -> None:
    print("=" * 66)
    print("PromptRL — Cost-Aware Baseline Inference Script")
    print(f"Model    : {MODEL_NAME}")
    print(f"Endpoint : {API_BASE_URL}")
    print(f"Alpha    : {ALPHA}")
    print("=" * 66)

    results = []
    for task in EVAL_TASKS:
        results.append(run_episode(task))

    print()
    print("=" * 66)
    print("BASELINE SCORES SUMMARY")
    print("=" * 66)
    print(f"{'Diff':<8} {'Score':>7} {'Tokens':>7} {'Budget':>7} {'Effic':>8} {'Reward':>8} {'Steps':>6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['difficulty']:<8} {r['final_score']:>7.4f} {r['final_token_count']:>7} "
            f"{r['token_budget']:>7} {r['efficiency']:>8.4f} {r['total_reward']:>8.4f} {r['steps']:>6}"
        )
    print("-" * 60)
    avg = round(sum(r["final_score"] for r in results) / len(results), 4)
    print(f"{'Average':<8} {avg:>7.4f}")
    print()
    print("Efficiency = final_score / final_token_count (higher = better)")
    print("All scores are ROUGE-L F1 in [0.0, 1.0]. Script complete.")


if __name__ == "__main__":
    main()
```

---

### 2.11 .env.example (repo root)

```bash
# .env.example — copy to .env, fill real values, NEVER commit .env

# MANDATORY — verified by submission checker
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

---

### 2.12 .gitignore (repo root)

```
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.venv/
venv/
uv.lock
.DS_Store
```

---

## 3. API Endpoints (auto-generated by OpenEnv)

| Method | Path | Description | Body | Response |
|---|---|---|---|---|
| GET | /health | Health check | — | `{"status": "ok"}` |
| WS | /ws | WebSocket for all env interactions | — | — |
| POST | /reset | HTTP reset | `{}` | PromptObservation JSON |
| POST | /step | HTTP step | `{"action_id": int}` | PromptObservation JSON |
| GET | /state | Episode state | — | State JSON |

All endpoints generated by `create_app()`. Do not write route handlers manually.
