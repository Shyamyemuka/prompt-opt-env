"""
Core RL environment for PromptOptEnv: cost-aware prompt optimisation.
Reward = quality_delta - alpha * token_overhead.
Token budget enforcement terminates episode on breach.
STOP action (action_id=5) lets agent voluntarily end with quality bonus.
"""
import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PromptAction, PromptObservation
    from .actions import apply_action, count_tokens, ACTION_NAMES
    from .grader import Grader
    from .task_bank import TASK_BANK, Task
except (ModuleNotFoundError, ImportError):
    from models import PromptAction, PromptObservation
    from server.actions import apply_action, count_tokens, ACTION_NAMES
    from server.grader import Grader
    from server.task_bank import TASK_BANK, Task


# ── Configuration ─────────────────────────────────────────────────────────────
MAX_STEPS: int            = int(os.getenv("MAX_STEPS", "7"))
DONE_THRESHOLD: float     = float(os.getenv("DONE_THRESHOLD", "0.85"))
TOKEN_PENALTY_ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
GRADER_TYPE: str          = os.getenv("GRADER", "rouge")
_TASK_SEED: str | None    = os.getenv("TASK_SEED", None)


class PromptOptEnvEnvironment(Environment):
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

    Configuration via environment variables:
        GRADER              — 'rouge' or 'openai_client'. Default: 'rouge'.
        MAX_STEPS           — Max steps per episode. Default: 7.
        DONE_THRESHOLD      — ROUGE-L for success. Default: 0.85.
        TOKEN_PENALTY_ALPHA — Cost penalty coefficient. Default: 0.02.
        API_BASE_URL        — OpenAI-compatible endpoint.
        MODEL_NAME          — Model identifier.
        HF_TOKEN            — API key.
    """

    # Enable concurrent WebSocket sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

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

    def step(self, action: PromptAction) -> PromptObservation:  # type: ignore[override]
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


# Backward-compat alias
PromptOptEnvironment = PromptOptEnvEnvironment
