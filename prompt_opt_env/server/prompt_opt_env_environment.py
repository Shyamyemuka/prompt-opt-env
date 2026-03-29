# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Core RL Environment — PromptOptEnvironment.

Implements the OpenEnv Environment interface.
The agent iteratively edits a prompt to maximize LLM output quality (ROUGE-L).

Episode lifecycle:
  1. reset() → picks a task, sets initial bad prompt, computes baseline score
  2. step(action) → applies editing action, scores result, returns reward
  3. done when: max steps reached, ROUGE score > threshold, or agent is stuck
"""

import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PromptOptAction, PromptOptObservation, PromptState
    from .actions import apply_action, ACTION_NAMES
    from .grader import Grader
    from .task_bank import TASK_BANK, Task
except ImportError:
    from models import PromptOptAction, PromptOptObservation, PromptState
    from server.actions import apply_action, ACTION_NAMES
    from server.grader import Grader
    from server.task_bank import TASK_BANK, Task


# Configuration — all tunable via environment variables
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "5"))
DONE_THRESHOLD: float = float(os.getenv("DONE_THRESHOLD", "0.85"))
GRADER_TYPE: str = os.getenv("GRADER", "rouge")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_MODEL: str = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TASK_SEED: str | None = os.getenv("TASK_SEED", None)


class PromptOptEnvironment(Environment):
    """
    OpenEnv RL environment for prompt optimization.

    The agent observes a task description and a current prompt string, then takes
    one of five deterministic editing actions (add context, shorten, add example,
    rephrase, add constraint) to improve the prompt. After each action, the
    environment scores the improved prompt's output using ROUGE-L against a
    reference answer and returns the score delta as the reward signal.

    Episodes run for a maximum of MAX_STEPS (default 5) steps or terminate early
    on success (score > DONE_THRESHOLD) or stuck detection (same action 3× in a row).
    """

    # Enable concurrent WebSocket sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise the RL environment with grader and empty episode state."""
        self._grader = Grader(
            grader_type=GRADER_TYPE,
            hf_token=HF_TOKEN,
            hf_model=HF_MODEL,
        )
        self._episode_id: str = ""
        self._step_count: int = 0
        self._task_id: int = 0
        self._current_task: Task | None = None
        self._current_prompt: str = ""
        self._previous_prompt: str = ""
        self._current_score: float = 0.0
        self._previous_score: float = 0.0
        self._last_action: int | None = None
        self._stuck_count: int = 0

    def reset(self) -> PromptOptObservation:
        """
        Start a new episode. Selects a task, sets the initial bad prompt,
        and computes the baseline ROUGE score.

        Returns:
            PromptOptObservation with all fields populated and reward=0.0, done=False.
        """
        # Task selection
        if TASK_SEED:
            task_id = int(TASK_SEED) % len(TASK_BANK)
        else:
            task_id = random.randint(0, len(TASK_BANK) - 1)

        self._task_id = task_id
        self._current_task = TASK_BANK[task_id]
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._current_prompt = self._current_task.initial_bad_prompt
        self._previous_prompt = ""
        self._last_action = None
        self._stuck_count = 0

        # Compute baseline score for the initial bad prompt
        initial_score, _ = self._grader.score(
            self._current_prompt,
            self._current_task.reference_answer,
            task_id,
        )
        self._current_score = initial_score
        self._previous_score = 0.0

        return PromptOptObservation(
            task_description=self._current_task.task_description,
            current_prompt=self._current_prompt,
            previous_prompt="",
            current_score=self._current_score,
            previous_score=0.0,
            reward=0.0,
            done=False,
            step_count=0,
            reference_answer=self._current_task.reference_answer,
            info={
                "grader_used": GRADER_TYPE,
                "action_applied": None,
                "stuck_count": 0,
            },
        )

    def step(self, action: PromptOptAction) -> PromptOptObservation:  # type: ignore[override]
        """
        Apply an editing action to the current prompt, compute reward.

        Handles stuck detection, no-op detection, grader call, and
        episode termination (max steps, success threshold, stuck).

        Args:
            action: PromptOptAction with action_id in {0, 1, 2, 3, 4}.

        Returns:
            PromptOptObservation with updated scores, reward, and done flag.

        Raises:
            RuntimeError: If called before reset().
        """
        if self._current_task is None:
            raise RuntimeError(
                "Call reset() before step(). No active episode."
            )

        # Stuck detection — same action repeated consecutively
        if action.action_id == self._last_action:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._last_action = action.action_id

        # Stuck termination — same action 3 times in a row
        if self._stuck_count >= 3:
            self._step_count += 1
            return PromptOptObservation(
                task_description=self._current_task.task_description,
                current_prompt=self._current_prompt,
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                reward=-0.5,
                done=True,
                step_count=self._step_count,
                reference_answer=self._current_task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": ACTION_NAMES[action.action_id],
                    "stuck_count": self._stuck_count,
                },
            )

        # Apply the editing action (pure string transformation)
        new_prompt = apply_action(action.action_id, self._current_prompt, self._current_task)

        # No-op detection — action produced no change
        if new_prompt == self._current_prompt:
            self._step_count += 1
            return PromptOptObservation(
                task_description=self._current_task.task_description,
                current_prompt=self._current_prompt,
                previous_prompt=self._previous_prompt,
                current_score=self._current_score,
                previous_score=self._previous_score,
                reward=-0.1,
                done=self._step_count >= MAX_STEPS,
                step_count=self._step_count,
                reference_answer=self._current_task.reference_answer,
                info={
                    "grader_used": GRADER_TYPE,
                    "action_applied": ACTION_NAMES[action.action_id],
                    "stuck_count": self._stuck_count,
                    "no_op": True,
                },
            )

        # Score the new prompt with the grader
        new_score, llm_output = self._grader.score(
            new_prompt,
            self._current_task.reference_answer,
            self._task_id,
        )

        # Compute delta reward
        reward = Grader.clip_reward(new_score - self._current_score)

        # Update state
        self._previous_prompt = self._current_prompt
        self._previous_score = self._current_score
        self._current_prompt = new_prompt
        self._current_score = new_score
        self._step_count += 1

        # Termination checks
        done = False
        if new_score > DONE_THRESHOLD:
            reward = min(2.0, reward + 1.0)  # success bonus
            done = True
        elif self._step_count >= MAX_STEPS:
            done = True

        return PromptOptObservation(
            task_description=self._current_task.task_description,
            current_prompt=self._current_prompt,
            previous_prompt=self._previous_prompt,
            current_score=self._current_score,
            previous_score=self._previous_score,
            reward=reward,
            done=done,
            step_count=self._step_count,
            reference_answer=self._current_task.reference_answer,
            info={
                "grader_used": GRADER_TYPE,
                "action_applied": ACTION_NAMES[action.action_id],
                "stuck_count": self._stuck_count,
                "llm_output_preview": llm_output[:100] if llm_output else "",
            },
        )

    @property
    def state(self) -> State:
        """
        Return current episode metadata.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )
