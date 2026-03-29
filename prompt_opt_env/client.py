# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt Opt Env — WebSocket Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PromptOptAction, PromptOptObservation


class PromptOptEnv(
    EnvClient[PromptOptAction, PromptOptObservation, State]
):
    """
    Client for the PromptOptEnv RL environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example::

        async with PromptOptEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            print(result.observation.task_description)

            result = await env.step(PromptOptAction(action_id=2))
            print(f"reward={result.reward:.3f} done={result.done}")
    """

    def _step_payload(self, action: PromptOptAction) -> Dict:
        """
        Convert PromptOptAction to JSON payload for the step WebSocket message.

        Args:
            action: PromptOptAction instance with action_id 0–4.

        Returns:
            Dictionary with 'action_id' key.
        """
        return {"action_id": action.action_id}

    def _parse_result(self, payload: Dict) -> StepResult[PromptOptObservation]:
        """
        Parse server response into StepResult[PromptOptObservation].

        Args:
            payload: JSON response data from the server.

        Returns:
            StepResult with a fully populated PromptOptObservation.
        """
        obs_data = payload.get("observation", {})
        observation = PromptOptObservation(
            task_description=obs_data.get("task_description", ""),
            current_prompt=obs_data.get("current_prompt", ""),
            previous_prompt=obs_data.get("previous_prompt", ""),
            current_score=obs_data.get("current_score", 0.0),
            previous_score=obs_data.get("previous_score", 0.0),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            step_count=obs_data.get("step_count", 0),
            reference_answer=obs_data.get("reference_answer", ""),
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
