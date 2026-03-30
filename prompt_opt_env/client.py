"""Prompt Opt Env / PromptRL — WebSocket Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PromptAction, PromptObservation


class PromptRLEnv(
    EnvClient[PromptAction, PromptObservation, State]
):
    """
    Client for the PromptRL RL environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example::

        async with PromptRLEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            obs = result.observation
            print(obs.task_description)
            print(f"Budget: {obs.token_budget} tokens | Current: {obs.current_token_count}")

            result = await env.step(PromptAction(action_id=5))  # STOP
            print(f"reward={result.reward:.3f} done={result.done}")
    """

    def _step_payload(self, action: PromptAction) -> Dict:
        """
        Convert PromptAction to JSON payload for the step WebSocket message.

        Args:
            action: PromptAction instance with action_id 0–5.

        Returns:
            Dictionary with 'action_id' key.
        """
        return {"action_id": action.action_id}

    def _parse_result(self, payload: Dict) -> StepResult[PromptObservation]:
        """
        Parse server response into StepResult[PromptObservation].

        Args:
            payload: JSON response data from the server.

        Returns:
            StepResult with a fully populated PromptObservation.
        """
        obs_data = payload.get("observation", {})
        observation = PromptObservation(
            task_description=obs_data.get("task_description", ""),
            current_prompt=obs_data.get("current_prompt", ""),
            previous_prompt=obs_data.get("previous_prompt", ""),
            current_score=obs_data.get("current_score", 0.0),
            previous_score=obs_data.get("previous_score", 0.0),
            current_token_count=obs_data.get("current_token_count", 0),
            previous_token_count=obs_data.get("previous_token_count", 0),
            token_budget=obs_data.get("token_budget", 80),
            tokens_remaining=obs_data.get("tokens_remaining", 80),
            token_overhead=obs_data.get("token_overhead", 0),
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


# Backward-compat alias
PromptOptEnv = PromptRLEnv
