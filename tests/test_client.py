"""
WebSocket client tests — requires a running environment server.

These tests start the server in a subprocess, connect via WebSocket,
and verify the full client-server round trip.

Run with:
    cd prompt_opt_env
    uv run server &     # start server in background
    python -m pytest ../tests/test_client.py -v

Or skip if no server is running:
    python -m pytest ../tests/test_client.py -v -m "not integration"
"""

import pytest
import asyncio

# Mark all tests in this file as integration (skipped by default in CI without server)
pytestmark = pytest.mark.asyncio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ─── Unit-level client tests (no server needed) ────────────────────────────────

def test_client_import():
    """Client class is importable and correctly subclasses EnvClient."""
    from prompt_opt_env.client import PromptOptEnv
    from openenv.core import EnvClient
    assert issubclass(PromptOptEnv, EnvClient)


def test_client_step_payload():
    """_step_payload correctly serialises PromptOptAction to action_id dict."""
    from prompt_opt_env.client import PromptOptEnv
    from prompt_opt_env.models import PromptOptAction

    # Instantiate without connecting (just to call the method)
    client = PromptOptEnv.__new__(PromptOptEnv)
    for action_id in range(5):
        payload = client._step_payload(PromptOptAction(action_id=action_id))
        assert payload == {"action_id": action_id}


def test_client_parse_state():
    """_parse_state correctly parses the state payload."""
    from prompt_opt_env.client import PromptOptEnv
    from openenv.core.env_server.types import State

    client = PromptOptEnv.__new__(PromptOptEnv)
    state = client._parse_state({"episode_id": "abc-123", "step_count": 3})
    assert state.episode_id == "abc-123"
    assert state.step_count == 3


def test_client_parse_result_full_observation():
    """_parse_result correctly parses a full server response into StepResult."""
    from prompt_opt_env.client import PromptOptEnv

    client = PromptOptEnv.__new__(PromptOptEnv)
    payload = {
        "observation": {
            "task_description": "Summarise Romeo and Juliet in 2 sentences",
            "current_prompt": "Summarise Romeo and Juliet.",
            "previous_prompt": "tell me about romeo and juliet",
            "current_score": 0.28,
            "previous_score": 0.12,
            "reward": 0.16,
            "done": False,
            "step_count": 1,
            "reference_answer": "Romeo and Juliet is a tragedy...",
            "info": {"grader_used": "rouge", "action_applied": "REPHRASE", "stuck_count": 0},
        },
        "reward": 0.16,
        "done": False,
    }
    result = client._parse_result(payload)
    obs = result.observation

    assert obs.task_description == "Summarise Romeo and Juliet in 2 sentences"
    assert obs.current_prompt == "Summarise Romeo and Juliet."
    assert obs.previous_prompt == "tell me about romeo and juliet"
    assert obs.current_score == 0.28
    assert obs.previous_score == 0.12
    assert obs.reward == 0.16
    assert obs.done is False
    assert obs.step_count == 1
    assert obs.reference_answer == "Romeo and Juliet is a tragedy..."
    assert obs.info["action_applied"] == "REPHRASE"
    assert result.reward == 0.16
    assert result.done is False
