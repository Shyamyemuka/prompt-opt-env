"""Integration tests for the PromptOptEnvEnvironment."""

import pytest

from prompt_opt_env.server.prompt_opt_env_environment import PromptOptEnvEnvironment
from prompt_opt_env.models import PromptAction


# ─── reset() ───────────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation():
    env = PromptOptEnvEnvironment()
    obs = env.reset()

    assert obs.task_description != ""
    assert obs.current_prompt != ""
    assert obs.previous_prompt == ""
    assert obs.reward == 0.0
    assert obs.done is False
    assert obs.step_count == 0
    assert 0.0 < obs.current_score < 1.0
    assert 0.0 < obs.previous_score < 1.0
    assert obs.reference_answer != ""
    assert obs.current_token_count > 0
    assert obs.previous_token_count == 0
    assert obs.token_budget > 0
    assert obs.token_overhead == 0
    assert isinstance(obs.info, dict)


def test_reset_starts_fresh_episode():
    env = PromptOptEnvEnvironment()
    obs1 = env.reset()
    # Take a step so state changes
    env.step(PromptAction(action_id=0))
    # Reset again — step_count should be 0
    obs2 = env.reset()
    assert obs2.step_count == 0
    assert obs2.done is False


# ─── step() ────────────────────────────────────────────────────────────────────

def test_step_returns_valid_observation():
    env = PromptOptEnvEnvironment()
    env.reset()
    obs = env.step(PromptAction(action_id=2))

    assert obs.step_count == 1
    assert isinstance(obs.reward, float)
    assert isinstance(obs.done, bool)
    assert obs.task_description != ""
    assert "action_applied" in obs.info


def test_all_six_actions_work():
    for action_id in range(6):
        env = PromptOptEnvEnvironment()
        env.reset()
        obs = env.step(PromptAction(action_id=action_id))
        assert obs is not None
        assert isinstance(obs.reward, float)
        assert obs.step_count == 1


def test_step_increments_step_count():
    env = PromptOptEnvEnvironment()
    env.reset()
    for i in range(1, 4):
        obs = env.step(PromptAction(action_id=i % 6))
        assert obs.step_count == i


# ─── episode termination ───────────────────────────────────────────────────────

def test_max_steps_terminates_episode():
    env = PromptOptEnvEnvironment()
    env.reset()
    obs = None
    # Take 7 steps cycling through different actions
    for i in range(7):
        obs = env.step(PromptAction(action_id=i % 5))
    assert obs.done is True


def test_stuck_detection_terminates_with_penalty():
    env = PromptOptEnvEnvironment()
    env.reset()
    obs = None
    # Take the same action 4 times (stuck_count will reach 3 on the 4th)
    for _ in range(4):
        obs = env.step(PromptAction(action_id=0))
        if obs.done:
            break
    assert obs.done is True
    assert obs.reward == -0.5


def test_stop_action_terminates_episode():
    env = PromptOptEnvEnvironment()
    env.reset()
    obs = env.step(PromptAction(action_id=5))
    assert obs.done is True
    assert "termination_reason" in obs.info
    assert obs.info["termination_reason"] == "voluntary_stop"


# ─── state() ───────────────────────────────────────────────────────────────────

def test_state_returns_episode_id():
    env = PromptOptEnvEnvironment()
    env.reset()
    state = env.state

    assert state.episode_id != ""
    assert state.step_count == 0


def test_state_step_count_updates():
    env = PromptOptEnvEnvironment()
    env.reset()
    env.step(PromptAction(action_id=1))
    env.step(PromptAction(action_id=2))
    state = env.state
    assert state.step_count == 2
