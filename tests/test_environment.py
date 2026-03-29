"""Integration tests for the PromptOptEnvironment."""

import pytest

from prompt_opt_env.server.prompt_opt_env_environment import PromptOptEnvironment
from prompt_opt_env.models import PromptOptAction


# ─── reset() ───────────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation():
    env = PromptOptEnvironment()
    obs = env.reset()

    assert obs.task_description != ""
    assert obs.current_prompt != ""
    assert obs.previous_prompt == ""
    assert obs.reward == 0.0
    assert obs.done is False
    assert obs.step_count == 0
    assert 0.0 <= obs.current_score <= 1.0
    assert 0.0 <= obs.previous_score <= 1.0
    assert obs.reference_answer != ""
    assert isinstance(obs.info, dict)


def test_reset_starts_fresh_episode():
    env = PromptOptEnvironment()
    obs1 = env.reset()
    # Take a step so state changes
    env.step(PromptOptAction(action_id=0))
    # Reset again — step_count should be 0
    obs2 = env.reset()
    assert obs2.step_count == 0
    assert obs2.done is False


# ─── step() ────────────────────────────────────────────────────────────────────

def test_step_returns_valid_observation():
    env = PromptOptEnvironment()
    env.reset()
    obs = env.step(PromptOptAction(action_id=2))

    assert obs.step_count == 1
    assert isinstance(obs.reward, float)
    assert isinstance(obs.done, bool)
    assert obs.task_description != ""
    assert "action_applied" in obs.info


def test_all_five_actions_work():
    for action_id in range(5):
        env = PromptOptEnvironment()
        env.reset()
        obs = env.step(PromptOptAction(action_id=action_id))
        assert obs is not None
        assert isinstance(obs.reward, float)
        assert obs.step_count == 1


def test_step_increments_step_count():
    env = PromptOptEnvironment()
    env.reset()
    for i in range(1, 4):
        obs = env.step(PromptOptAction(action_id=i % 5))
        assert obs.step_count == i


# ─── episode termination ───────────────────────────────────────────────────────

def test_max_steps_terminates_episode():
    env = PromptOptEnvironment()
    env.reset()
    obs = None
    # Take 5 steps cycling through different actions
    for i in range(5):
        obs = env.step(PromptOptAction(action_id=i % 5))
    assert obs.done is True


def test_stuck_detection_terminates_with_penalty():
    env = PromptOptEnvironment()
    env.reset()
    obs = None
    # Take the same action 4 times (stuck_count will reach 3 on the 4th)
    for _ in range(4):
        obs = env.step(PromptOptAction(action_id=0))
        if obs.done:
            break
    assert obs.done is True
    assert obs.reward == -0.5


# ─── state() ───────────────────────────────────────────────────────────────────

def test_state_returns_episode_id():
    env = PromptOptEnvironment()
    env.reset()
    state = env.state

    assert state.episode_id != ""
    assert state.step_count == 0


def test_state_step_count_updates():
    env = PromptOptEnvironment()
    env.reset()
    env.step(PromptOptAction(action_id=1))
    env.step(PromptOptAction(action_id=2))
    state = env.state
    assert state.step_count == 2
