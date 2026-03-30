"""Unit tests for all 5 prompt-editing action functions + token counting."""

import pytest

from prompt_opt_env.server.actions import (
    add_context,
    add_example,
    add_constraint,
    rephrase,
    shorten,
    apply_action,
    count_tokens,
    ACTION_NAMES,
)
from prompt_opt_env.server.task_bank import TASK_BANK


# ─── count_tokens ──────────────────────────────────────────────────────────────

def test_count_tokens():
    assert count_tokens("one two three") == 3
    assert count_tokens("  hello   world  ") == 2
    assert count_tokens("") == 0

# ─── add_context ───────────────────────────────────────────────────────────────

def test_add_context_appends():
    result = add_context("Tell me about dogs.", "Dogs are domesticated mammals.")
    assert "Context: Dogs are domesticated mammals." in result


def test_add_context_no_duplicate():
    prompt = "Tell me about dogs.\nContext: Dogs are domesticated mammals."
    result = add_context(prompt, "Dogs are domesticated mammals.")
    assert result == prompt  # no-op — already present


# ─── shorten ───────────────────────────────────────────────────────────────────

def test_shorten_removes_please():
    result = shorten("Please explain machine learning.")
    assert "please" not in result.lower()
    assert "explain machine learning" in result.lower()


def test_shorten_removes_could_you():
    result = shorten("Could you tell me about Python?")
    assert "could you" not in result.lower()


def test_shorten_no_change_on_short_clean_prompt():
    prompt = "Explain recursion."
    result = shorten(prompt)
    assert result == prompt  # nothing to remove


# ─── add_example ───────────────────────────────────────────────────────────────

def test_add_example_appends():
    result = add_example("Summarise this.", "Example: [key points]")
    assert "Example output format:" in result
    assert "Example: [key points]" in result


def test_add_example_no_duplicate():
    prompt = "Summarise this.\nExample output format: [key points]"
    result = add_example(prompt, "[key points]")
    assert result == prompt  # no-op — marker already present


# ─── rephrase ──────────────────────────────────────────────────────────────────

def test_rephrase_question_to_imperative():
    result = rephrase("Can you explain recursion?")
    assert "can you" not in result.lower()
    assert "explain recursion" in result.lower()


def test_rephrase_ends_with_punctuation():
    result = rephrase("Could you summarise this article")
    assert result[-1] in ".!?"


# ─── add_constraint ────────────────────────────────────────────────────────────

def test_add_constraint_appends():
    result = add_constraint("Summarise this.", "In under 50 words.")
    assert "Requirement:" in result
    assert "In under 50 words." in result


def test_add_constraint_no_duplicate():
    prompt = "Summarise this.\nRequirement: In under 50 words."
    result = add_constraint(prompt, "In under 50 words.")
    assert result == prompt  # no-op — already present


# ─── apply_action dispatcher ───────────────────────────────────────────────────

def test_apply_action_dispatcher_0_to_4():
    task = TASK_BANK[0]
    for action_id in range(5):
        result = apply_action(action_id, "simple prompt", task)
        assert isinstance(result, str)
        assert len(result) > 0


def test_apply_action_invalid_raises():
    with pytest.raises(ValueError, match="Invalid action_id=5"):
        apply_action(5, "prompt", TASK_BANK[0])
    with pytest.raises(ValueError, match="Invalid action_id=99"):
        apply_action(99, "prompt", TASK_BANK[0])


def test_action_names_complete():
    assert set(ACTION_NAMES.keys()) == {0, 1, 2, 3, 4, 5}
    assert ACTION_NAMES[0] == "ADD_CONTEXT"
    assert ACTION_NAMES[5] == "STOP"
