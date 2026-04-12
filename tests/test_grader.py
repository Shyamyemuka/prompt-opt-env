"""Unit tests for the Grader (ROUGE scoring + OpenAI Python client fallback)."""

import pytest
from unittest.mock import patch, MagicMock

from prompt_opt_env.server.grader import Grader, SCORE_EPSILON


def test_rouge_score_returns_float():
    grader = Grader(grader_type="rouge")
    score, output = grader.score("Explain recursion.", "Recursion calls itself.", task_id=14)
    assert isinstance(score, float)
    assert 0 < score < 1


def test_rouge_mode_rewards_related_prompt_more():
    grader = Grader(grader_type="rouge")
    reference = "Recursion is when a function calls itself."
    related_score, _ = grader.score("Explain recursion with base case and recursive case.", reference, task_id=14)
    unrelated_score, _ = grader.score("Tell me about climate change", reference, task_id=14)
    assert related_score >= unrelated_score


def test_rouge_output_is_deterministic_and_non_empty():
    grader = Grader(grader_type="rouge")
    _, output1 = grader.score("test prompt", "any reference", task_id=5)
    _, output2 = grader.score("test prompt", "any reference", task_id=5)
    assert output1 != ""
    assert output1 == output2


def test_clip_reward_within_range():
    assert Grader.clip_reward(0.5) == 0.5
    assert Grader.clip_reward(2.0) == 2.0  # within [-2.0, 2.0]
    assert Grader.clip_reward(-2.0) == -2.0 # range is wider now


def test_clip_reward_lower_bound():
    assert Grader.clip_reward(-2.5) == -2.0  # clipped at -2.0


def test_clip_reward_upper_bound():
    assert Grader.clip_reward(3.0) == 2.0  # clipped at 2.0


def test_openai_api_fallback_on_failure():
    grader = Grader(grader_type="openai_client")
    with patch.object(grader, "_call_llm", side_effect=Exception("API down")):
        score, output = grader.score("test prompt", "reference answer", task_id=0)
    assert isinstance(score, float)
    assert 0 < score < 1
    expected = grader._deterministic_rouge_output("test prompt", "reference answer", task_id=0)
    assert output == expected


def test_rouge_mode_is_prompt_sensitive_for_same_task():
    grader = Grader(grader_type="rouge")
    reference = (
        "Binary search has O(log n) complexity because each comparison removes "
        "half of the remaining search space."
    )

    weak_prompt = "binary search"
    strong_prompt = (
        "Explain binary search complexity. Context: Binary search halves the search "
        "space each step. Requirement: Include Big O notation and explain why."
    )

    weak_score, _ = grader.score(weak_prompt, reference, task_id=5)
    strong_score, _ = grader.score(strong_prompt, reference, task_id=5)
    assert strong_score >= weak_score


def test_compute_rouge_empty_inputs():
    grader = Grader()
    assert grader._compute_rouge("", "reference") == SCORE_EPSILON
    assert grader._compute_rouge("hypothesis", "") == SCORE_EPSILON
    assert grader._compute_rouge("", "") == SCORE_EPSILON


def test_compute_rouge_exact_match_stays_below_one():
    grader = Grader()
    score = grader._compute_rouge("identical answer", "identical answer")
    assert 0 < score < 1
