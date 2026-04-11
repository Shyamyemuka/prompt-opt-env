"""Unit tests for the Grader (ROUGE scoring + OpenAI Python client fallback)."""

import pytest
from unittest.mock import patch, MagicMock

from prompt_opt_env.server.grader import Grader, DUMMY_OUTPUTS, SCORE_EPSILON


def test_rouge_score_returns_float():
    grader = Grader(grader_type="rouge")
    score, output = grader.score("Explain recursion.", "Recursion calls itself.", task_id=14)
    assert isinstance(score, float)
    assert 0.0 < score < 1.0


def test_rouge_score_is_nonzero_for_related_content():
    grader = Grader(grader_type="rouge")
    reference = "Recursion is when a function calls itself."
    # task 14's dummy output is about recursion → should score higher than task 0's
    score14, _ = grader.score("prompt", reference, task_id=14)
    score0, _ = grader.score("prompt", reference, task_id=0)
    assert score14 >= score0


def test_rouge_output_is_dummy_when_no_api_token():
    grader = Grader(grader_type="rouge")
    _, output = grader.score("test prompt", "any reference", task_id=5)
    assert output == DUMMY_OUTPUTS[5]


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
    assert 0.0 < score < 1.0
    assert output == DUMMY_OUTPUTS[0]  # fell back to dummy


def test_compute_rouge_empty_inputs():
    grader = Grader()
    assert grader._compute_rouge("", "reference") == SCORE_EPSILON
    assert grader._compute_rouge("hypothesis", "") == SCORE_EPSILON
    assert grader._compute_rouge("", "") == SCORE_EPSILON


def test_compute_rouge_exact_match_stays_below_one():
    grader = Grader()
    score = grader._compute_rouge("identical answer", "identical answer")
    assert 0.0 < score < 1.0
