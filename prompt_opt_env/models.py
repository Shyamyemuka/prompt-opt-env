# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PromptOptEnv RL environment.

Three models:
  - PromptOptAction: the action taken by the agent (which editing action to apply)
  - PromptOptObservation: the full observation returned after reset() or step()
  - PromptState: internal episode state returned by state()
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class PromptOptAction(Action):
    """
    Action taken by the agent to edit the current prompt.

    action_id must be one of:
      0: ADD_CONTEXT    — append domain context sentence
      1: SHORTEN         — remove filler and redundant phrases
      2: ADD_EXAMPLE     — append example output format
      3: REPHRASE        — rewrite in direct imperative voice
      4: ADD_CONSTRAINT  — append explicit output constraint
    """

    action_id: int = Field(
        ...,
        ge=0,
        le=4,
        description="Integer 0–4 identifying which editing action to apply",
    )


class PromptOptObservation(Observation):
    """
    Full observation returned after reset() or step().

    All fields are always present — no optional fields that could be missing.
    Judges' programmatic checks will access these fields by name.
    """

    task_description: str = Field(
        ..., description="English description of what the prompt should accomplish"
    )
    current_prompt: str = Field(
        ..., description="The prompt string after this step's action"
    )
    previous_prompt: str = Field(
        ..., description="The prompt string before this step's action"
    )
    current_score: float = Field(
        ..., ge=0.0, le=1.0, description="ROUGE-L F1 score of current prompt output"
    )
    previous_score: float = Field(
        ..., ge=0.0, le=1.0, description="ROUGE-L F1 score before this step"
    )
    reward: float = Field(
        ..., description="current_score minus previous_score, clipped to [-1, 2]"
    )
    done: bool = Field(..., description="True if episode has ended")
    step_count: int = Field(..., ge=0, description="Number of steps taken so far this episode")
    reference_answer: str = Field(
        ..., description="Gold-standard answer used by the grader"
    )
    info: dict = Field(
        default_factory=dict,
        description="Extra metadata: grader_used, action_applied, stuck_count, llm_output_preview",
    )


class PromptState:
    """Internal episode state tracker. Returned by state() endpoint."""

    def __init__(self, episode_id: str, step_count: int, task_id: int):
        self.episode_id = episode_id
        self.step_count = step_count
        self.task_id = task_id
