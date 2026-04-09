"""
Typed Pydantic v2 models for PromptOptEnv.
All fields always present — no Optional fields that might be missing.
The programmatic checker accesses fields by name.
"""
from typing import Any
from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class PromptAction(Action):
    """
    Action taken by the agent.

    action_id:
      0: ADD_CONTEXT    — append context sentence (adds ~10-15 tokens)
      1: SHORTEN        — remove filler phrases (reduces ~5-12 tokens)
      2: ADD_EXAMPLE    — append example output (adds ~12-20 tokens)
      3: REPHRASE       — convert to direct imperative (net 0 token change)
      4: ADD_CONSTRAINT — append constraint (adds ~8-12 tokens)
      5: STOP           — voluntarily end episode (reward = current_score × 1.5)
    """
    action_id: int = Field(
        ..., ge=0, le=5,
        description="Integer 0–5. Action 5 = STOP (voluntary episode end)."
    )


class PromptObservation(Observation):
    """
    Full observation returned by reset() and step().
    Every field is always present. Never null.
    """
    task_description: str = Field(..., description="What the prompt should accomplish")
    current_prompt: str = Field(..., description="Prompt after this step's action")
    previous_prompt: str = Field(..., description="Prompt before this step (empty at reset)")
    current_score: float = Field(..., gt=0.0, lt=1.0, description="ROUGE-L F1 of current prompt output (strictly between 0 and 1)")
    previous_score: float = Field(..., gt=0.0, lt=1.0, description="ROUGE-L F1 before this step (strictly between 0 and 1)")
    current_token_count: int = Field(..., ge=0, description="Word-level token count of current prompt")
    previous_token_count: int = Field(..., ge=0, description="Word-level token count before this step")
    token_budget: int = Field(..., ge=1, description="Hard ceiling on prompt token count for this task")
    tokens_remaining: int = Field(..., description="token_budget - current_token_count")
    token_overhead: int = Field(..., description="Tokens added this step (negative if SHORTEN applied)")
    reward: float = Field(..., description="Combined reward: quality_delta - alpha*token_overhead, clipped [-2, +2]")
    done: bool = Field(..., description="True if episode ended")
    step_count: int = Field(..., ge=0, description="Steps taken this episode")
    reference_answer: str = Field(..., description="Gold-standard answer for grader")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="grader_used, action_applied, stuck_count, termination_reason, llm_output_preview, no_op"
    )


# Keep backward-compatible aliases for any existing code
PromptOptAction = PromptAction
PromptOptObservation = PromptObservation
