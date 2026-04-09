"""
LLM-powered prompt action transforms for PromptOptEnv.

These helpers replace naive string appends with model-based rewrites while keeping
safe deterministic fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

try:
    from .task_bank import Task
    from .actions import add_context, shorten, add_example, rephrase, add_constraint
    from ..llm_router import create_default_router
except (ModuleNotFoundError, ImportError):
    from task_bank import Task
    from actions import add_context, shorten, add_example, rephrase, add_constraint
    from llm_router import create_default_router


_ACTION_LLM_TIMEOUT_SECONDS = float(os.getenv("ACTION_LLM_TIMEOUT_SECONDS", "30"))
_ACTION_LLM_TEMPERATURE = float(os.getenv("ACTION_LLM_TEMPERATURE", "0.2"))
_ACTION_LLM_MAX_CALLS = int(os.getenv("ACTION_LLM_MAX_CALLS", "250"))
_ACTION_LLM_DRY_RUN = os.getenv("ACTION_LLM_DRY_RUN", "false").lower() == "true"
_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
_ACTION_LLM_MAX_RETRIES = int(os.getenv("ACTION_LLM_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "2")))


@dataclass
class ActionResult:
    """Result payload for one action application."""

    new_prompt: str
    success: bool
    explanation: str
    tokens_before: int
    tokens_after: int
    api_error: Optional[str] = None
    used_fallback: bool = False


_ACTION_CALLS = 0
_ACTION_CACHE: dict[tuple[str, str, int, str], str] = {}
_CACHE_HITS = 0
_ROUTER = create_default_router(
    default_model=_MODEL_NAME,
    default_base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/"),
    timeout_seconds=_ACTION_LLM_TIMEOUT_SECONDS,
    max_retries=_ACTION_LLM_MAX_RETRIES,
)


def count_tokens(text: str) -> int:
    return len((text or "").split())


def get_action_llm_stats() -> dict[str, int]:
    """Expose lightweight counters for diagnostics/benchmarking."""
    return {
        "calls": _ACTION_CALLS,
        "cache_hits": _CACHE_HITS,
        "cache_size": len(_ACTION_CACHE),
    }


def _sanitize_model_rewrite(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()

    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    return cleaned


def _call_llm_rewrite(prompt: str, instruction: str, max_tokens: int = 220) -> tuple[str, bool, str, Optional[str]]:
    """Return (prompt, success, explanation, api_error)."""
    global _ACTION_CALLS, _CACHE_HITS

    if not _ROUTER.has_provider():
        return prompt, False, "missing_api_key", "missing_api_key"

    cache_key = (instruction, prompt, max_tokens, _MODEL_NAME)
    cached = _ACTION_CACHE.get(cache_key)
    if cached:
        _CACHE_HITS += 1
        return cached, True, "cache_hit", None

    if _ACTION_LLM_DRY_RUN:
        return prompt, False, "dry_run_enabled", "dry_run_enabled"

    if _ACTION_CALLS >= _ACTION_LLM_MAX_CALLS:
        msg = f"action_llm_call_limit_reached:{_ACTION_LLM_MAX_CALLS}"
        return prompt, False, msg, msg

    full_prompt = (
        f"{instruction}\n\n"
        f"Current prompt:\n{prompt}\n\n"
        "Return ONLY the rewritten prompt text. No commentary, no markdown fences."
    )

    try:
        _ACTION_CALLS += 1
        rewritten_text = _ROUTER.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a prompt-optimization assistant. Rewrite prompts to be clearer and concise.",
                },
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=max_tokens,
            temperature=_ACTION_LLM_TEMPERATURE,
        )
        rewritten = _sanitize_model_rewrite(rewritten_text)
        if not rewritten:
            return prompt, False, "empty_rewrite", "empty_rewrite"

        _ACTION_CACHE[cache_key] = rewritten
        return rewritten, True, "llm_rewrite_success", None
    except Exception as exc:  # pragma: no cover - network/provider variance
        error_text = f"llm_api_error:{type(exc).__name__}:{str(exc)[:120]}"
        return prompt, False, "llm_api_error", error_text


def _task_fields(task: Any) -> tuple[str, str, str, str]:
    task_description = getattr(task, "task_description", "") or ""
    context_sentence = getattr(task, "context_sentence", "") or ""
    example_output = getattr(task, "example_output", "") or ""
    constraint_sentence = getattr(task, "constraint_sentence", "") or ""
    return task_description, context_sentence, example_output, constraint_sentence


def _rewrite_with_fallback(
    prompt: str,
    instruction: str,
    fallback_fn,
    fallback_arg: str,
    max_tokens: int = 220,
) -> ActionResult:
    before = count_tokens(prompt)
    rewritten, success, explanation, api_error = _call_llm_rewrite(prompt, instruction, max_tokens=max_tokens)

    used_fallback = False
    if not success or rewritten == prompt:
        rewritten = fallback_fn(prompt, fallback_arg)
        used_fallback = True
        success = True
        explanation = f"fallback_after_{explanation}"

    return ActionResult(
        new_prompt=rewritten,
        success=success,
        explanation=explanation,
        tokens_before=before,
        tokens_after=count_tokens(rewritten),
        api_error=api_error,
        used_fallback=used_fallback,
    )


def add_context_intelligent(prompt: str, task: Task) -> ActionResult:
    task_description, context_sentence, _, _ = _task_fields(task)
    instruction = (
        "Rewrite this prompt to include useful context naturally. "
        "Do not append a raw 'Context:' line.\n"
        f"Task: {task_description}\n"
        f"Context to weave in: {context_sentence}"
    )
    return _rewrite_with_fallback(prompt, instruction, add_context, context_sentence)


def shorten_intelligent(prompt: str) -> ActionResult:
    before = count_tokens(prompt)
    instruction = (
        "Rewrite this prompt to be concise while preserving requirements and intent. "
        "Remove filler and redundancy. Use direct language."
    )
    rewritten, success, explanation, api_error = _call_llm_rewrite(prompt, instruction, max_tokens=200)
    used_fallback = False
    if not success or rewritten == prompt:
        rewritten = shorten(prompt)
        used_fallback = True
        success = True
        explanation = f"fallback_after_{explanation}"

    # Guard: if rewrite became longer, keep deterministic SHORTEN output.
    if count_tokens(rewritten) > before:
        shrunk = shorten(prompt)
        if count_tokens(shrunk) <= count_tokens(rewritten):
            rewritten = shrunk
            used_fallback = True
            explanation = f"length_guard_after_{explanation}"

    return ActionResult(
        new_prompt=rewritten,
        success=success,
        explanation=explanation,
        tokens_before=before,
        tokens_after=count_tokens(rewritten),
        api_error=api_error,
        used_fallback=used_fallback,
    )


def add_example_intelligent(prompt: str, task: Task) -> ActionResult:
    task_description, _, example_output, _ = _task_fields(task)
    instruction = (
        "Rewrite this prompt to include a compact example of the desired output format. "
        "Integrate it naturally and keep the prompt concise.\n"
        f"Task: {task_description}\n"
        f"Example format: {example_output}"
    )
    return _rewrite_with_fallback(prompt, instruction, add_example, example_output)


def rephrase_intelligent(prompt: str) -> ActionResult:
    before = count_tokens(prompt)
    instruction = (
        "Rewrite this prompt for clarity and directness. "
        "Prefer imperative language and remove ambiguity."
    )
    rewritten, success, explanation, api_error = _call_llm_rewrite(prompt, instruction, max_tokens=200)
    used_fallback = False
    if not success or rewritten == prompt:
        rewritten = rephrase(prompt)
        used_fallback = True
        success = True
        explanation = f"fallback_after_{explanation}"

    return ActionResult(
        new_prompt=rewritten,
        success=success,
        explanation=explanation,
        tokens_before=before,
        tokens_after=count_tokens(rewritten),
        api_error=api_error,
        used_fallback=used_fallback,
    )


def add_constraint_intelligent(prompt: str, task: Task) -> ActionResult:
    task_description, _, _, constraint_sentence = _task_fields(task)
    instruction = (
        "Rewrite this prompt so constraints are explicit and naturally integrated. "
        "Do not just append a standalone requirement line.\n"
        f"Task: {task_description}\n"
        f"Constraint: {constraint_sentence}"
    )
    return _rewrite_with_fallback(prompt, instruction, add_constraint, constraint_sentence)


def rewrite_full(prompt: str, task: Task) -> ActionResult:
    """Full prompt rewrite helper (not part of action-id space in env runtime)."""
    task_description, context_sentence, example_output, constraint_sentence = _task_fields(task)
    instruction = (
        "Rewrite this prompt end-to-end for best output quality under token constraints.\n"
        f"Task: {task_description}\n"
        f"Context: {context_sentence}\n"
        f"Example output pattern: {example_output}\n"
        f"Constraint: {constraint_sentence}\n"
        "Keep it concise and unambiguous."
    )

    before = count_tokens(prompt)
    rewritten, success, explanation, api_error = _call_llm_rewrite(prompt, instruction, max_tokens=260)
    if not success or not rewritten:
        rewritten = prompt
        explanation = f"failed_{explanation}"

    return ActionResult(
        new_prompt=rewritten,
        success=bool(rewritten and rewritten != prompt),
        explanation=explanation,
        tokens_before=before,
        tokens_after=count_tokens(rewritten),
        api_error=api_error,
        used_fallback=(rewritten == prompt),
    )


def apply_action_intelligent(action_id: int, prompt: str, task: Task) -> ActionResult:
    """
    Apply intelligent action transform for action ids 0..4.

    Action id 5 remains STOP in the environment and is intentionally not handled
    here to preserve OpenEnv action semantics.
    """
    if action_id == 0:
        return add_context_intelligent(prompt, task)
    if action_id == 1:
        return shorten_intelligent(prompt)
    if action_id == 2:
        return add_example_intelligent(prompt, task)
    if action_id == 3:
        return rephrase_intelligent(prompt)
    if action_id == 4:
        return add_constraint_intelligent(prompt, task)
    raise ValueError(f"Invalid action_id={action_id}. Intelligent actions support ids 0..4 only.")


ACTION_NAMES_INTELLIGENT = {
    0: "ADD_CONTEXT_LLM",
    1: "SHORTEN_LLM",
    2: "ADD_EXAMPLE_LLM",
    3: "REPHRASE_LLM",
    4: "ADD_CONSTRAINT_LLM",
    5: "STOP",
}
