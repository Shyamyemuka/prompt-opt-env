"""
Six action functions for PromptOptEnv.
Actions 0–4: deterministic prompt edits (no LLM calls, no randomness).
Action 5: STOP — handled in environment, not here.
"""
import re

try:
    from .task_bank import Task
except (ModuleNotFoundError, ImportError):
    from task_bank import Task


ACTION_NAMES: dict[int, str] = {
    0: "ADD_CONTEXT",
    1: "SHORTEN",
    2: "ADD_EXAMPLE",
    3: "REPHRASE",
    4: "ADD_CONSTRAINT",
    5: "STOP",
}

_FILLER_PATTERNS: list[str] = [
    r"\bplease\b",
    r"\bcould you\b",
    r"\bi would like you to\b",
    r"\bi want you to\b",
    r"\bi need you to\b",
    r"\bcan you\b",
    r"\bwould you\b",
    r"\bkindly\b",
    r"\bif possible\b",
    r"\bjust\b",
]


def count_tokens(text: str) -> int:
    """Word-level token count. Simple, no dependencies, consistent for relative comparison."""
    return len(text.split())


def apply_action(action_id: int, current_prompt: str, task: Task) -> str:
    """
    Apply one of actions 0–4 to the prompt. Action 5 (STOP) is handled
    by PromptOptEnvEnvironment directly.

    Returns new prompt string. May equal current_prompt (no-op) — caller detects this.

    Raises:
        ValueError: If action_id not in {0,1,2,3,4}.
    """
    if action_id == 0:
        return add_context(current_prompt, task.context_sentence)
    elif action_id == 1:
        return shorten(current_prompt)
    elif action_id == 2:
        return add_example(current_prompt, task.example_output)
    elif action_id == 3:
        return rephrase(current_prompt)
    elif action_id == 4:
        return add_constraint(current_prompt, task.constraint_sentence)
    else:
        raise ValueError(
            f"Invalid action_id={action_id} passed to apply_action. "
            f"Must be in {{0,1,2,3,4}}. Action 5 (STOP) is handled by the environment."
        )


def add_context(prompt: str, context_sentence: str) -> str:
    """Append domain context if not already present."""
    if context_sentence[:30].lower() in prompt.lower():
        return prompt  # no-op
    return f"{prompt}\nContext: {context_sentence}"


def shorten(prompt: str) -> str:
    """Remove filler phrases. The only action that reduces token count."""
    result = prompt
    for pattern in _FILLER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result if result != prompt else prompt


def add_example(prompt: str, example_output: str) -> str:
    """Append example output format if not already present."""
    if "Example output format:" in prompt:
        return prompt  # no-op
    return f"{prompt}\nExample output format: {example_output}"


def rephrase(prompt: str) -> str:
    """Convert question phrasing to direct imperative. Net-neutral token change."""
    result = prompt
    result = re.sub(
        r"^(?:can you|could you|would you|please)\s+(.+?)[?\.]*$",
        lambda m: m.group(1).strip().capitalize() + ".",
        result, flags=re.IGNORECASE | re.MULTILINE,
    )
    result = re.sub(
        r"^i (?:want|need|would like) you to\s+",
        "", result, flags=re.IGNORECASE | re.MULTILINE,
    )
    result = result.strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result if result != prompt else prompt


def add_constraint(prompt: str, constraint_sentence: str) -> str:
    """Append output constraint if not already present."""
    if "Requirement:" in prompt:
        return prompt  # no-op
    return f"{prompt}\nRequirement: {constraint_sentence}"
