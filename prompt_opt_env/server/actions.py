"""
All 5 deterministic prompt-editing action functions.

No LLM calls. No randomness. Pure string transformations.
Each action operates on the current prompt string and returns a new prompt string.
"""

import re


ACTION_NAMES: dict[int, str] = {
    0: "ADD_CONTEXT",
    1: "SHORTEN",
    2: "ADD_EXAMPLE",
    3: "REPHRASE",
    4: "ADD_CONSTRAINT",
}

FILLER_PHRASES = [
    r"\bplease\b",
    r"\bcould you\b",
    r"\bi would like you to\b",
    r"\bi want you to\b",
    r"\bcan you\b",
    r"\bwould you\b",
    r"\bkindly\b",
    r"\bif possible\b",
    r"\bjust\b",
]


def apply_action(action_id: int, current_prompt: str, task) -> str:
    """
    Apply the specified action to the current prompt.

    Args:
        action_id: Integer 0–4 identifying which action to apply.
        current_prompt: The current prompt string to edit.
        task: Task dataclass from task_bank.py (provides context/example/constraint strings).

    Returns:
        The new prompt string after applying the action.

    Raises:
        ValueError: If action_id is not in the range 0–4.
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
            f"Invalid action_id={action_id}. "
            f"Must be an integer in {{0, 1, 2, 3, 4}}. "
            f"Received type={type(action_id).__name__}."
        )


def add_context(prompt: str, context_sentence: str) -> str:
    """
    Append domain context to the prompt if not already present.

    Args:
        prompt: Current prompt string.
        context_sentence: Domain-relevant context sentence from the task.

    Returns:
        Prompt with context appended, or original if context already present.
    """
    if context_sentence.lower()[:30] in prompt.lower():
        return prompt  # already present, no-op
    return f"{prompt}\nContext: {context_sentence}"


def shorten(prompt: str) -> str:
    """
    Remove filler words and redundant phrases from the prompt.

    Targets: 'please', 'could you', 'can you', 'would you', 'kindly', etc.
    If prompt is already short (<50 chars) and no fillers found, returns unchanged.

    Args:
        prompt: Current prompt string.

    Returns:
        Shortened prompt, or original if no changes were made.
    """
    result = prompt
    for pattern in FILLER_PHRASES:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # Collapse multiple spaces
    result = re.sub(r"  +", " ", result).strip()
    # Capitalise first character if present
    if result:
        result = result[0].upper() + result[1:]
    return result if result != prompt else prompt


def add_example(prompt: str, example_output: str) -> str:
    """
    Append an example of the desired output format to the prompt.

    Args:
        prompt: Current prompt string.
        example_output: Example output string from the task.

    Returns:
        Prompt with example appended, or original if already present.
    """
    marker = "Example output format:"
    if marker in prompt:
        return prompt  # already present, no-op
    return f"{prompt}\nExample output format: {example_output}"


def rephrase(prompt: str) -> str:
    """
    Convert passive or question phrasing to direct imperative form.

    Converts:
      - 'Can you explain X?' → 'Explain X.'
      - 'Could you X?' → 'X.'
      - 'I want you to X' → 'X.'
      - 'I need you to X' → 'X.'

    Args:
        prompt: Current prompt string.

    Returns:
        Rephrased prompt in imperative voice, or original if no change.
    """
    result = prompt
    # Convert "Can you X?" / "Could you X?" / "Would you X?" / "Please X?" → "X."
    result = re.sub(
        r"^(?:can you|could you|would you|please)\s+(.+?)[?.]*$",
        lambda m: m.group(1).capitalize() + ".",
        result,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Convert "I want you to X" / "I need you to X" / "I would like you to X" → "X"
    result = re.sub(
        r"^i (?:want|need|would like) you to\s+",
        "",
        result,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    result = result.strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result if result != prompt else prompt


def add_constraint(prompt: str, constraint_sentence: str) -> str:
    """
    Append an explicit output constraint to the prompt.

    Args:
        prompt: Current prompt string.
        constraint_sentence: Constraint string from the task (format, length, style).

    Returns:
        Prompt with constraint appended, or original if already present.
    """
    marker = "Requirement:"
    if marker in prompt:
        return prompt  # already present, no-op
    return f"{prompt}\nRequirement: {constraint_sentence}"
