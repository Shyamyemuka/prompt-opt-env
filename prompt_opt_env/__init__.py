"""
Prompt Opt Env / PromptRL Environment — public API.
"""

from .models import PromptAction, PromptObservation
from .models import PromptOptAction, PromptOptObservation  # backward-compat aliases
from .client import PromptRLEnv, PromptOptEnv  # backward-compat alias

__all__ = [
    # Canonical names (per BACKEND_STRUCTURE.md)
    "PromptAction",
    "PromptObservation",
    "PromptRLEnv",
    # Legacy aliases
    "PromptOptAction",
    "PromptOptObservation",
    "PromptOptEnv",
]
