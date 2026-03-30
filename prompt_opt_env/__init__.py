"""
Prompt Opt Env / PromptOptEnv Environment — public API.
"""

from .models import PromptAction, PromptObservation
from .models import PromptOptAction, PromptOptObservation  # backward-compat aliases
from .client import PromptOptEnvEnv, PromptOptEnv  # backward-compat alias

__all__ = [
    # Canonical names (per BACKEND_STRUCTURE.md)
    "PromptAction",
    "PromptObservation",
    "PromptOptEnvEnv",
    # Legacy aliases
    "PromptOptAction",
    "PromptOptObservation",
    "PromptOptEnv",
]
