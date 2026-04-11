"""
Provider-cycling LLM router for OpenAI-compatible endpoints.

Supports OpenAI/Gemini/HF-style providers with automatic failover and
round-robin success cycling.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from openai import OpenAI

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HF_DEFAULT_BASE_URL = "https://router.huggingface.co/v1/"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str


def _normalize_base_url(url: str) -> str:
    clean = (url or "").strip().strip("'").strip('"')
    if clean and not clean.endswith("/"):
        clean += "/"
    return clean


def _looks_provider_specific_model(model_name: str) -> bool:
    """Detect model IDs that are unlikely to be valid on native OpenAI endpoints."""
    model = (model_name or "").strip().lower()
    if not model:
        return False
    if "/" in model:
        return True
    return model.startswith("gemini")


def build_provider_specs(default_model: str, default_base_url: str = "") -> list[ProviderSpec]:
    """
    Build provider list from environment.

    Order matters and defines the round-robin cycle:
    1) OpenAI
    2) Gemini
    3) HF/OpenAI-compatible fallback
    """
    providers: list[ProviderSpec] = []

    # OpenAI
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key:
        openai_base_url = _normalize_base_url(
            os.getenv("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE_URL)
        )
        explicit_openai_model = (os.getenv("OPENAI_MODEL") or "").strip()
        if explicit_openai_model:
            openai_model = explicit_openai_model
        elif "api.openai.com" in openai_base_url and _looks_provider_specific_model(default_model):
            # Avoid sending HF/Gemini model IDs to native OpenAI, which returns invalid model ID.
            openai_model = "gpt-4o-mini"
        else:
            openai_model = default_model

        providers.append(
            ProviderSpec(
                name="openai",
                base_url=openai_base_url,
                api_key=openai_key,
                model=(openai_model or "gpt-4o-mini").strip(),
            )
        )

    # Gemini (OpenAI-compatible endpoint)
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if gemini_key:
        providers.append(
            ProviderSpec(
                name="gemini",
                base_url=_normalize_base_url(
                    os.getenv("GEMINI_BASE_URL", GEMINI_DEFAULT_BASE_URL)
                ),
                api_key=gemini_key,
                model=(os.getenv("GEMINI_MODEL") or default_model).strip() or default_model,
            )
        )

    # HF/OpenAI-compatible fallback
    hf_key = (os.getenv("HF_TOKEN") or "").strip()
    if hf_key:
        providers.append(
            ProviderSpec(
                name="hf",
                base_url=_normalize_base_url(
                    os.getenv(
                        "HF_BASE_URL",
                        os.getenv("API_BASE_URL", default_base_url or HF_DEFAULT_BASE_URL),
                    )
                ),
                api_key=hf_key,
                model=(os.getenv("HF_MODEL") or default_model).strip() or default_model,
            )
        )

    # Legacy single key fallback for compatibility
    legacy_key = (os.getenv("API_KEY") or "").strip()
    if legacy_key and not providers:
        providers.append(
            ProviderSpec(
                name="legacy",
                base_url=_normalize_base_url(default_base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1/")),
                api_key=legacy_key,
                model=default_model,
            )
        )

    # Deduplicate exact duplicates while preserving order.
    deduped: list[ProviderSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for provider in providers:
        key = (provider.name, provider.base_url, provider.model)
        if key not in seen:
            deduped.append(provider)
            seen.add(key)

    return deduped


class LLMRouter:
    """Round-robin provider router with failover per request."""

    def __init__(self, providers: list[ProviderSpec], timeout_seconds: float = 45, max_retries: int = 2):
        self._providers = providers
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._clients: dict[str, OpenAI] = {}
        self._next_index = 0
        self._lock = threading.Lock()
        self.last_error = ""
        self.last_provider = ""
        self.last_finish_reason = ""

    def _client_for(self, provider: ProviderSpec) -> OpenAI:
        client = self._clients.get(provider.name)
        if client is None:
            client = OpenAI(
                base_url=provider.base_url,
                api_key=provider.api_key,
                max_retries=self._max_retries,
                timeout=self._timeout_seconds + 5,
            )
            self._clients[provider.name] = client
        return client

    def has_provider(self) -> bool:
        return bool(self._providers)

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.3,
    ) -> str:
        """
        Attempt request across providers in a cycle.
        Success advances starting provider for next call.
        """
        if not self._providers:
            self.last_error = "no_provider_configured"
            self.last_provider = ""
            self.last_finish_reason = ""
            return ""

        with self._lock:
            start_index = self._next_index

        errors: list[str] = []
        total = len(self._providers)

        for offset in range(total):
            idx = (start_index + offset) % total
            provider = self._providers[idx]
            try:
                client = self._client_for(provider)
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self._timeout_seconds,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    errors.append(f"{provider.name} -> empty_response")
                    continue

                with self._lock:
                    self._next_index = idx
                self.last_error = ""
                self.last_provider = provider.name
                self.last_finish_reason = (response.choices[0].finish_reason or "").strip()
                return text
            except Exception as exc:  # pragma: no cover - provider-specific runtime errors
                errors.append(f"{provider.name} -> {type(exc).__name__}: {str(exc)[:140]}")

        with self._lock:
            self._next_index = (start_index + 1) % total
        self.last_provider = ""
        self.last_finish_reason = ""
        self.last_error = " | ".join(errors)[:500]
        return ""

    def status(self) -> dict[str, object]:
        return {
            "has_provider": bool(self._providers),
            "providers": [
                {"name": p.name, "base_url": p.base_url, "model": p.model}
                for p in self._providers
            ],
            "active_provider": self.last_provider,
            "last_error": self.last_error,
            "last_finish_reason": self.last_finish_reason,
        }


def create_default_router(
    default_model: str,
    default_base_url: str = "",
    timeout_seconds: float = 45,
    max_retries: int = 2,
) -> LLMRouter:
    providers = build_provider_specs(default_model=default_model, default_base_url=default_base_url)
    return LLMRouter(providers=providers, timeout_seconds=timeout_seconds, max_retries=max_retries)
