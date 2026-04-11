"""
Web front-end for PromptOptEnv - Interactive Prompt Optimizer

Run:
    python web_app.py

The server starts automatically, loads .env.local, and opens your browser.
"""
import os
import sys
import threading
import webbrowser
from typing import Any

# ── Resolve project root relative to this file ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Auto-load .env.local (HF_TOKEN, MODEL_NAME, etc.) ────────────────────────
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(BASE_DIR, ".env.local")
    if os.path.exists(_env_file):
        load_dotenv(_env_file, override=True)
        print(f"[INFO] Loaded environment from {_env_file}")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell env

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Import optimizer components
try:
    from prompt_opt_env.server.actions import (
        count_tokens, add_context, shorten, add_example, rephrase, add_constraint
    )
except (ImportError, ModuleNotFoundError):
    from server.actions import (
        count_tokens, add_context, shorten, add_example, rephrase, add_constraint
    )

from openai import OpenAI
from rouge_score import rouge_scorer
try:
    from .llm_router import create_default_router
except Exception:
    from llm_router import create_default_router

# ── Environment variables (loaded from .env.local above) ─────────────────────
def _first_env(*keys: str, default: str = "") -> str:
    """Return first non-empty environment value among aliases."""
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _normalize_base_url(url: str) -> str:
    """Normalize endpoint URL and ensure trailing slash for client consistency."""
    clean = (url or "").strip().strip("'").strip('"')
    if clean and not clean.endswith("/"):
        clean += "/"
    return clean


def _build_api_base_candidates(primary_url: str, fallback_url: str) -> list[str]:
    """Build deduplicated endpoint candidates with safe defaults for HF router outages."""
    candidates: list[str] = []
    for raw_url in (primary_url, fallback_url):
        normalized = _normalize_base_url(raw_url)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    if "router.huggingface.co" in (primary_url or ""):
        alt_hf = "https://api-inference.huggingface.co/v1/"
        if alt_hf not in candidates:
            candidates.append(alt_hf)

    return candidates


API_BASE_URL: str = _normalize_base_url(
    _first_env("API_BASE_URL", "OPENAI_BASE_URL", default="https://router.huggingface.co/v1/")
)
API_BASE_URL_FALLBACK: str = _normalize_base_url(
    _first_env("API_BASE_URL_FALLBACK", "OPENAI_BASE_URL_FALLBACK", default="")
)
MODEL_NAME: str = _first_env("MODEL_NAME", "OPENAI_MODEL", default="Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = _first_env("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", default="")
ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
DONE_THRESHOLD: float = float(os.getenv("DONE_THRESHOLD", "0.85"))
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "7"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
OPTIMIZER_FAST_MODE: bool = _env_bool("OPTIMIZER_FAST_MODE", False)
MAX_LLM_CALLS_PER_RUN: int = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "18"))
MAX_RESCUE_TRIALS: int = int(os.getenv("MAX_RESCUE_TRIALS", "10"))
OUTPUT_TOKEN_COST: float = float(os.getenv("OUTPUT_TOKEN_COST", "1.0"))
USE_INTELLIGENT_ACTIONS: bool = _env_bool("USE_INTELLIGENT_ACTIONS", True)

if not (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or HF_TOKEN):
    print("[WARNING] No LLM provider key found. Set OPENAI_API_KEY and/or GEMINI_API_KEY (or HF_TOKEN).")
if any(
    os.getenv(name)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
):
    print("[INFO] Proxy environment variables detected. OpenAI client uses direct connection (trust_env=False).")

_CLIENTS: dict[str, OpenAI] = {}
_API_BASE_URL_CANDIDATES: list[str] = _build_api_base_candidates(API_BASE_URL, API_BASE_URL_FALLBACK)
if not _API_BASE_URL_CANDIDATES and API_BASE_URL:
    _API_BASE_URL_CANDIDATES = [API_BASE_URL]
_ACTIVE_API_BASE_URL: str = _API_BASE_URL_CANDIDATES[0] if _API_BASE_URL_CANDIDATES else API_BASE_URL
_LAST_LLM_ERROR = ""
_LAST_LLM_FINISH_REASON = ""
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
SCORE_EPSILON = 0.11
_LLM_ROUTER = create_default_router(
    default_model=MODEL_NAME,
    default_base_url=API_BASE_URL,
    timeout_seconds=LLM_TIMEOUT_SECONDS,
    max_retries=LLM_MAX_RETRIES,
)


def get_client(base_url: str | None = None):
    """Lazy load OpenAI client per endpoint."""
    target_base_url = _normalize_base_url(base_url or _ACTIVE_API_BASE_URL or API_BASE_URL)
    if not HF_TOKEN or not target_base_url:
        return None
    if target_base_url not in _CLIENTS:
        _CLIENTS[target_base_url] = OpenAI(
            base_url=target_base_url,
            api_key=HF_TOKEN,
            max_retries=LLM_MAX_RETRIES,
            timeout=LLM_TIMEOUT_SECONDS + 5,
        )
    return _CLIENTS[target_base_url]


def _is_retryable_connection_error(exc: Exception) -> bool:
    """Retry transient transport/server issues, but fail fast on hard API errors."""
    error_name = type(exc).__name__.lower()
    if any(token in error_name for token in ("connection", "timeout", "rate", "server")):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


def _ordered_api_base_urls() -> list[str]:
    """Try last-known good endpoint first, then remaining configured candidates."""
    ordered: list[str] = []
    for raw_url in [_ACTIVE_API_BASE_URL, *_API_BASE_URL_CANDIDATES]:
        normalized = _normalize_base_url(raw_url)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call LLM with the prompt and return output."""
    global _LAST_LLM_ERROR, _LAST_LLM_FINISH_REASON
    if not _LLM_ROUTER.has_provider():
        _LAST_LLM_ERROR = "missing_api_key"
        _LAST_LLM_FINISH_REASON = ""
        return ""
    text = _LLM_ROUTER.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    _LAST_LLM_ERROR = _LLM_ROUTER.last_error
    _LAST_LLM_FINISH_REASON = _LLM_ROUTER.last_finish_reason
    return text


def get_llm_runtime_status() -> dict[str, Any]:
    """Expose sanitized runtime status for debugging connection/key issues."""
    router_status = _LLM_ROUTER.status()
    providers = router_status.get("providers", [])
    active_provider = str(router_status.get("active_provider") or "").strip()
    selected_provider = None
    for provider in providers:
        if provider.get("name") == active_provider:
            selected_provider = provider
            break
    if selected_provider is None and providers:
        selected_provider = providers[0]

    first_base = selected_provider["base_url"] if selected_provider else API_BASE_URL
    model_name = selected_provider.get("model") if selected_provider else MODEL_NAME
    return {
        "api_base_url": first_base,
        "api_base_candidates": _API_BASE_URL_CANDIDATES,
        "model_name": model_name,
        "token_present": bool(router_status.get("has_provider")),
        "providers": providers,
        "active_provider": active_provider,
        "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
        "llm_max_retries": LLM_MAX_RETRIES,
        "optimizer_fast_mode": OPTIMIZER_FAST_MODE,
        "max_llm_calls_per_run": MAX_LLM_CALLS_PER_RUN,
        "proxy_env_present": any(
            bool(os.getenv(name))
            for name in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy",
            )
        ),
        "last_llm_error": router_status.get("last_error") or _LAST_LLM_ERROR,
        "last_finish_reason": router_status.get("last_finish_reason") or _LAST_LLM_FINISH_REASON,
    }


def _build_fallback_notice(
    used_fallback: bool,
    offline_mode: bool,
    fallback_calls: int,
    llm_calls_attempted: int,
    fallback_reason: str,
) -> dict[str, Any]:
    """Return a user-facing notice that is quiet for benign failover and loud for actionable issues."""
    if not used_fallback:
        return {
            "show": False,
            "severity": "info",
            "title": "",
            "message": "",
            "action": "",
            "fallback_calls": fallback_calls,
            "llm_calls_attempted": llm_calls_attempted,
            "raw_reason": "",
        }

    attempts = max(1, llm_calls_attempted)
    ratio = fallback_calls / attempts
    reason_clean = " ".join((fallback_reason or "").split())
    reason = reason_clean.lower()

    if any(token in reason for token in ("missing_api_key", "no_provider_configured")):
        category = "credentials"
    elif any(token in reason for token in ("401", "unauthorized", "invalid_api_key", "authentication")):
        category = "credentials"
    elif any(token in reason for token in ("invalid model", "model_not_found", "invalid model id")):
        category = "model"
    elif any(token in reason for token in ("402", "quota", "billing", "credits", "depleted", "429", "rate limit")):
        category = "quota"
    elif any(token in reason for token in ("timeout", "connection", "503", "502", "504", "temporar", "server")):
        category = "transient"
    else:
        category = "unknown"

    # Suppress noisy notices for rare transient misses when most calls succeeded.
    if category == "transient" and not offline_mode and ratio <= 0.15:
        return {
            "show": False,
            "severity": "info",
            "title": "",
            "message": "",
            "action": "",
            "fallback_calls": fallback_calls,
            "llm_calls_attempted": llm_calls_attempted,
            "raw_reason": reason_clean,
        }

    severity = "warning"
    title = "Partial fallback outputs were used."
    message = "Some generations used fallback text because all providers failed for those calls."
    action = ""

    if offline_mode:
        severity = "danger"
        title = "Offline mode: all LLM calls failed."

    if category == "quota":
        if offline_mode:
            severity = "danger"
        title = "Provider quota/credits exceeded during this run."
        message = "Failover is working, but at least one provider is out of quota/credits."
        action = "Top up Hugging Face credits and/or increase OpenAI quota to remove fallback usage."
    elif category == "credentials":
        severity = "danger"
        title = "Provider authentication is misconfigured."
        message = "At least one provider key is invalid or missing for current endpoint usage."
        action = "Verify OPENAI_API_KEY, GEMINI_API_KEY, and HF_TOKEN in Space secrets."
    elif category == "model":
        severity = "danger"
        title = "Provider model configuration mismatch detected."
        message = "A provider rejected the configured model ID for its endpoint."
        action = "Set provider-specific model vars (OPENAI_MODEL, GEMINI_MODEL, HF_MODEL) to valid IDs."
    elif category == "transient":
        severity = "warning" if offline_mode or ratio > 0.3 else "info"
        title = "Temporary provider/network instability detected."
        message = "Most runs should still complete, but some calls timed out or failed transiently."
        action = "If frequent, increase LLM timeout or reduce concurrent traffic."

    return {
        "show": True,
        "severity": severity,
        "title": title,
        "message": message,
        "action": action,
        "fallback_calls": fallback_calls,
        "llm_calls_attempted": llm_calls_attempted,
        "raw_reason": reason_clean,
    }


def score_output(output: str, reference: str, task: str = "General") -> float:
    """Compute ROUGE-L score with task-aware calibration."""
    if (
        not output
        or not reference
        or output.startswith("[Error:")
        or reference.startswith("[Error:")
    ):
        return SCORE_EPSILON
    base_score = _ROUGE.score(reference, output)["rougeL"].fmeasure

    # Summaries should be concise; penalize overly long outputs against compact references.
    if task == "Summarization":
        out_len = max(1, len(output.split()))
        ref_len = max(1, len(reference.split()))
        if out_len > ref_len:
            brevity_factor = max(0.55, min(1.0, (ref_len / out_len) * 1.25))
            base_score *= brevity_factor

    bounded = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, float(base_score)))
    return round(bounded, 4)


def clip_reward(value: float, lower: float = -2.0, upper: float = 2.0) -> float:
    """Clip reward to stable bounds."""
    return round(max(lower, min(upper, value)), 4)


def build_inference_prompt(task: str, instruction_prompt: str, input_data: str) -> str:
    """Build model input while keeping optimizable prompt separate from user content."""
    prompt = instruction_prompt.strip()
    data = input_data.strip()
    if not data:
        return prompt

    if task == "Question Answering":
        header = "Context and Question"
    elif task == "Paraphrasing":
        header = "Sentence"
    elif task == "Instruction Following":
        header = "Instruction Input"
    else:
        header = "Input"

    return f"{prompt}\n\n{header}:\n{data}"


def build_reference_prompt(task: str, input_data: str) -> str:
    """Ask the LLM for an expert answer used as a scoring reference."""
    task_guidance = {
        "Summarization": "Produce a concise summary in 2-3 sentences and keep it under 70 words.",
        "Question Answering": "Answer accurately using only the given context and question in 2-4 concise sentences under 90 words.",
        "Paraphrasing": "Rewrite the sentence in one concise sentence with the same meaning and better clarity.",
        "Instruction Following": "Follow the instruction precisely and produce a concise result under 120 words.",
    }
    guidance = task_guidance.get(task, "Produce the highest quality response.")
    data = input_data.strip() or "No additional input provided."
    return (
        "You are an expert assistant.\n"
        f"Task type: {task}\n"
        f"Goal: {guidance}\n\n"
        "User-provided input:\n"
        f"{data}\n\n"
        "Return only the final answer text."
    )


def response_token_limit(task: str, goal: str) -> int:
    """Task-aware output cap to avoid overly verbose generations."""
    if task == "Summarization":
        if goal == "Low Cost":
            return 80
        if goal == "High Quality":
            return 110
        return 95
    if task == "Paraphrasing":
        if goal == "Low Cost":
            return 70
        if goal == "High Quality":
            return 100
        return 85
    if task == "Question Answering":
        if goal == "Low Cost":
            return 75
        if goal == "High Quality":
            return 110
        return 90
    if task == "Instruction Following":
        if goal == "Low Cost":
            return 90
        if goal == "High Quality":
            return 130
        return 110
    return 120


def _first_n_words(text: str, n: int) -> str:
    """Return first n words and add ellipsis when truncated."""
    words = text.split()
    if not words:
        return ""
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]).strip() + " ..."


def _normalized_text_signature(text: str) -> str:
    """Normalize text for fair equality checks across whitespace/casing differences."""
    return " ".join((text or "").strip().lower().split())


def _parse_qa_input(input_data: str) -> tuple[str, str]:
    """Extract context and question from QA combined input."""
    context = ""
    question = ""
    for line in input_data.splitlines():
        lower = line.lower().strip()
        if lower.startswith("context:"):
            context = line.split(":", 1)[1].strip()
        elif lower.startswith("question:"):
            question = line.split(":", 1)[1].strip()
    return context, question


def fallback_output(task: str, prompt: str, input_data: str, is_reference: bool = False) -> str:
    """
    Deterministic fallback output for offline/API-failure cases.
    Keeps the UI functional for hackathon demos when endpoint calls fail.
    """
    prompt_lower = prompt.lower()
    data = input_data.strip()

    if task == "Question Answering":
        context, question = _parse_qa_input(data)
        source = context or data
        if not source:
            return "No context provided."
        answer_len = 32 if is_reference else 16
        if "context" in prompt_lower:
            answer_len += 6
        if "requirement:" in prompt_lower or "example output format:" in prompt_lower:
            answer_len += 6
        if is_reference:
            prefix = f"Best answer to '{question}': " if question else "Best answer: "
        else:
            prefix = f"Answer to '{question}': " if question else "Answer: "
        return prefix + _first_n_words(source, answer_len)

    if task == "Paraphrasing":
        sentence = data or "No sentence provided."
        prefix = "Paraphrased:" if not is_reference else "Improved paraphrase:"
        if "concise" in prompt_lower:
            sentence = _first_n_words(sentence, 18)
        suffix = " Keep the meaning unchanged." if is_reference else ""
        return f"{prefix} {sentence}{suffix}"

    if task == "Instruction Following":
        instruction = data or "No instruction provided."
        result_len = 40 if is_reference else 22
        if "requirement:" in prompt_lower:
            result_len += 8
        if is_reference:
            return "Expected result:\n1. Understand the instruction.\n2. Execute it.\n3. Deliver: " + _first_n_words(instruction, result_len)
        return "Completed task: " + _first_n_words(instruction, result_len)

    # Default to summarization behavior.
    text = data or "No input text provided."
    if is_reference:
        words = text.split()
        if not words:
            return "No input text provided."
        if len(words) <= 16:
            return "Summary: " + " ".join(words[:10])
        first = " ".join(words[:8])
        mid_start = max(8, len(words) // 2 - 4)
        middle = " ".join(words[mid_start:mid_start + 8])
        last = " ".join(words[-8:])
        return f"- {first}\n- {middle}\n- {last}"

    summary_len = 24
    if "concise" in prompt_lower or "summarize" in prompt_lower:
        summary_len -= 2
    if "example output format:" in prompt_lower:
        summary_len += 2
    if "context:" in prompt_lower:
        summary_len += 1
    if "requirement:" in prompt_lower:
        summary_len -= 7
    if "under 50 words" in prompt_lower or "exactly 2 sentences" in prompt_lower:
        summary_len = min(summary_len, 15)
    elif "under 70 words" in prompt_lower or "2-3 sentences" in prompt_lower:
        summary_len = min(summary_len, 19)
    summary_len = max(10, summary_len)
    return _first_n_words(text, summary_len)


def _build_action_rewrite_instruction(
    action_id: int,
    task: str,
    goal: str,
    input_data: str,
    context_text: str,
    example: str,
    constraint: str,
) -> str:
    """Create a task-aware rewrite instruction for one action."""
    if action_id == 0:
        return (
            "Rewrite the prompt to naturally incorporate relevant context. "
            "Do not append a raw 'Context:' line.\n"
            f"Task: {task}\nGoal: {goal}\nContext to include: {context_text or input_data[:140].strip()}"
        )
    if action_id == 1:
        return (
            "Rewrite the prompt to be shorter and clearer while preserving intent and constraints. "
            "Remove filler and redundancy."
        )
    if action_id == 2:
        return (
            "Rewrite the prompt to include a compact example of desired output format in a natural way. "
            "Avoid boilerplate labels unless needed.\n"
            f"Example to include: {example}"
        )
    if action_id == 3:
        return (
            "Rewrite the prompt for directness and clarity using imperative style. "
            "Fix obvious spelling issues if present."
        )
    if action_id == 4:
        return (
            "Rewrite the prompt to integrate constraints naturally and explicitly. "
            "Do not append a detached requirement line.\n"
            f"Constraint to enforce: {constraint}"
        )
    return "Keep the prompt unchanged."


def _apply_action_for_web_rule_based(
    action_id: int,
    task: str,
    current_prompt: str,
    input_data: str,
    goal: str,
) -> str:
    """Deterministic fallback action transformation for the web optimizer."""
    if action_id == 0:
        if task == "Summarization":
            context_text = "Focus only on key events, actions, and outcomes."
        else:
            context_text = input_data[:80].strip()
        return add_context(current_prompt, context_text) if context_text else current_prompt
    if action_id == 1:
        return shorten(current_prompt)
    if action_id == 2:
        if task == "Summarization":
            if goal == "Low Cost":
                example = "2 concise sentences: who/what happened and why it matters."
            else:
                example = "3 concise sentences covering setup, action, and outcome."
        elif task == "Question Answering":
            example = (
                "Answer in 3-4 short sentences with only essential calculations."
                if goal == "High Quality"
                else "Answer directly in a short paragraph with essential steps only."
            )
        elif task == "Paraphrasing":
            example = "Rewrite in one concise sentence while preserving meaning."
        elif task == "Instruction Following":
            example = "Deliver a concise result with only required details."
        else:
            example = (
                "Give a clear answer with strong structure."
                if goal == "High Quality"
                else "Answer directly in one short paragraph."
            )
        return add_example(current_prompt, example)
    if action_id == 3:
        return rephrase(current_prompt)
    if action_id == 4:
        if task == "Summarization":
            if goal == "Low Cost":
                constraint = "Write exactly 2 sentences and keep total length under 50 words."
            elif goal == "High Quality":
                constraint = "Write 3 concise sentences under 90 words with high factual fidelity."
            else:
                constraint = "Write 2-3 sentences under 70 words with only essential details."
        elif task == "Question Answering":
            if goal == "Low Cost":
                constraint = "Answer accurately in at most 70 words using only essential calculations."
            elif goal == "High Quality":
                constraint = "Answer accurately in at most 110 words with only required calculations."
            else:
                constraint = "Answer accurately in at most 90 words with essential calculations only."
        elif task == "Paraphrasing":
            constraint = "Return one concise sentence under 45 words with same meaning."
        elif task == "Instruction Following":
            constraint = (
                "Follow all requirements in under 90 words."
                if goal == "Low Cost"
                else "Follow all requirements in under 120 words."
            )
        else:
            constraint = (
                "Keep the answer under 80 words."
                if goal == "Low Cost"
                else "Ensure factual accuracy and clear structure."
            )
        return add_constraint(current_prompt, constraint)
    return current_prompt


def apply_action_for_web(
    action_id: int,
    task: str,
    current_prompt: str,
    input_data: str,
    goal: str,
    use_intelligent_actions: bool = True,
) -> str:
    """Apply intelligent rewrite first, then deterministic fallback."""
    fallback_prompt = _apply_action_for_web_rule_based(action_id, task, current_prompt, input_data, goal)
    if not use_intelligent_actions or action_id not in (0, 1, 2, 3, 4):
        return fallback_prompt

    # Build task-specific values used by both fallback and rewrite prompts.
    if task == "Summarization":
        context_text = "Focus only on key events, actions, and outcomes."
        if goal == "Low Cost":
            example = "2 concise sentences: who/what happened and why it matters."
            constraint = "Write exactly 2 sentences and keep total length under 50 words."
        elif goal == "High Quality":
            example = "3 concise sentences covering setup, action, and outcome."
            constraint = "Write 3 concise sentences under 90 words with high factual fidelity."
        else:
            example = "3 concise sentences covering setup, action, and outcome."
            constraint = "Write 2-3 sentences under 70 words with only essential details."
    elif task == "Question Answering":
        context_text = input_data[:120].strip()
        example = (
            "Answer in 3-4 short sentences with only essential calculations."
            if goal == "High Quality"
            else "Answer directly in a short paragraph with essential steps only."
        )
        if goal == "Low Cost":
            constraint = "Answer accurately in at most 70 words using only essential calculations."
        elif goal == "High Quality":
            constraint = "Answer accurately in at most 110 words with only required calculations."
        else:
            constraint = "Answer accurately in at most 90 words with essential calculations only."
    elif task == "Paraphrasing":
        context_text = input_data[:120].strip()
        example = "Rewrite in one concise sentence while preserving meaning."
        constraint = "Return one concise sentence under 45 words with same meaning."
    elif task == "Instruction Following":
        context_text = input_data[:120].strip()
        example = "Deliver a concise result with only required details."
        constraint = (
            "Follow all requirements in under 90 words."
            if goal == "Low Cost"
            else "Follow all requirements in under 120 words."
        )
    else:
        context_text = input_data[:120].strip()
        example = (
            "Give a clear answer with strong structure."
            if goal == "High Quality"
            else "Answer directly in one short paragraph."
        )
        constraint = (
            "Keep the answer under 80 words."
            if goal == "Low Cost"
            else "Ensure factual accuracy and clear structure."
        )

    instruction = _build_action_rewrite_instruction(
        action_id,
        task,
        goal,
        input_data,
        context_text,
        example,
        constraint,
    )
    rewrite_prompt = (
        f"{instruction}\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        "Return only the rewritten prompt. No commentary."
    )
    rewrite_tokens = max(120, min(260, count_tokens(current_prompt) * 3 + 40))
    rewritten = (call_llm(rewrite_prompt, max_tokens=rewrite_tokens) or "").strip()
    if rewritten.startswith("```"):
        lines = rewritten.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            rewritten = "\n".join(lines[1:-1]).strip()
    if rewritten.startswith('"') and rewritten.endswith('"') and len(rewritten) >= 2:
        rewritten = rewritten[1:-1].strip()
    if rewritten.startswith("'") and rewritten.endswith("'") and len(rewritten) >= 2:
        rewritten = rewritten[1:-1].strip()

    return rewritten if rewritten else fallback_prompt


def intelligent_action_selection(
    task: str,
    goal: str,
    current_prompt: str,
    current_score: float,
    tokens_remaining: int,
    step: int,
) -> int:
    """Select best action based on current state."""
    prompt_lower = current_prompt.lower()
    prompt_tokens = count_tokens(current_prompt)

    if task == "Summarization":
        if current_score >= 0.70:
            return 5

        if "please" in prompt_lower or "could you" in prompt_lower:
            return 1

        if step == 0:
            return 3

        if (
            current_score < 0.60
            and "requirement:" not in prompt_lower
            and tokens_remaining > 10
            and prompt_tokens < 16
        ):
            return 4

        if (
            "example output format:" not in prompt_lower
            and goal == "High Quality"
            and tokens_remaining > 18
            and step <= 1
            and current_score < 0.58
        ):
            return 2

        if goal == "Low Cost" and prompt_tokens > 8:
            return 1

        return 5

    if current_score >= 0.80:
        return 5
    if tokens_remaining <= 8:
        return 1 if ("please" in prompt_lower or "could you" in prompt_lower) else 5
    if "please" in prompt_lower or "could you" in prompt_lower:
        return 1
    if step == 1 and "example output format:" not in prompt_lower and tokens_remaining > 20:
        return 2
    if step >= 2 and "requirement:" not in prompt_lower and tokens_remaining > 18:
        return 4
    if current_prompt.endswith("?") or "can you" in prompt_lower:
        return 3
    if tokens_remaining > 24 and step == 0:
        return 0
    if current_score > 0.70 and step >= 2:
        return 5
    return 3


def optimize_prompt_with_state(state: dict) -> dict:
    """Optimize a prompt using RL-based action selection with state object."""

    # Extract from state
    task = state.get("task", "General")
    prompt = (state.get("prompt", "") or "").strip()
    input_data = (state.get("input", "") or "").strip()
    goal = state.get("goal", "Balanced")
    use_intelligent_actions = USE_INTELLIGENT_ACTIONS and _LLM_ROUTER.has_provider()

    effective_max_steps = MAX_STEPS
    if OPTIMIZER_FAST_MODE:
        if goal == "High Quality":
            effective_max_steps = min(MAX_STEPS, 5)
        elif goal == "Balanced":
            effective_max_steps = min(MAX_STEPS, 4)
        else:
            effective_max_steps = min(MAX_STEPS, 3)

    # Generate a reference answer (ground truth)
    used_fallback = False
    fallback_reasons: list[str] = []
    llm_calls_attempted = 0
    fallback_calls = 0

    def record_fallback(default_reason: str) -> None:
        nonlocal used_fallback, fallback_calls
        used_fallback = True
        fallback_calls += 1
        reason = (_LAST_LLM_ERROR or default_reason).strip()
        if reason and reason not in fallback_reasons:
            fallback_reasons.append(reason)

    def last_response_truncated() -> bool:
        return (_LAST_LLM_FINISH_REASON or "").strip().lower() == "length"

    output_token_cap = response_token_limit(task, goal)
    expert_prompt = build_reference_prompt(task, input_data)
    llm_calls_attempted += 1
    reference_answer = call_llm(expert_prompt, max_tokens=min(220, output_token_cap + 80))
    if not reference_answer:
        record_fallback("reference_generation_failed")
        reference_answer = fallback_output(task, prompt, input_data, is_reference=True)

    # Adjust token budget based on goal
    token_budget = 80
    if goal == "High Quality":
        token_budget = 100
    elif goal == "Low Cost":
        token_budget = 60

    # Initialize current state variables
    current_prompt = prompt
    llm_calls_attempted += 1
    current_output = call_llm(
        build_inference_prompt(task, current_prompt, input_data),
        max_tokens=output_token_cap,
    )
    if not current_output:
        record_fallback("baseline_generation_failed")
        current_output = fallback_output(task, current_prompt, input_data)
    elif last_response_truncated() and llm_calls_attempted < MAX_LLM_CALLS_PER_RUN:
        # Retry once with a larger cap to avoid displaying visibly cut-off baseline text.
        llm_calls_attempted += 1
        retry_cap = min(260, max(output_token_cap + 40, int(output_token_cap * 1.5)))
        retry_output = call_llm(
            build_inference_prompt(task, current_prompt, input_data),
            max_tokens=retry_cap,
        )
        if retry_output and not last_response_truncated():
            current_output = retry_output
    current_score = score_output(current_output, reference_answer, task=task)
    current_tokens = count_tokens(current_prompt)

    # Store initial values
    initial_prompt = current_prompt
    initial_output = current_output
    initial_score = current_score
    initial_tokens = current_tokens
    initial_output_tokens = count_tokens(initial_output)

    # Do not force lower caps than baseline; that can create visibly truncated outputs.
    optimized_output_cap = output_token_cap

    # Optimization loop variables
    best_prompt = current_prompt
    best_output = current_output
    best_score = current_score
    best_tokens = current_tokens
    best_output_tokens = initial_output_tokens
    best_total_tokens = current_tokens + initial_output_tokens
    total_reward = 0.0

    # Track candidates that improve both objectives simultaneously.
    dual_best_prompt = ""
    dual_best_output = ""
    dual_best_score = -1.0
    dual_best_tokens = 0
    dual_best_output_tokens = 0
    dual_goal_met = False

    # Track cost-safe candidates that do not regress quality.
    nonreg_best_prompt = initial_prompt
    nonreg_best_output = initial_output
    nonreg_best_score = initial_score
    nonreg_best_tokens = initial_tokens
    nonreg_best_output_tokens = initial_output_tokens

    steps_taken = 0
    termination_reason = "max_steps"

    for step in range(effective_max_steps):
        steps_taken += 1
        tokens_remaining = token_budget - current_tokens
        action_id = intelligent_action_selection(
            task,
            goal,
            current_prompt,
            current_score,
            tokens_remaining,
            step,
        )

        if action_id == 5:  # STOP
            stop_bonus = clip_reward(current_score * 1.5, lower=0.0, upper=2.0)
            total_reward += stop_bonus
            termination_reason = "voluntary_stop"
            break

        # Apply action (LLM-powered rewrite path when enabled)
        if use_intelligent_actions:
            if llm_calls_attempted >= MAX_LLM_CALLS_PER_RUN:
                termination_reason = "latency_guard"
                break
            llm_calls_attempted += 1
        new_prompt = apply_action_for_web(
            action_id, task, current_prompt, input_data, goal, use_intelligent_actions=use_intelligent_actions
        )

        if new_prompt == current_prompt:
            total_reward -= 0.1
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > token_budget:
            total_reward -= 0.5
            termination_reason = "budget_exceeded"
            break

        if llm_calls_attempted >= MAX_LLM_CALLS_PER_RUN:
            termination_reason = "latency_guard"
            break

        llm_calls_attempted += 1
        new_output = call_llm(
            build_inference_prompt(task, new_prompt, input_data),
            max_tokens=optimized_output_cap,
        )
        if not new_output:
            record_fallback(f"step_{step + 1}_generation_failed")
            new_output = fallback_output(task, new_prompt, input_data)
        elif last_response_truncated():
            # Never treat cut-off generations as optimized results.
            total_reward -= 0.2
            continue
        new_score = score_output(new_output, reference_answer, task=task)
        new_output_tokens = count_tokens(new_output)

        if new_score > initial_score and new_output_tokens < initial_output_tokens:
            if (
                dual_best_score < 0
                or new_score > dual_best_score
                or (new_score >= dual_best_score - 0.01 and new_output_tokens < dual_best_output_tokens)
            ):
                dual_best_prompt = new_prompt
                dual_best_output = new_output
                dual_best_score = new_score
                dual_best_tokens = new_tokens
                dual_best_output_tokens = new_output_tokens
                if OPTIMIZER_FAST_MODE and goal in ("Balanced", "Low Cost"):
                    dual_goal_met = True

        if new_score >= initial_score and new_output_tokens <= initial_output_tokens:
            if (
                new_output_tokens < nonreg_best_output_tokens
                or (new_output_tokens == nonreg_best_output_tokens and new_score > nonreg_best_score)
            ):
                nonreg_best_prompt = new_prompt
                nonreg_best_output = new_output
                nonreg_best_score = new_score
                nonreg_best_tokens = new_tokens
                nonreg_best_output_tokens = new_output_tokens

        quality_delta = new_score - current_score
        token_overhead = new_tokens - current_tokens
        # Penalize output verbosity heavily to keep cost trending down.
        output_growth = max(0, new_output_tokens - initial_output_tokens)
        step_reward = clip_reward(quality_delta - ALPHA * token_overhead - 0.04 * output_growth)
        total_reward += step_reward

        # Hard guard for demo consistency: never accept candidates that increase
        # output-token cost beyond baseline.
        if new_output_tokens > initial_output_tokens:
            current_prompt = new_prompt
            current_output = new_output
            current_score = new_score
            current_tokens = new_tokens
            continue

        # Acceptance logic with strong cost preference:
        # prioritize lower output tokens, then quality.
        new_total_tokens = new_tokens + new_output_tokens
        if (
            new_output_tokens < best_output_tokens
            or (new_output_tokens == best_output_tokens and new_score > best_score)
            or (
                new_score > best_score + 0.03
                and new_output_tokens <= best_output_tokens
            )
            or (
                new_score >= best_score - 0.02
                and new_total_tokens < best_total_tokens
            )
        ):
            best_prompt = new_prompt
            best_output = new_output
            best_score = new_score
            best_tokens = new_tokens
            best_output_tokens = new_output_tokens
            best_total_tokens = new_total_tokens

        current_prompt = new_prompt
        current_output = new_output
        current_score = new_score
        current_tokens = new_tokens

        if OPTIMIZER_FAST_MODE and dual_goal_met:
            termination_reason = "dual_goal_met"
            break

        if current_score > DONE_THRESHOLD:
            total_reward += 1.0
            termination_reason = "success"
            break

    # Rescue sweep: if dual-improvement not found, run a concise-focused search.
    if (
        not (dual_best_score > initial_score and dual_best_output_tokens < initial_output_tokens)
        and llm_calls_attempted < MAX_LLM_CALLS_PER_RUN
    ):
        if initial_output_tokens > 1:
            rescue_token_cap = max(8, min(optimized_output_cap, initial_output_tokens - 1))
            seen_prompts = {initial_prompt, best_prompt, current_prompt}
            rescue_word_target = max(12, int(initial_output_tokens * 0.75))
            rescue_trials = 0

            # Explicit dual-objective prompts to improve chance of quality-up + cost-down.
            crafted_candidates = [
                f"{best_prompt}\nRequirement: Improve quality and factual clarity while staying under {rescue_word_target} words. Use only essential details.",
                f"{initial_prompt}\nRequirement: Produce a better answer than baseline under {rescue_word_target} words. Keep it concise and accurate.",
            ]
            if OPTIMIZER_FAST_MODE:
                crafted_candidates = crafted_candidates[:1]

            for crafted_prompt in crafted_candidates:
                if llm_calls_attempted >= MAX_LLM_CALLS_PER_RUN or rescue_trials >= MAX_RESCUE_TRIALS:
                    break
                if crafted_prompt in seen_prompts:
                    continue
                seen_prompts.add(crafted_prompt)
                rescue_trials += 1

                llm_calls_attempted += 1
                crafted_output = call_llm(
                    build_inference_prompt(task, crafted_prompt, input_data),
                    max_tokens=rescue_token_cap,
                )
                if not crafted_output:
                    record_fallback("rescue_crafted_prompt_failed")
                    crafted_output = fallback_output(task, crafted_prompt, input_data)
                elif last_response_truncated():
                    continue

                crafted_score = score_output(crafted_output, reference_answer, task=task)
                crafted_tokens = count_tokens(crafted_prompt)
                crafted_output_tokens = count_tokens(crafted_output)

                if crafted_score > initial_score and crafted_output_tokens < initial_output_tokens:
                    if (
                        dual_best_score < 0
                        or crafted_score > dual_best_score
                        or (
                            crafted_score >= dual_best_score - 0.01
                            and crafted_output_tokens < dual_best_output_tokens
                        )
                    ):
                        dual_best_prompt = crafted_prompt
                        dual_best_output = crafted_output
                        dual_best_score = crafted_score
                        dual_best_tokens = crafted_tokens
                        dual_best_output_tokens = crafted_output_tokens

                if crafted_score >= initial_score and crafted_output_tokens <= initial_output_tokens:
                    if (
                        crafted_output_tokens < nonreg_best_output_tokens
                        or (
                            crafted_output_tokens == nonreg_best_output_tokens
                            and crafted_score > nonreg_best_score
                        )
                    ):
                        nonreg_best_prompt = crafted_prompt
                        nonreg_best_output = crafted_output
                        nonreg_best_score = crafted_score
                        nonreg_best_tokens = crafted_tokens
                        nonreg_best_output_tokens = crafted_output_tokens

            rescue_base_prompts = (best_prompt, current_prompt, initial_prompt)
            rescue_actions = (4, 1, 3, 2)
            if OPTIMIZER_FAST_MODE:
                rescue_base_prompts = (best_prompt, initial_prompt)
                rescue_actions = (4, 1)

            for base_prompt in rescue_base_prompts:
                for rescue_action in rescue_actions:
                    if llm_calls_attempted >= MAX_LLM_CALLS_PER_RUN or rescue_trials >= MAX_RESCUE_TRIALS:
                        break
                    candidate_prompt = apply_action_for_web(
                        rescue_action, task, base_prompt, input_data, "Low Cost"
                    )
                    if not candidate_prompt or candidate_prompt in seen_prompts:
                        continue
                    seen_prompts.add(candidate_prompt)
                    rescue_trials += 1

                    llm_calls_attempted += 1
                    candidate_output = call_llm(
                        build_inference_prompt(task, candidate_prompt, input_data),
                        max_tokens=rescue_token_cap,
                    )
                    if not candidate_output:
                        record_fallback(f"rescue_action_{rescue_action}_failed")
                        candidate_output = fallback_output(task, candidate_prompt, input_data)
                    elif last_response_truncated():
                        continue

                    candidate_score = score_output(candidate_output, reference_answer, task=task)
                    candidate_tokens = count_tokens(candidate_prompt)
                    candidate_output_tokens = count_tokens(candidate_output)

                    if candidate_score > initial_score and candidate_output_tokens < initial_output_tokens:
                        if (
                            dual_best_score < 0
                            or candidate_score > dual_best_score
                            or (
                                candidate_score >= dual_best_score - 0.01
                                and candidate_output_tokens < dual_best_output_tokens
                            )
                        ):
                            dual_best_prompt = candidate_prompt
                            dual_best_output = candidate_output
                            dual_best_score = candidate_score
                            dual_best_tokens = candidate_tokens
                            dual_best_output_tokens = candidate_output_tokens

                    if candidate_score >= initial_score and candidate_output_tokens <= initial_output_tokens:
                        if (
                            candidate_output_tokens < nonreg_best_output_tokens
                            or (
                                candidate_output_tokens == nonreg_best_output_tokens
                                and candidate_score > nonreg_best_score
                            )
                        ):
                            nonreg_best_prompt = candidate_prompt
                            nonreg_best_output = candidate_output
                            nonreg_best_score = candidate_score
                            nonreg_best_tokens = candidate_tokens
                            nonreg_best_output_tokens = candidate_output_tokens

                if llm_calls_attempted >= MAX_LLM_CALLS_PER_RUN or rescue_trials >= MAX_RESCUE_TRIALS:
                    break

    final_output_tokens = count_tokens(best_output)

    # Safety net: never report a final output that is costlier than baseline.
    if final_output_tokens > initial_output_tokens:
        best_prompt = initial_prompt
        best_output = initial_output
        best_score = initial_score
        best_tokens = initial_tokens
        final_output_tokens = initial_output_tokens

    # Selection priority:
    # 1) true dual-improvement (quality up + cost down)
    # 2) non-regression cost-safe candidate (quality >= baseline, cost <= baseline)
    # 3) baseline fallback
    if dual_best_score > initial_score and dual_best_output_tokens < initial_output_tokens:
        best_prompt = dual_best_prompt
        best_output = dual_best_output
        best_score = dual_best_score
        best_tokens = dual_best_tokens
        final_output_tokens = dual_best_output_tokens
    elif nonreg_best_score >= initial_score and nonreg_best_output_tokens <= initial_output_tokens:
        best_prompt = nonreg_best_prompt
        best_output = nonreg_best_output
        best_score = nonreg_best_score
        best_tokens = nonreg_best_tokens
        final_output_tokens = nonreg_best_output_tokens
    else:
        best_prompt = initial_prompt
        best_output = initial_output
        best_score = initial_score
        best_tokens = initial_tokens
        final_output_tokens = initial_output_tokens

    # If outputs are effectively identical, prevent false "improvement/reduction" reporting.
    if _normalized_text_signature(best_output) == _normalized_text_signature(initial_output):
        best_prompt = initial_prompt
        best_output = initial_output
        best_score = initial_score
        best_tokens = initial_tokens
        final_output_tokens = initial_output_tokens

    # Calculate final metrics after applying all selection guards.
    improvement_pct = ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
    token_change = best_tokens - initial_tokens
    token_reduction_pct = (abs(token_change) / initial_tokens * 100) if initial_tokens > 0 and token_change < 0 else 0

    initial_total_tokens = initial_tokens + initial_output_tokens
    final_total_tokens = best_tokens + final_output_tokens

    # Displayed reduction metrics are output-focused for user-facing cost tracking.
    token_reduction_abs = initial_output_tokens - final_output_tokens
    token_reduction_pct_total = (
        (token_reduction_abs / initial_output_tokens) * 100.0 if initial_output_tokens > 0 else 0.0
    )

    initial_cost = initial_output_tokens * OUTPUT_TOKEN_COST
    final_cost = final_output_tokens * OUTPUT_TOKEN_COST
    cost_reduction_abs = initial_cost - final_cost
    cost_reduction_pct = (cost_reduction_abs / initial_cost * 100.0) if initial_cost > 0 else 0.0

    quality_improvement_abs = best_score - initial_score
    quality_improvement_pct = (
        (quality_improvement_abs / initial_score) * 100.0 if initial_score > 0 else quality_improvement_abs * 100.0
    )
    runtime_status = get_llm_runtime_status()
    fallback_reason = " | ".join(fallback_reasons)
    offline_mode = bool(used_fallback and llm_calls_attempted > 0 and fallback_calls == llm_calls_attempted)
    fallback_notice = _build_fallback_notice(
        used_fallback=used_fallback,
        offline_mode=offline_mode,
        fallback_calls=fallback_calls,
        llm_calls_attempted=llm_calls_attempted,
        fallback_reason=fallback_reason or _LAST_LLM_ERROR,
    )

    return {
        "task": task,
        "steps_taken": steps_taken,
        "initial_prompt": initial_prompt,
        "initial_output": initial_output,
        "initial_reward": initial_score,
        "initial_score": initial_score,
        "initial_tokens": initial_tokens,
        "final_prompt": best_prompt,
        "final_output": best_output,
        "final_reward": best_score,
        "final_score": best_score,
        "final_tokens": best_tokens,
        "improvement_pct": improvement_pct,
        "token_change": token_change,
        "token_reduction_pct": token_reduction_pct,
        "initial_output_tokens": initial_output_tokens,
        "final_output_tokens": final_output_tokens,
        "initial_total_tokens": initial_total_tokens,
        "final_total_tokens": final_total_tokens,
        "token_reduction_abs": token_reduction_abs,
        "token_reduction_pct_total": token_reduction_pct_total,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_reduction_abs": cost_reduction_abs,
        "cost_reduction_pct": cost_reduction_pct,
        "quality_improvement_abs": quality_improvement_abs,
        "quality_improvement_pct": quality_improvement_pct,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "used_fallback": used_fallback,
        "offline_mode": offline_mode,
        "fallback_calls": fallback_calls,
        "llm_calls_attempted": llm_calls_attempted,
        "fallback_reason": fallback_reason or _LAST_LLM_ERROR,
        "fallback_notice": fallback_notice,
        "llm_runtime": runtime_status,
        "intelligent_actions_enabled": use_intelligent_actions,
    }


# Keep for backward compatibility
def optimize_prompt(initial_prompt: str, context: str = "", reference: str = "",
                   example: str = "", constraint: str = "", max_steps: int = 7) -> dict:
    """Optimize a prompt using RL-based action selection."""
    state = {
        "task": "General",
        "prompt": initial_prompt,
        "input": context,
        "goal": "Balanced"
    }
    return optimize_prompt_with_state(state)


# ── Ensure templates directory and HTML files exist ──────────────────────────
def _write_templates():
    """Write HTML templates to disk (always, so they stay in sync)."""
    tpl_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    _write_landing_html(tpl_dir)
    _write_index_html(tpl_dir)
    _write_results_html(tpl_dir)


# Create FastAPI app
TPL_DIR = os.path.join(BASE_DIR, "templates")
app = FastAPI(title="PromptOptEnv Web")
templates = Jinja2Templates(directory=TPL_DIR)


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
async def landing(request: Request):
    """Render the landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
@app.get("/web/app", response_class=HTMLResponse)
@app.get("/web/app/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main optimizer form page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Default prompts for each task
DEFAULT_PROMPTS = {
    "Summarization": "Summarize the following text concisely.",
    "Question Answering": "Answer the question based on the given context.",
    "Paraphrasing": "Paraphrase the following sentence.",
    "Instruction Following": "Follow the instruction and generate an appropriate response."
}


@app.post("/optimize", response_class=HTMLResponse)
@app.post("/web/optimize", response_class=HTMLResponse)
async def optimize(
    request: Request,
    task: str = Form("Summarization"),
    input_text: str = Form(None, alias="input-text"),
    context: str = Form(None),
    question: str = Form(None),
    user_prompt: str = Form(None, alias="initial-prompt"),
    goal: str = Form("Balanced", alias="optimization-goal")
):
    """Handle optimization form submission with structured task input."""
    selected_task = task if task in DEFAULT_PROMPTS else "Summarization"
    provided_prompt = (user_prompt or "").strip()

    state = {
        "task": selected_task,
        "input": "",
        "prompt": provided_prompt if provided_prompt else DEFAULT_PROMPTS[selected_task],
        "goal": goal
    }

    if selected_task == "Question Answering":
        if not context and not question:
            # Fallback if UI misbehaves
            state["input"] = input_text or ""
        else:
            state["input"] = f"Context: {context or ''}\nQuestion: {question or ''}"
    else:
        state["input"] = input_text or ""

    results = optimize_prompt_with_state(state)
    return templates.TemplateResponse("results.html", {
        "request": request,
        **results
    })


@app.websocket("/ws/ui")
async def hf_ui_websocket(websocket: WebSocket):
    """
    HF App shell may probe /ws/ui. Accept a no-op websocket so the app
    doesn't emit repeated 403 logs and the shell handshake stays healthy.
    """
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


def _write_landing_html(tpl_dir: str):
    landing_html = """<!DOCTYPE html>
<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>PromptOptEnv - RL Prompt Optimization</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&amp;family=Inter:wght@300;400;500;600&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<style>
        body {
            background-color: #000000;
            color: #e2e2e2;
            overflow-x: hidden;
        }
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
        }
        /* Kinetic Void Background Orbs */
        .void-orb {
            position: absolute;
            filter: blur(120px);
            z-index: -1;
            opacity: 0.15;
            border-radius: 50%;
        }
        .orb-primary { background: #ff5708; width: 600px; height: 600px; top: -200px; right: -100px; }
        .orb-secondary { background: #0055ff; width: 500px; height: 500px; bottom: -100px; left: -100px; }
        
        .glass-card {
            background: rgba(27, 27, 27, 0.4);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .kinetic-glow-border {
            position: relative;
            background: #ff5708;
            transition: all 0.3s ease;
        }
        .kinetic-glow-border::before {
            content: '';
            position: absolute;
            inset: -2px;
            background: conic-gradient(from 0deg, #ff5708, #0055ff, #ff5708);
            border-radius: inherit;
            z-index: -1;
            animation: rotate-glow 8s linear infinite;
        }

        @keyframes rotate-glow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    
    .glow-pill {
      position: relative;
      border-radius: 999px;
      z-index: 1;
      overflow: hidden;
      padding: 2px;
    }
    .glow-pill::before {
      content: "";
      position: absolute;
      top: -200%; left: -50%; bottom: -200%; right: -50%;
      background: conic-gradient(transparent, transparent, transparent, #4d79ff, #ff4d4d);
      z-index: -1;
      animation: rotate-glow 4s linear infinite;
    }
    .glow-pill-inner {
      background: #050505;
      border-radius: 999px;
      position: relative;
      z-index: 2;
      height: 100%;
      cursor: pointer;
    }
</style>
<script id="tailwind-config">
        tailwind.config = {
          darkMode: "class",
          theme: {
            extend: {
              "colors": {
                      "tertiary-container": "#0494fc",
                      "on-secondary-container": "#e4e7ff",
                      "inverse-on-surface": "#303030",
                      "on-error": "#690005",
                      "inverse-surface": "#e2e2e2",
                      "surface-container-lowest": "#0e0e0e",
                      "on-primary": "#5c1900",
                      "on-surface-variant": "#e5beb2",
                      "secondary-fixed": "#dce1ff",
                      "tertiary-fixed": "#d2e4ff",
                      "secondary": "#b6c4ff",
                      "inverse-primary": "#aa3600",
                      "tertiary": "#a1c9ff",
                      "on-tertiary-fixed": "#001c37",
                      "surface-dim": "#131313",
                      "secondary-fixed-dim": "#b6c4ff",
                      "error-container": "#93000a",
                      "primary-fixed": "#ffdbcf",
                      "surface-container": "#1f1f1f",
                      "outline-variant": "#5c4037",
                      "on-tertiary": "#00325a",
                      "on-secondary": "#002780",
                      "on-primary-container": "#511500",
                      "on-tertiary-fixed-variant": "#004880",
                      "surface-container-low": "#1b1b1b",
                      "surface-container-high": "#2a2a2a",
                      "on-error-container": "#ffdad6",
                      "on-primary-fixed": "#390c00",
                      "tertiary-fixed-dim": "#a1c9ff",
                      "surface-bright": "#393939",
                      "background": "#131313",
                      "error": "#ffb4ab",
                      "primary-fixed-dim": "#ffb59c",
                      "surface-tint": "#ffb59c",
                      "primary-container": "#ff5708",
                      "on-surface": "#e2e2e2",
                      "on-secondary-fixed-variant": "#0039b3",
                      "on-secondary-fixed": "#001551",
                      "on-primary-fixed-variant": "#822700",
                      "surface-container-highest": "#353535",
                      "surface": "#131313",
                      "on-tertiary-container": "#002b4f",
                      "secondary-container": "#0356ff",
                      "on-background": "#e2e2e2",
                      "outline": "#ac897e",
                      "surface-variant": "#353535",
                      "primary": "#ffb59c"
              },
              "borderRadius": {
                      "DEFAULT": "0.125rem",
                      "lg": "0.25rem",
                      "xl": "0.5rem",
                      "full": "0.75rem"
              },
              "fontFamily": {
                      "headline": ["Space Grotesk"],
                      "body": ["Inter"],
                      "label": ["Inter"]
              }
            },
          },
        }
      </script>
<style>

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: rgba(255, 255, 255, 0.25);
    }
    * {
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
    }

</style>
</head>
<body class="font-body selection:bg-primary-container selection:text-white">
<!-- Ambient Visuals -->
<div class="void-orb orb-primary"></div>
<div class="void-orb orb-secondary"></div>
<!-- TopNavBar -->
<header class="w-full top-0 sticky bg-transparent backdrop-blur-xl z-50">
<nav class="flex justify-between items-center w-full px-8 py-6 max-w-7xl mx-auto">
<div class="text-2xl font-bold tracking-tighter text-zinc-100 font-headline" style="">PromptOptEnv</div>
<div class="hidden md:flex items-center space-x-12">
<button onclick="window.location.href='/app'" class="bg-white/10 hover:bg-white/20 border border-white/20 text-white px-6 py-2 rounded-lg font-bold transition-all active:scale-95">
                    Try Now
                </button>
</div>
<!-- Mobile Toggle -->
<button class="md:hidden text-on-surface" style="">
<span class="material-symbols-outlined" data-icon="menu" style="">menu</span>
</button>
</nav>
</header>
<main class="relative z-10">
<!-- Hero Section -->
<section class="min-h-[70vh] flex flex-col items-center justify-center text-center px-4 pt-20">
<div class="max-w-4xl mx-auto">
<h1 class="text-6xl md:text-8xl font-headline font-bold text-white tracking-tighter mb-6 leading-none" style="">
                    PromptOptEnv
                </h1>
<p class="text-xl md:text-2xl font-light text-on-surface-variant max-w-2xl mx-auto mb-12 font-body tracking-tight" style="">
                    RL-Powered Prompt Optimization with <span class="text-primary" style="">Cost Awareness</span>.
                </p>
<div class="flex justify-center mt-4">
<div class="glow-pill w-fit transition-transform hover:scale-105 active:scale-95 shadow-2xl" onclick="window.location.href='/app'">
    <div class="glow-pill-inner px-12 py-4 flex items-center justify-center">
        <span class="text-white font-bold text-lg">Try Now</span>
    </div>
</div>
</div>
<div class="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl mx-auto px-4">
<div class="h-px bg-gradient-to-r from-transparent via-outline-variant to-transparent opacity-30 md:hidden"></div>
</div>
</div>
</section>
<!-- About the Project Section -->
<section class="pb-16 px-8 max-w-7xl mx-auto">
<div class="glass-card p-8 md:p-16 rounded-3xl border border-white/5 relative overflow-hidden group">
<div class="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:bg-primary/10 transition-colors"></div>
<div class="relative z-10 flex flex-col md:flex-row gap-12 items-center">
<div class="md:w-1/3">
<h2 class="text-4xl md:text-5xl font-headline font-bold text-white tracking-tighter leading-tight" style="">
                    About the Project
                </h2>
<div class="mt-4 w-24 h-1 bg-primary rounded-full"></div>
</div>
<div class="md:w-2/3">
<p class="text-xl md:text-2xl font-body font-light text-on-surface-variant leading-relaxed" style="">
                    PromptOptEnv is a high-fidelity <span class="text-white font-medium">RL-Powered Prompt Optimization</span> tool designed for the next generation of AI development. 
                    By integrating deep Reinforcement Learning with <span class="text-tertiary font-medium">Cost Awareness</span>, we've engineered a platform that doesn't just improve outputs, but does so with surgical precision. 
                    Built specifically for high-performance <span class="text-secondary font-medium">production workloads</span>, it delivers consistent efficiency when every token counts.
                </p>
<div class="mt-8 flex flex-wrap gap-4">
<span class="px-4 py-2 rounded-full border border-outline-variant/30 bg-white/5 text-xs font-label uppercase tracking-widest text-zinc-400">Reinforcement Learning</span>
<span class="px-4 py-2 rounded-full border border-outline-variant/30 bg-white/5 text-xs font-label uppercase tracking-widest text-zinc-400">Token Efficiency</span>
<span class="px-4 py-2 rounded-full border border-outline-variant/30 bg-white/5 text-xs font-label uppercase tracking-widest text-zinc-400">Production Grade</span>
</div>
</div>
</div>
</div>
</section>
<!-- Features Section (Bento Inspired) -->
<section class="pb-32 px-8 max-w-7xl mx-auto">
<div class="grid grid-cols-1 md:grid-cols-12 gap-6">
<!-- Feature 1: RL-Powered -->
<div class="md:col-span-8 glass-card p-12 rounded-xl group hover:border-primary/30 transition-all duration-500">
<div class="flex flex-col h-full justify-between">
<div>
<div class="mb-8 w-12 h-12 rounded-full bg-primary-container/10 flex items-center justify-center border border-primary/20">
<span class="material-symbols-outlined text-primary" data-icon="auto_awesome" style="">auto_awesome</span>
</div>
<h3 class="text-4xl font-headline font-bold text-white mb-6" style="">RL-Powered Precision</h3>
<p class="text-on-surface-variant text-lg leading-relaxed max-w-xl" style="">
                                Our Reinforcement Learning engine autonomously explores the prompt latent space, adapting optimization strategies for maximum context alignment and output fidelity.
                            </p>
</div>
<div class="mt-12 h-48 w-full overflow-hidden rounded-lg bg-black/40 border border-white/5">
<img class="w-full h-full object-cover opacity-50 grayscale hover:grayscale-0 transition-all duration-700" data-alt="abstract neural network visualization with glowing orange nodes and connections on a dark technical background" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDHfagrpd5Gwoee-oPV1fsj5bz4-dbzS36Gzqd87-aY4n-MmeOLDU_cTS2R0VS_QyO5iQEGiKjoRqbGA0eArKYRrPdKOa84iwiKkMbhBl9pW9byuflJm-jLk96GfWhm46803LAsO3o5s2nevrRnPYQl802wEnJpzk46GIlJ2fzBs1q6tgoihmFvMGO1fKWJ-kclxPpMyYdbCk2kuaxyCvf-vFIr3pl_hdCYWqX9tApZ15Llu600Lqu-6ppOns0hqSp5yZDqVDxHyMda" style=""/>
</div>
</div>
</div>
<!-- Feature 2: Cost Awareness -->
<div class="md:col-span-4 glass-card p-12 rounded-xl border-l-2 border-l-tertiary-container group hover:bg-surface-container-low transition-all">
<div class="flex flex-col h-full">
<div class="mb-8 w-12 h-12 rounded-full bg-tertiary-container/10 flex items-center justify-center border border-tertiary/20">
<span class="material-symbols-outlined text-tertiary" data-icon="account_balance_wallet" style="">account_balance_wallet</span>
</div>
<h3 class="text-3xl font-headline font-bold text-white mb-6" style="">Cost Awareness</h3>
<p class="text-on-surface-variant leading-relaxed" style="">
                            Intelligent budgeting algorithms to reduce token consumption without sacrificing performance quality.
                        </p>
<div class="mt-auto pt-12">
<div class="text-6xl font-headline font-bold text-tertiary/20" style="">90%</div>
<div class="text-sm font-label uppercase tracking-widest text-zinc-500" style="">Efficiency Gain</div>
</div>
</div>
</div>
<!-- Feature 3: Dynamic Task Adapters -->
<div class="md:col-span-4 glass-card p-12 rounded-xl group hover:bg-surface-container-low transition-all">
<div class="flex flex-col h-full">
<div class="mb-8 w-12 h-12 rounded-full bg-secondary-container/10 flex items-center justify-center border border-secondary/20">
<span class="material-symbols-outlined text-secondary" data-icon="extension" style="">extension</span>
</div>
<h3 class="text-3xl font-headline font-bold text-white mb-6" style="">Dynamic Task Adapters</h3>
<p class="text-on-surface-variant leading-relaxed" style="">
                            Modular grading architectures designed universally for Summarization, QA, and advanced Instruction Following tasks.
                        </p>
</div>
</div>
<!-- Feature 4: Visual Edge -->
<div class="md:col-span-8 glass-card rounded-xl overflow-hidden relative group">
<img class="w-full h-full object-cover opacity-40 group-hover:scale-105 transition-transform duration-1000" data-alt="high-tech dashboard interface with holographic data visualizations and code structures in deep space blue and orange" src="https://lh3.googleusercontent.com/aida-public/AB6AXuC9L4wg9NZ7NI4V7KRKT44KbOIJrTh574LoiFRPxkBWVTWtuerdEuUCadVPAUm5p4Zi1l14exATHaau0xBdty19zmS_DHLyvA320oiDFpN0A7hbc-0Intk6DBQ265HETMd9MJcYNx3tk0NRJgJlGjTNmHLvbQOi9xdVB-pYfPpQMmm7_QCvCllmtLoeZgRSrsW33ac8Re6iNk3TbD1Zg6o_-nnubHMP5njMQR0qoBRwXHZV1SRekKfJtDoX3HEGUPBN15EQszgTWbZ1" style=""/>
<div class="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent p-12 flex flex-col justify-end">
<h3 class="text-4xl font-headline font-bold text-white mb-2" style="">Infinite Scaling</h3>
<p class="text-on-surface-variant max-w-md" style="">Deploy RL-optimized environments across distributed clusters with one-click orchestration.</p>
</div>
</div>
</div>
</section>
<!-- CTA Section -->
<section class="py-12 text-center px-4">
<div class="max-w-3xl mx-auto py-24 glass-card rounded-3xl relative overflow-hidden border border-white/5">
<div class="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-primary to-transparent opacity-50"></div>
<h2 class="text-4xl md:text-5xl font-headline font-bold text-white mb-8" style="">Ready to Optimize?</h2>
<p class="text-on-surface-variant mb-12 text-lg" style="">Join the frontier of automated prompt engineering.</p>

<div class="flex justify-center mt-4">
<div class="glow-pill w-fit transition-transform hover:scale-105 active:scale-95 shadow-2xl" onclick="window.location.href='/app'">
    <div class="glow-pill-inner px-12 py-4 flex items-center justify-center">
        <span class="text-white font-bold text-lg">Get Started</span>
    </div>
</div>
</div>
</div>
</section>
</main>
<!-- Footer -->
<footer class="w-full border-t border-zinc-800/30 tonal-shift bg-zinc-950">
<div class="flex flex-col md:flex-row justify-between items-center w-full px-8 py-12 max-w-7xl mx-auto">
<div class="mb-8 md:mb-0">
<div class="text-lg font-bold text-zinc-100 mb-2 font-headline" style="">PromptOptEnv</div>
<p class="font-['Inter'] text-sm tracking-tight text-zinc-500" style="">© 2026 PromptOptEnv. Precise RL Optimization.</p>
</div>
<div class="flex items-center space-x-8">
<a class="text-zinc-500 hover:text-orange-500 transition-all text-sm font-['Inter']" href="https://github.com/Shyamyemuka" target="_blank" rel="noopener noreferrer" style="">GitHub</a>
<a class="text-zinc-500 hover:text-orange-500 transition-all text-sm font-['Inter']" href="https://www.linkedin.com/in/shyam-yemuka-0aba06205/" target="_blank" rel="noopener noreferrer" style="">LinkedIn</a>
</div>
<div class="mt-8 md:mt-0 flex items-center space-x-4 opacity-80 hover:opacity-100">
<div class="w-2 h-2 rounded-full bg-green-500"></div>
<span class="text-xs font-label uppercase tracking-widest text-zinc-400" style="">All Systems Operational</span>
</div>
</div>
</footer>
</body></html>"""
    with open(os.path.join(tpl_dir, "landing.html"), "w", encoding="utf-8") as f:
        f.write(landing_html)

def _write_index_html(tpl_dir: str):
    index_html = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>PromptOptEnv - Prompt Optimization</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<style data-purpose="custom-fonts">
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
</style>
<style data-purpose="custom-effects">
    .glow-border {
      position: relative;
      border-radius: 14px;
      z-index: 1;
      overflow: hidden;
      padding: 3px;
    }
    .glow-border::before {
      content: "";
      position: absolute;
      top: -50%; left: -50%; bottom: -50%; right: -50%;
      background: conic-gradient(transparent, transparent, transparent, #4d79ff, #ff4d4d);
      z-index: -1;
      animation: spin 10s linear infinite;
    }
    .glow-border-inner {
      background: #050505;
      border-radius: 12px;
      position: relative;
      z-index: 2;
      border: 1px solid rgba(255, 255, 255, 0.1);
      height: 100%;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    .hidden { display: none !important; }
    .loading-overlay {
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(0, 0, 0, 0.72);
      backdrop-filter: blur(6px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }
    .loading-card {
      width: 100%;
      max-width: 26rem;
      border-radius: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: rgba(5, 5, 5, 0.95);
      padding: 1.25rem;
      box-shadow: 0 25px 50px rgba(0, 0, 0, 0.45);
    }
    .loading-ring {
      width: 2.75rem;
      height: 2.75rem;
      border-radius: 9999px;
      border: 2px solid rgba(255, 255, 255, 0.16);
      border-top-color: #60a5fa;
      border-right-color: #f87171;
      animation: spin 0.9s linear infinite;
      flex-shrink: 0;
    }
    .loading-track {
      margin-top: 0.9rem;
      height: 0.35rem;
      width: 100%;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
      position: relative;
    }
    .loading-track::before {
      content: "";
      position: absolute;
      inset: 0;
      width: 42%;
      border-radius: 999px;
      background: linear-gradient(90deg, #60a5fa, #818cf8, #f87171);
      animation: slide 1.2s ease-in-out infinite;
    }
    @keyframes slide {
      0% { transform: translateX(-110%); }
      100% { transform: translateX(260%); }
    }
</style>
<style>

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: rgba(255, 255, 255, 0.25);
    }
    * {
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
    }

</style>
</head>
<body class="bg-black text-gray-300 min-h-screen flex flex-col items-center justify-center p-4">
<!-- Back to Home -->
<a href="/" class="absolute top-6 left-6 flex items-center gap-2 text-gray-400 hover:text-white transition-colors group z-50">
    <svg class="h-5 w-5 transform group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M10 19l-7-7m0 0l7-7m-7 7h18" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path></svg>
    <span class="text-sm font-medium">Back to Home</span>
</a>

<!-- BEGIN: Header Section -->
<header class="text-center mb-6">
<h1 class="text-white text-4xl font-bold mb-2 tracking-tight"><a href="/">PromptOptEnv</a></h1>
<p class="text-gray-400 text-sm">RL-Powered Prompt Optimization with Cost Awareness</p>
</header>
<!-- END: Header Section -->
<!-- BEGIN: Main Form Container -->
<main class="w-full max-w-2xl glow-border">
<div class="glow-border-inner p-8 shadow-2xl">
<form action="/optimize" class="space-y-6" method="POST" id="optimizer-form">
<!-- Select Task -->
<div data-purpose="form-group">
<label class="block text-sm font-medium text-gray-300 mb-2" for="task">Select Task</label>
<div class="relative">
<select class="block w-full bg-black border border-gray-700 rounded-md py-3 pl-4 pr-10 text-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm appearance-none" id="task" name="task">
<option value="Summarization">Summarization</option>
<option value="Question Answering">Question Answering</option>
<option value="Paraphrasing">Paraphrasing</option>
<option value="Instruction Following">Instruction Following</option>
</select>
</div>
</div>
<!-- Input Text -->
<div id="generic-input-group" data-purpose="form-group">
<label id="generic-input-label" class="block text-sm font-medium text-gray-300 mb-2" for="input-text">Input Text</label>
<textarea class="block w-full bg-black border border-gray-700 rounded-md py-3 px-4 text-gray-400 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm resize-y" id="input-text" name="input-text" placeholder="Paste the paragraph you want to optimize for submission..." rows="4"></textarea>
</div>

<!-- QA Inputs -->
<div id="qa-input-group" class="hidden space-y-6">
    <div data-purpose="form-group">
    <label class="block text-sm font-medium text-gray-300 mb-2" for="context">Context</label>
    <textarea class="block w-full bg-black border border-gray-700 rounded-md py-3 px-4 text-gray-400 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm resize-y" id="context" name="context" placeholder="Enter context passage..." rows="3"></textarea>
    </div>
    
    <div data-purpose="form-group">
    <label class="block text-sm font-medium text-gray-300 mb-2" for="question">Question</label>
    <textarea class="block w-full bg-black border border-gray-700 rounded-md py-3 px-4 text-gray-400 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm resize-y" id="question" name="question" placeholder="Enter your question..." rows="2"></textarea>
    </div>
</div>

<!-- Divider / Optional Config Header -->
<div class="pt-2">
<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">OPTIONAL CONFIGURATION</h3>
</div>
<!-- Initial Prompt -->
<div data-purpose="form-group">
<label class="block text-sm font-medium text-gray-300 mb-2" for="initial-prompt">Initial Prompt (Optional)</label>
<textarea class="block w-full bg-black border border-gray-700 rounded-md py-3 px-4 text-gray-400 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm resize-y" id="initial-prompt" name="initial-prompt" placeholder="e.g., Summarize this for a technical audience..." rows="3"></textarea>
</div>
<!-- Optimization Goal -->
<div data-purpose="form-group">
<label class="block text-sm font-medium text-gray-300 mb-2" for="optimization-goal">Optimization Goal</label>
<div class="relative">
<select class="block w-full bg-black border border-gray-700 rounded-md py-3 pl-4 pr-10 text-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-500 focus:border-gray-500 sm:text-sm appearance-none" id="optimization-goal" name="optimization-goal">
<option value="Balanced">Balanced</option>
<option value="High Quality">High Quality</option>
<option value="Low Cost">Low Cost</option>
</select>

</div>
</div>
<!-- Submit Button -->
<div class="pt-4">
<button id="optimize-submit-btn" class="w-full bg-black border border-gray-600 rounded-full py-3 px-4 text-sm font-medium text-white hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 focus:ring-offset-black transition-colors duration-200 disabled:opacity-60 disabled:cursor-not-allowed" type="submit">
<span id="optimize-submit-text">Optimize Prompt</span>
</button>
</div>
</form>
</div>
</main>
<!-- END: Main Form Container -->
<div id="loading-overlay" class="loading-overlay hidden" aria-live="polite" aria-busy="true" aria-label="Optimizing prompt">
    <div class="loading-card">
        <div class="flex items-center gap-4">
            <div class="loading-ring"></div>
            <div>
                <div class="text-white font-semibold tracking-tight">Optimizing Prompt</div>
                <div class="text-sm text-gray-400">Running prompt search and scoring responses...</div>
            </div>
        </div>
        <div class="loading-track"></div>
    </div>
</div>
<script>
    const optimizeForm = document.getElementById("optimizer-form");
    const submitBtn = document.getElementById("optimize-submit-btn");
    const submitText = document.getElementById("optimize-submit-text");
    const loadingOverlay = document.getElementById("loading-overlay");
    const taskField = document.getElementById("task");
    const genericInputGroup = document.getElementById("generic-input-group");
    const genericInputLabel = document.getElementById("generic-input-label");
    const genericInput = document.getElementById("input-text");
    const qaInputGroup = document.getElementById("qa-input-group");
    let submitLocked = false;

    const inputConfig = {
        "Summarization": { label: "Input Text", placeholder: "Paste the paragraph you want to summarize..." },
        "Paraphrasing": { label: "Sentence", placeholder: "Enter a sentence to paraphrase..." },
        "Instruction Following": { label: "Instruction", placeholder: "e.g., Explain photosynthesis in simple terms" }
    };

    function updateTaskInputs() {
        const selectedTask = taskField.value;
        if (selectedTask === "Question Answering") {
            genericInputGroup.classList.add("hidden");
            qaInputGroup.classList.remove("hidden");
            return;
        }
        qaInputGroup.classList.add("hidden");
        genericInputGroup.classList.remove("hidden");
        const config = inputConfig[selectedTask] || inputConfig["Summarization"];
        genericInputLabel.textContent = config.label;
        genericInput.placeholder = config.placeholder;
    }
    taskField.addEventListener("change", updateTaskInputs);
    updateTaskInputs();

    function setLoadingState(active) {
        if (active) {
            loadingOverlay.classList.remove('hidden');
            document.body.classList.add('overflow-hidden');
            submitBtn.disabled = true;
            submitText.textContent = 'Optimizing...';
            return;
        }
        loadingOverlay.classList.add('hidden');
        document.body.classList.remove('overflow-hidden');
        submitBtn.disabled = false;
        submitText.textContent = 'Optimize Prompt';
    }

    window.addEventListener('pageshow', function() {
        submitLocked = false;
        setLoadingState(false);
    });

    optimizeForm.addEventListener('submit', function(e) {
        if (submitLocked) {
            e.preventDefault();
            return;
        }

        const selected = taskField.value;
        let isValid = true;
        if (selected === "Question Answering") {
            const ctx = document.getElementById('context').value.trim();
            const q = document.getElementById('question').value.trim();
            if (!ctx && !q) {
                e.preventDefault();
                alert('Please provide Context or a Question before optimizing.');
                isValid = false;
            }
        } else {
            const txt = document.getElementById('input-text').value.trim();
            if (!txt) {
                e.preventDefault();
                alert('Please provide Input Text before optimizing.');
                isValid = false;
            }
        }

        if (!isValid) {
            setLoadingState(false);
            return;
        }

        submitLocked = true;
        setLoadingState(true);
    });
</script>
</body></html>"""
    with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)


def _write_results_html(tpl_dir: str):
    results_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Optimization Results - PromptOptEnv</title>
<script src="https://cdn.tailwindcss.com"></script>
<style data-purpose="custom-fonts">
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
</style>
<style data-purpose="custom-effects">
    .glow-border {
      position: relative;
      border-radius: 14px;
      z-index: 1;
      overflow: hidden;
      padding: 3px;
    }
    .glow-border::before {
      content: "";
      position: absolute;
      top: -50%; left: -50%; bottom: -50%; right: -50%;
      background: conic-gradient(transparent, transparent, transparent, #4d79ff, #ff4d4d);
      z-index: -1;
      animation: spin 10s linear infinite;
    }
    .glow-border-inner {
      background: #050505;
      border-radius: 12px;
      position: relative;
      z-index: 2;
      border: 1px solid rgba(255, 255, 255, 0.1);
      height: 100%;
    }
    .scrollable-block {
      min-height: 8rem;
      max-height: 20rem;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      scrollbar-gutter: stable;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
</style>
<style>

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: rgba(255, 255, 255, 0.25);
    }
    * {
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
    }

</style>
</head>
<body class="bg-black text-gray-300 min-h-screen flex flex-col items-center py-10 px-4">
<!-- Back to Home -->
<a href="/" class="absolute top-6 left-6 flex items-center gap-2 text-gray-400 hover:text-white transition-colors group z-50">
    <svg class="h-5 w-5 transform group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M10 19l-7-7m0 0l7-7m-7 7h18" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path></svg>
    <span class="text-sm font-medium">Back to Home</span>
</a>


<main class="w-full max-w-5xl glow-border">
<div class="glow-border-inner p-8 shadow-2xl">
    <div class="flex items-center justify-between xl:mb-8 mb-6">
        <h2 class="text-2xl font-bold text-white tracking-tight">Optimization Results: {{ task }}</h2>
        <a href="/app" class="px-4 py-2 rounded-full border border-gray-600 text-sm font-medium hover:bg-gray-800 transition-colors">&larr; Optimize Another</a>
    </div>

    <!-- Before / After split -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        <!-- BEFORE -->
        <div class="p-6 bg-[#0a0a0a] rounded-xl border border-gray-800">
            <h3 class="text-red-400 font-semibold mb-4 text-sm uppercase tracking-wider">Before (Baseline)</h3>
            
            <div class="mb-5">
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">Prompt</span>
                <div class="p-4 bg-black border border-gray-800 rounded-md text-sm text-gray-300 leading-relaxed italic scrollable-block">
                    {{ initial_prompt }}
                </div>
            </div>
            
            <div class="mb-5">
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">LLM Output</span>
                <div class="p-4 bg-black border border-gray-800 rounded-md text-sm text-gray-400 leading-relaxed scrollable-block">
                    {{ initial_output or '[No output generated]' }}
                </div>
            </div>
            
            <div>
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">Reward Score</span>
                <div class="text-xl text-white font-bold">{{ "%.2f"|format(initial_reward) }}</div>
            </div>
        </div>

        <!-- AFTER -->
        <div class="p-6 bg-[#0a0a0a] rounded-xl border border-gray-800 relative shadow-[0_0_25px_rgba(77,121,255,0.05)]">
            <h3 class="text-blue-400 font-semibold mb-4 text-sm uppercase tracking-wider">After (Optimized - {{ steps_taken }} steps)</h3>
            
            <div class="mb-5">
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">Optimized Prompt</span>
                <div class="p-4 bg-black border border-gray-700 rounded-md text-sm text-gray-200 leading-relaxed font-mono scrollable-block">
                    {{ final_prompt }}
                </div>
            </div>
            
            <div class="mb-5">
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">LLM Output</span>
                <div class="p-4 bg-black border border-gray-800 rounded-md text-sm text-gray-300 leading-relaxed scrollable-block">
                    {{ final_output or '[No output generated]' }}
                </div>
            </div>
            
            <div>
                <span class="block text-xs text-gray-500 font-medium mb-2 uppercase">Reward Score</span>
                <div class="text-xl text-white font-bold">{{ "%.2f"|format(final_reward) }}</div>
            </div>
        </div>
    </div>
    
    <div class="mt-8 pt-6 border-t border-gray-800">
        <h3 class="text-xs text-gray-500 font-semibold mb-4 uppercase tracking-wider">Episode Metrics</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="p-4 bg-[#0d0d0d] rounded-lg border border-gray-800">
                <div class="text-sm text-gray-400 mb-1">Quality Imp.</div>
                <div class="text-lg text-white">{{ "%+.4f"|format(quality_improvement_abs) }} ({{ "%+.1f"|format(quality_improvement_pct) }}%)</div>
                {% if quality_improvement_abs > 0 %}
                <div class="text-xs text-green-400 mt-1">Improved by {{ "%.4f"|format(quality_improvement_abs) }} ({{ "%.1f"|format(quality_improvement_pct) }}%)</div>
                {% elif quality_improvement_abs < 0 %}
                <div class="text-xs text-amber-300 mt-1">Reduced by {{ "%.4f"|format(-quality_improvement_abs) }} ({{ "%.1f"|format(-quality_improvement_pct) }}%)</div>
                {% else %}
                <div class="text-xs text-gray-400 mt-1">No change</div>
                {% endif %}
            </div>
            <div class="p-4 bg-[#0d0d0d] rounded-lg border border-gray-800">
                <div class="text-sm text-gray-400 mb-1">Token Reduction</div>
                <div class="text-lg text-white">{{ "%+.0f"|format(-token_reduction_abs) }} ({{ "%+.1f"|format(-token_reduction_pct_total) }}%)</div>
                {% if token_reduction_abs > 0 %}
                <div class="text-xs text-green-400 mt-1">Reduced by {{ "%.0f"|format(token_reduction_abs) }} ({{ "%.1f"|format(token_reduction_pct_total) }}%)</div>
                {% elif token_reduction_abs < 0 %}
                <div class="text-xs text-amber-300 mt-1">Increased by {{ "%.0f"|format(-token_reduction_abs) }} ({{ "%.1f"|format(-token_reduction_pct_total) }}%)</div>
                {% else %}
                <div class="text-xs text-gray-400 mt-1">No change</div>
                {% endif %}
            </div>
            <div class="p-4 bg-[#0d0d0d] rounded-lg border border-gray-800">
                <div class="text-sm text-gray-400 mb-1">Cost Reduction</div>
                <div class="text-lg text-white">{{ "%+.2f"|format(-cost_reduction_abs) }} ({{ "%+.1f"|format(-cost_reduction_pct) }}%)</div>
                {% if cost_reduction_abs > 0 %}
                <div class="text-xs text-green-400 mt-1">Reduced by {{ "%.2f"|format(cost_reduction_abs) }} ({{ "%.1f"|format(cost_reduction_pct) }}%)</div>
                {% elif cost_reduction_abs < 0 %}
                <div class="text-xs text-amber-300 mt-1">Increased by {{ "%.2f"|format(-cost_reduction_abs) }} ({{ "%.1f"|format(-cost_reduction_pct) }}%)</div>
                {% else %}
                <div class="text-xs text-gray-400 mt-1">No change</div>
                {% endif %}
            </div>
        </div>
    </div>

    {% if fallback_notice.show %}
    <div class="mt-6 p-4 rounded-lg text-sm {% if fallback_notice.severity == 'danger' %}bg-red-900/20 border border-red-500/30 text-red-200{% elif fallback_notice.severity == 'warning' %}bg-amber-900/20 border border-amber-500/30 text-amber-100{% else %}bg-slate-900/40 border border-slate-500/30 text-slate-200{% endif %}">
        <strong>{{ fallback_notice.title }}</strong><br>
        {{ fallback_notice.message }}<br>
        {% if fallback_notice.action %}
        <span class="font-medium">Action:</span> {{ fallback_notice.action }}<br>
        {% endif %}
        Endpoint: {{ llm_runtime.api_base_url }} | Model: {{ llm_runtime.model_name }} | Fallback calls: {{ fallback_notice.fallback_calls }}/{{ fallback_notice.llm_calls_attempted }}
        {% if fallback_notice.raw_reason and fallback_notice.severity != 'info' %}
        <details class="mt-2 text-xs text-current/90">
            <summary class="cursor-pointer">Technical details</summary>
            <div class="mt-1">{{ fallback_notice.raw_reason }}</div>
        </details>
        {% endif %}
    </div>
    {% endif %}

</div>
</main>

</body>
</html>"""
    with open(os.path.join(tpl_dir, "results.html"), "w", encoding="utf-8") as f:
        f.write(results_html)

if __name__ == "__main__":
    import threading
    import webbrowser
    import uvicorn
    PORT = 5000
    URL = f"http://localhost:{PORT}"

    # Write/refresh templates from embedded HTML
    _write_templates()

    print("=" * 60)
    print("  PromptOptEnv Web Server Starting")
    print("=" * 60)
    print(f"\n  Open your browser to: {URL}")
    print(f"  Templates written to: {os.path.join(BASE_DIR, 'templates')}")
    print(f"  Press Ctrl+C to stop\n")

    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
