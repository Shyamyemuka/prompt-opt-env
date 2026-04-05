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

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Import optimizer components
from prompt_opt_env.server.actions import (
    count_tokens,
    add_context, shorten, add_example, rephrase, add_constraint
)
from openai import OpenAI
import httpx
from rouge_score import rouge_scorer

# ── Environment variables (loaded from .env.local above) ─────────────────────
def _first_env(*keys: str, default: str = "") -> str:
    """Return first non-empty environment value among aliases."""
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return default


API_BASE_URL: str = _first_env("API_BASE_URL", "OPENAI_BASE_URL", default="https://router.huggingface.co/v1/")
MODEL_NAME: str = _first_env("MODEL_NAME", "OPENAI_MODEL", default="Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = _first_env("HF_TOKEN", "OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN", default="")
ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
DONE_THRESHOLD: float = float(os.getenv("DONE_THRESHOLD", "0.85"))
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "7"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN not set. Add it to .env.local as HF_TOKEN=hf_...")
if any(
    os.getenv(name)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
):
    print("[INFO] Proxy environment variables detected. OpenAI client uses direct connection (trust_env=False).")

_CLIENT = None
_LAST_LLM_ERROR = ""
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_client():
    """Lazy load OpenAI client."""
    global _CLIENT
    if _CLIENT is None and HF_TOKEN:
        _CLIENT = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            max_retries=0,
            # Ignore broken shell proxy env vars so API keys/base URL can work directly.
            http_client=httpx.Client(trust_env=False),
        )
    return _CLIENT


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call LLM with the prompt and return output."""
    global _LAST_LLM_ERROR
    client = get_client()
    if not client:
        _LAST_LLM_ERROR = "missing_api_key"
        return ""
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3, timeout=LLM_TIMEOUT_SECONDS,
        )
        _LAST_LLM_ERROR = ""
        return r.choices[0].message.content or ""
    except Exception as exc:
        _LAST_LLM_ERROR = f"{type(exc).__name__}: {exc}"
        return ""


def get_llm_runtime_status() -> dict[str, Any]:
    """Expose sanitized runtime status for debugging connection/key issues."""
    return {
        "api_base_url": API_BASE_URL,
        "model_name": MODEL_NAME,
        "token_present": bool(HF_TOKEN),
        "proxy_env_present": any(
            bool(os.getenv(name))
            for name in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy",
            )
        ),
        "last_llm_error": _LAST_LLM_ERROR,
    }


def score_output(output: str, reference: str, task: str = "General") -> float:
    """Compute ROUGE-L score with task-aware calibration."""
    if (
        not output
        or not reference
        or output.startswith("[Error:")
        or reference.startswith("[Error:")
    ):
        return 0.0
    base_score = _ROUGE.score(reference, output)["rougeL"].fmeasure

    # Summaries should be concise; penalize overly long outputs against compact references.
    if task == "Summarization":
        out_len = max(1, len(output.split()))
        ref_len = max(1, len(reference.split()))
        if out_len > ref_len:
            brevity_factor = max(0.55, min(1.0, (ref_len / out_len) * 1.25))
            base_score *= brevity_factor

    return round(base_score, 4)


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
        "Question Answering": "Answer accurately using only the given context and question.",
        "Paraphrasing": "Rewrite the sentence with the same meaning and better clarity.",
        "Instruction Following": "Follow the instruction precisely and produce the best result.",
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
        return 90 if goal == "Low Cost" else 120
    if task == "Paraphrasing":
        return 120
    if task == "Question Answering":
        return 180
    return 200


def _first_n_words(text: str, n: int) -> str:
    """Return first n words and add ellipsis when truncated."""
    words = text.split()
    if not words:
        return ""
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]).strip() + " ..."


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


def apply_action_for_web(
    action_id: int,
    task: str,
    current_prompt: str,
    input_data: str,
    goal: str,
) -> str:
    """Apply a deterministic action transformation for the web optimizer."""
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
        else:
            constraint = (
                "Keep the answer under 80 words."
                if goal == "Low Cost"
                else "Ensure factual accuracy and clear structure."
            )
        return add_constraint(current_prompt, constraint)
    return current_prompt


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

    if task == "Summarization":
        if current_score >= 0.72:
            return 5
        if "requirement:" not in prompt_lower:
            return 4
        if (
            "example output format:" not in prompt_lower
            and goal == "High Quality"
            and tokens_remaining > 18
            and step <= 1
        ):
            return 2
        if "please" in prompt_lower or "could you" in prompt_lower:
            return 1
        return 5 if step >= 2 else 3

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

    runtime_status = get_llm_runtime_status()

    # Generate a reference answer (ground truth)
    used_fallback = False
    output_token_cap = response_token_limit(task, goal)
    expert_prompt = build_reference_prompt(task, input_data)
    reference_answer = call_llm(expert_prompt, max_tokens=min(220, output_token_cap + 80))
    if not reference_answer:
        used_fallback = True
        reference_answer = fallback_output(task, prompt, input_data, is_reference=True)

    # Adjust token budget based on goal
    token_budget = 80
    if goal == "High Quality":
        token_budget = 100
    elif goal == "Low Cost":
        token_budget = 60

    # Initialize current state variables
    current_prompt = prompt
    current_output = call_llm(
        build_inference_prompt(task, current_prompt, input_data),
        max_tokens=output_token_cap,
    )
    if not current_output:
        used_fallback = True
        current_output = fallback_output(task, current_prompt, input_data)
    current_score = score_output(current_output, reference_answer, task=task)
    current_tokens = count_tokens(current_prompt)

    # Store initial values
    initial_prompt = current_prompt
    initial_output = current_output
    initial_score = current_score
    initial_tokens = current_tokens

    # Optimization loop variables
    best_prompt = current_prompt
    best_output = current_output
    best_score = current_score
    best_tokens = current_tokens
    total_reward = 0.0

    steps_taken = 0
    termination_reason = "max_steps"

    for step in range(MAX_STEPS):
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

        # Apply action natively
        new_prompt = apply_action_for_web(action_id, task, current_prompt, input_data, goal)

        if new_prompt == current_prompt:
            total_reward -= 0.1
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > token_budget:
            total_reward -= 0.5
            termination_reason = "budget_exceeded"
            break

        new_output = call_llm(
            build_inference_prompt(task, new_prompt, input_data),
            max_tokens=output_token_cap,
        )
        if not new_output:
            used_fallback = True
            new_output = fallback_output(task, new_prompt, input_data)
        new_score = score_output(new_output, reference_answer, task=task)

        quality_delta = new_score - current_score
        token_overhead = new_tokens - current_tokens
        step_reward = clip_reward(quality_delta - ALPHA * token_overhead)
        total_reward += step_reward

        # Acceptance logic
        if new_score > best_score or (new_score >= best_score - 0.05 and new_tokens < best_tokens):
            best_prompt = new_prompt
            best_output = new_output
            best_score = new_score
            best_tokens = new_tokens

        current_prompt = new_prompt
        current_output = new_output
        current_score = new_score
        current_tokens = new_tokens

        if current_score > DONE_THRESHOLD:
            total_reward += 1.0
            termination_reason = "success"
            break

    # Calculate final metrics
    improvement_pct = ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
    token_change = best_tokens - initial_tokens
    token_reduction_pct = (abs(token_change) / initial_tokens * 100) if initial_tokens > 0 and token_change < 0 else 0

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
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "used_fallback": used_fallback,
        "fallback_reason": _LAST_LLM_ERROR,
        "llm_runtime": runtime_status,
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
    _write_index_html(tpl_dir)
    _write_results_html(tpl_dir)


# Create FastAPI app
TPL_DIR = os.path.join(BASE_DIR, "templates")
app = FastAPI(title="PromptOptEnv Web")
templates = Jinja2Templates(directory=TPL_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Default prompts for each task
DEFAULT_PROMPTS = {
    "Summarization": "Summarize the following text concisely.",
    "Question Answering": "Answer the question based on the given context.",
    "Paraphrasing": "Paraphrase the following sentence.",
    "Instruction Following": "Follow the instruction and generate an appropriate response."
}


@app.post("/optimize", response_class=HTMLResponse)
async def optimize(
    request: Request,
    task: str = Form("Summarization"),
    input_text: str = Form(""),
    context: str = Form(""),
    question: str = Form(""),
    user_prompt: str = Form(""),
    prompt: str = Form(""),
    goal: str = Form("Balanced")
):
    """Handle optimization form submission with structured task input."""

    # Backward compatibility:
    # - accept legacy `prompt` field used by older UI versions
    # - default unknown/missing task to Summarization
    selected_task = task if task in DEFAULT_PROMPTS else "Summarization"
    provided_prompt = user_prompt.strip() if user_prompt.strip() else prompt.strip()

    # Construct state object
    state = {
        "task": selected_task,
        "input": "",
        "prompt": provided_prompt if provided_prompt else DEFAULT_PROMPTS[selected_task],
        "goal": goal
    }

    # Process input based on task
    if selected_task == "Question Answering":
        state["input"] = f"Context: {context}\nQuestion: {question}" if context or question else ""
    else:
        state["input"] = input_text

    # Pass state to optimizer (using existing backend logic)
    results = optimize_prompt_with_state(state)
    return templates.TemplateResponse("results.html", {
        "request": request,
        **results
    })


def _write_index_html(tpl_dir: str):
    """Write index.html into tpl_dir."""
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PromptOptEnv - RL Prompt Optimizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: rgba(255,255,255,0.8);
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        textarea, input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            background: white;
        }
        textarea:focus, input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea { min-height: 100px; resize: vertical; }
        .hint {
            font-size: 13px;
            color: #666;
            margin-top: 5px;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .optional-fields {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .optional-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PromptOptEnv</h1>
        <p class="subtitle">RL-Powered Prompt Optimization with Cost Awareness</p>

        <div class="card">
            <form action="/optimize" method="post">
                <div class="form-group">
                    <label for="task">Select Task</label>
                    <select name="task" id="task">
                        <option value="Summarization">Summarization</option>
                        <option value="Question Answering">Question Answering</option>
                        <option value="Paraphrasing">Paraphrasing</option>
                        <option value="Instruction Following">Instruction Following</option>
                    </select>
                </div>

                <div class="form-group" id="generic-input-group">
                    <label for="input_text" id="generic-input-label">Input Text</label>
                    <textarea name="input_text" id="input_text" placeholder="Paste the paragraph you want to summarize..."></textarea>
                </div>

                <div id="qa-input-group" class="hidden">
                    <div class="form-group">
                        <label for="context">Context</label>
                        <textarea name="context" id="context" placeholder="Enter context passage..."></textarea>
                    </div>

                    <div class="form-group">
                        <label for="question">Question</label>
                        <textarea name="question" id="question" placeholder="Enter your question..."></textarea>
                    </div>
                </div>

                <div class="optional-fields">
                    <p class="optional-title">Optional Configuration</p>

                    <div class="form-group">
                        <label for="user_prompt">Initial Prompt (Optional)</label>
                        <textarea
                            name="user_prompt"
                            id="user_prompt"
                            placeholder="e.g., Summarize the following text clearly and concisely"
                        ></textarea>
                        <p class="hint">Leave blank to use a task-specific default prompt</p>
                    </div>

                    <div class="form-group">
                        <label for="goal">Optimization Goal</label>
                        <select name="goal" id="goal">
                            <option value="Balanced" selected>Balanced</option>
                            <option value="High Quality">High Quality</option>
                            <option value="Low Cost">Low Cost</option>
                        </select>
                    </div>
                </div>

                <button type="submit">Optimize Prompt</button>
            </form>
        </div>
    </div>
    <script>
        const taskField = document.getElementById("task");
        const genericInputGroup = document.getElementById("generic-input-group");
        const genericInputLabel = document.getElementById("generic-input-label");
        const genericInput = document.getElementById("input_text");
        const qaInputGroup = document.getElementById("qa-input-group");

        const inputConfig = {
            "Summarization": {
                label: "Input Text",
                placeholder: "Paste the paragraph you want to summarize..."
            },
            "Paraphrasing": {
                label: "Sentence",
                placeholder: "Enter a sentence to paraphrase..."
            },
            "Instruction Following": {
                label: "Instruction",
                placeholder: "e.g., Explain photosynthesis in simple terms"
            }
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
            const config = inputConfig[selectedTask];
            genericInputLabel.textContent = config.label;
            genericInput.placeholder = config.placeholder;
        }

        taskField.addEventListener("change", updateTaskInputs);
        updateTaskInputs();
    </script>
</body>
</html>'''

    with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)


def _write_results_html(tpl_dir: str):
    """Write results.html into tpl_dir."""
    results_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Results - PromptOptEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
            color: #333;
        }
        .container { max-width: 700px; margin: 0 auto; }
        .card {
            background: white; border-radius: 12px; padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            font-size: 16px; line-height: 1.6;
        }
        .task-header {
            font-size: 20px; font-weight: bold; margin-bottom: 25px;
        }
        hr {
            border: 0; height: 1px; background: #ddd; margin: 30px 0;
        }
        .label {
            font-weight: bold; margin-bottom: 5px;
        }
        .quote {
            font-style: italic; color: #555; margin-bottom: 20px;
            background: #f9f9f9; padding: 10px; border-left: 4px solid #667eea;
        }
        .success-text {
            color: #2e7d32; font-weight: bold; margin-top: 20px;
        }
        .info-text {
            color: #5f6368; margin-top: 14px; font-size: 14px;
        }
        .debug-box {
            margin-top: 10px;
            padding: 10px;
            background: #f4f6fb;
            border: 1px solid #dbe1f0;
            border-radius: 8px;
            color: #3f4a5f;
            font-size: 13px;
            line-height: 1.45;
        }
        .back-btn {
            display: inline-block; margin-top: 30px; padding: 12px 30px;
            background: white; color: #667eea; text-decoration: none;
            border-radius: 8px; font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        
        <div class="card">
            <div class="task-header">Task: {{ task }}</div>
            
            <div class="label">Initial Prompt:</div>
            <div class="quote">{{ initial_prompt }}</div>
            
            <div class="label">Output:</div>
            <div class="quote">{{ initial_output or '[No output generated]' }}</div>
            
            <div class="label" style="margin-bottom: 30px;">Reward: {{ "%.2f"|format(initial_reward) }}</div>
            
            <hr>
            
            <div class="task-header">After Optimization ({{ steps_taken }} steps):</div>
            
            <div class="label">Prompt:</div>
            <div class="quote">{{ final_prompt }}</div>
            
            <div class="label">Output:</div>
            <div class="quote">{{ final_output or '[No output generated]' }}</div>
            
            <div class="label">Reward: {{ "%.2f"|format(final_reward) }}</div>
            <div class="label">Total Episode Reward: {{ "%.2f"|format(total_reward) }}</div>
            
            {% if token_change < 0 %}
            <div class="success-text">
                Token cost reduced by {{ "%.0f"|format(token_reduction_pct) }}%
            </div>
            {% elif token_change > 0 %}
            <div class="success-text">
                Token cost increased by {{ token_change }} tokens
            </div>
            {% else %}
            <div class="success-text">
                Token cost unchanged
            </div>
            {% endif %}
            {% if used_fallback %}
            <div class="info-text">
                API connection failed, so offline fallback outputs were used for this run.
            </div>
            <div class="debug-box">
                <div>Reason: {{ fallback_reason or "unknown_connection_error" }}</div>
                <div>Token detected: {{ "yes" if llm_runtime.token_present else "no" }}</div>
                <div>Model: {{ llm_runtime.model_name }}</div>
                <div>Base URL: {{ llm_runtime.api_base_url }}</div>
            </div>
            {% endif %}
        </div>

        <div style="text-align: center;">
            <a href="/" class="back-btn">&#8592; Optimize Another</a>
        </div>
    </div>
</body>
</html>'''

    with open(os.path.join(tpl_dir, "results.html"), "w", encoding="utf-8") as f:
        f.write(results_html)


if __name__ == "__main__":
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

    # Open browser after a short delay so the server has time to start
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT)
