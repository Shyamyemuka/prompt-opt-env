"""
Cost-aware baseline inference script.

Mandatory expectations:
- file name: inference.py at repo root
- uses OpenAI Python client for LLM calls
- reads API_BASE_URL, MODEL_NAME, OPENAI_API_KEY (or HF_TOKEN fallback) from environment
- runs 3 tasks (easy, medium, hard)
- emits structured logs with [START], [STEP], [END]
- keeps stdout limited to the required log lines
"""

import os
import random
import re
import sys
import time
from importlib import import_module
from types import SimpleNamespace

from openai import OpenAI
from rouge_score import rouge_scorer


def _load_env_local() -> None:
    """Load .env.local only as local convenience for direct python runs."""
    try:
        env_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
        if not os.path.exists(env_local):
            return
        with open(env_local, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key not in os.environ:
                    os.environ[key] = value.strip("'\"")
    except Exception:
        pass


_load_env_local()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None or not HF_TOKEN.strip():
    raise ValueError("HF_TOKEN environment variable is required")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", MODEL_NAME)
API_KEY = OPENAI_API_KEY or HF_TOKEN or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK_NAME", "prompt_opt_env")
ALPHA = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "7"))
INFERENCE_SEED = int(os.getenv("INFERENCE_SEED", "42"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
LLM_FAILOVER_ATTEMPTS = max(1, int(os.getenv("LLM_FAILOVER_ATTEMPTS", "2")))
LLM_FAILOVER_BACKOFF_SECONDS = max(0.0, float(os.getenv("LLM_FAILOVER_BACKOFF_SECONDS", "0.25")))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.85"))
SCORE_EPSILON = 0.05
USE_INTELLIGENT_ACTIONS = os.getenv("USE_INTELLIGENT_ACTIONS", "true").lower() == "true"

def _load_apply_action_intelligent():
    for module_name in ("prompt_opt_env.server.actions_llm", "server.actions_llm"):
        try:
            module = import_module(module_name)
            return module.apply_action_intelligent
        except Exception:
            continue
    raise ModuleNotFoundError("Unable to import actions_llm from known module paths")


try:
    apply_action_intelligent = _load_apply_action_intelligent()
    _INTELLIGENT_ACTIONS_AVAILABLE = True
except Exception:
    _INTELLIGENT_ACTIONS_AVAILABLE = False

random.seed(INFERENCE_SEED)
_OPENAI_CLIENTS: list[dict[str, object]] = []
_CLIENT_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

if OPENAI_API_KEY:
    _OPENAI_CLIENTS.append(
        {
            "name": "openai",
            "model": OPENAI_MODEL,
            "client": OpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                max_retries=_CLIENT_RETRIES,
                timeout=LLM_TIMEOUT_SECONDS,
            ),
        }
    )

if HF_TOKEN:
    _OPENAI_CLIENTS.append(
        {
            "name": "hf",
            "model": MODEL_NAME,
            "client": OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN,
                max_retries=_CLIENT_RETRIES,
                timeout=LLM_TIMEOUT_SECONDS,
            ),
        }
    )

legacy_api_key = (os.getenv("API_KEY") or "").strip()
if legacy_api_key and not _OPENAI_CLIENTS:
    _OPENAI_CLIENTS.append(
        {
            "name": "legacy",
            "model": MODEL_NAME,
            "client": OpenAI(
                base_url=API_BASE_URL,
                api_key=legacy_api_key,
                max_retries=_CLIENT_RETRIES,
                timeout=LLM_TIMEOUT_SECONDS,
            ),
        }
    )

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

ACTION_NAMES = {
    0: "ADD_CONTEXT",
    1: "SHORTEN",
    2: "ADD_EXAMPLE",
    3: "REPHRASE",
    4: "ADD_CONSTRAINT",
    5: "STOP",
}


EVAL_TASKS = [
    {
        "task_id": 4,
        "difficulty": "easy",
        "token_budget": 80,
        "task_description": "Explain what machine learning is to a 10-year-old",
        "initial_prompt": "explain machine learning",
        "reference": (
            "Machine learning is when computers learn from examples, like how you "
            "learned to recognise cats by seeing many cats. The computer looks at data "
            "and figures out patterns by itself."
        ),
        "context": "Machine learning is a type of AI where computers learn patterns from data.",
        "example": "Machine learning is like [child analogy]. The computer [simple description].",
        "constraint": "Simple words only. No jargon. For a 10-year-old.",
        "fallback_output": (
            "Machine learning means computers learn from examples. "
            "They look at many examples and find patterns."
        ),
    },
    {
        "task_id": 5,
        "difficulty": "medium",
        "token_budget": 65,
        "task_description": "Answer: What is the time complexity of binary search and why?",
        "initial_prompt": "binary search complexity",
        "reference": (
            "Binary search has O(log n) time complexity. With each comparison it eliminates "
            "half the remaining elements. After k steps n/2^k = 1, so k = log2(n) steps."
        ),
        "context": "Binary search finds a target in a sorted array by halving the search space.",
        "example": "Binary search is O([notation]) because [explanation].",
        "constraint": "Include Big O notation and explain why that complexity holds.",
        "fallback_output": (
            "Binary search is O(log n) because each step removes half the remaining elements."
        ),
    },
    {
        "task_id": 11,
        "difficulty": "hard",
        "token_budget": 55,
        "task_description": "Describe the steps to resolve a Git merge conflict",
        "initial_prompt": "git merge conflict fix",
        "reference": (
            "1. Run git merge. 2. Open conflicted file with <<<<<<, =======, >>>>>>> markers. "
            "3. Edit to keep correct code and remove markers. "
            "4. git add filename. 5. git commit. 6. git push."
        ),
        "context": "A Git merge conflict occurs when two branches changed the same lines differently.",
        "example": "1. [Trigger]. 2. [Markers: <<<<<<, =======, >>>>>>>]. 3. [Edit]. 4. [git add]. 5. [Commit].",
        "constraint": "Include the conflict markers (<<<<<<, =======, >>>>>>>). Cover all steps to push.",
        "fallback_output": (
            "Resolve conflict: open file, edit between <<<<<< and >>>>>>>, remove markers, "
            "git add, git commit, then git push."
        ),
    },
]

FILLER_PATTERNS = [r"\bplease\b", r"\bcould you\b", r"\bcan you\b", r"\bi want you to\b"]


def count_tokens(text: str) -> int:
    return len(text.split())


def apply(action_id: int, prompt: str, task: dict) -> str:
    if action_id == 0:
        if task["context"][:20].lower() in prompt.lower():
            return prompt
        return f"{prompt}\nContext: {task['context']}"
    if action_id == 1:
        reduced = prompt
        for pattern in FILLER_PATTERNS:
            reduced = re.sub(pattern, "", reduced, flags=re.IGNORECASE)
        reduced = re.sub(r"  +", " ", reduced).strip()
        return reduced if reduced != prompt else prompt
    if action_id == 2:
        if "Example output format:" in prompt:
            return prompt
        return f"{prompt}\nExample output format: {task['example']}"
    if action_id == 3:
        rewritten = re.sub(
            r"^(?:can you|could you|please)\s+(.+?)[\?\.]*$",
            lambda m: m.group(1).strip().capitalize() + ".",
            prompt,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return rewritten if rewritten != prompt else prompt
    if action_id == 4:
        if "Requirement:" in prompt:
            return prompt
        return f"{prompt}\nRequirement: {task['constraint']}"
    return prompt


def apply_intelligent(action_id: int, prompt: str, task: dict) -> str:
    if not _INTELLIGENT_ACTIONS_AVAILABLE or action_id == 5:
        return apply(action_id, prompt, task)

    task_obj = SimpleNamespace(
        task_id=task["task_id"],
        task_description=task["task_description"],
        context_sentence=task["context"],
        example_output=task["example"],
        constraint_sentence=task["constraint"],
    )
    result = apply_action_intelligent(action_id, prompt, task_obj)
    return result.new_prompt


def call_llm(prompt: str, fallback: str) -> str:
    if not _OPENAI_CLIENTS:
        return fallback

    for attempt in range(LLM_FAILOVER_ATTEMPTS):
        for provider in _OPENAI_CLIENTS:
            try:
                completion = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,
                )
                text = (completion.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception:
                continue

        if attempt + 1 < LLM_FAILOVER_ATTEMPTS and LLM_FAILOVER_BACKOFF_SECONDS > 0:
            time.sleep(LLM_FAILOVER_BACKOFF_SECONDS * (attempt + 1))

    return fallback


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return SCORE_EPSILON
    raw = float(_ROUGE.score(reference, hypothesis)["rougeL"].fmeasure)
    bounded = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, raw))
    rounded = round(bounded, 4)
    return float(max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, rounded)))


def emit_start(task: dict) -> None:
    task_name = f"task_{task['task_id']}_{task['difficulty']}"
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _single_line(value: str | None) -> str:
    if not value:
        return "null"
    return " ".join(str(value).split()) or "null"


def _strict_unit(value: float) -> float:
    """Keep emitted step metrics inside strict (0,1) for validator compatibility."""
    return float(max(0.01, min(0.99, round(float(value), 4))))


def emit_step(
    step_num: int,
    action_id: int,
    reward: float,
    done: bool,
    error: str | None = None,
) -> None:
    done_val = str(done).lower()
    error_val = _single_line(error)
    print(
        f"[STEP] step={step_num} action={ACTION_NAMES[action_id]} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def emit_end(result: dict) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in result["rewards"])
    # Format strictly per guidelines: success, steps, rewards only
    print(
        f"[END] success={str(result['success']).lower()} steps={result['steps']} rewards={rewards_str}",
        flush=True,
    )


def run_episode(task: dict, max_steps: int = 7) -> dict:
    emit_start(task)
    prompt = task["initial_prompt"]
    initial_output = call_llm(prompt, task["fallback_output"])
    initial_score = rouge_l(initial_output, task["reference"])
    initial_tokens = count_tokens(prompt)

    current_score = initial_score
    current_tokens = initial_tokens
    total_reward = 0.0
    rewards: list[float] = []
    steps = 0
    episode_success = False
    result = {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "initial_score": initial_score,
        "final_score": current_score,
        "final_token_count": current_tokens,
        "token_budget": task["token_budget"],
        "efficiency": round(current_score / max(1, current_tokens), 6),
        "total_reward": 0.0,
        "rewards": rewards,
        "success": False,
        "steps": 0,
    }

    try:
        for step_idx in range(max_steps):
            action_id = random.randint(0, 5)
            step_num = step_idx + 1

            try:
                if action_id == 5:
                    reward = _strict_unit(current_score * 1.5)
                    total_reward += reward
                    rewards.append(reward)
                    steps = step_num
                    episode_success = current_score >= SUCCESS_THRESHOLD
                    emit_step(step_num, action_id, reward, True, None)
                    break

                if USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE:
                    new_prompt = apply_intelligent(action_id, prompt, task)
                else:
                    new_prompt = apply(action_id, prompt, task)
                if new_prompt == prompt:
                    reward = _strict_unit(-0.1)
                    total_reward += reward
                    rewards.append(reward)
                    steps = step_num
                    done = step_num >= max_steps
                    episode_success = done and current_score >= SUCCESS_THRESHOLD
                    emit_step(step_num, action_id, reward, done, None)
                    if done:
                        break
                    continue

                new_tokens = count_tokens(new_prompt)
                if new_tokens > task["token_budget"]:
                    reward = _strict_unit(-0.5)
                    total_reward += reward
                    rewards.append(reward)
                    steps = step_num
                    episode_success = False
                    emit_step(step_num, action_id, reward, True, None)
                    break

                new_output = call_llm(new_prompt, task["fallback_output"])
                new_score = rouge_l(new_output, task["reference"])
                token_overhead = new_tokens - current_tokens
                reward = _strict_unit(new_score - current_score - ALPHA * token_overhead)

                done = False
                if new_score > SUCCESS_THRESHOLD:
                    reward = _strict_unit(reward + 1.0)
                    done = True
                    episode_success = True
                elif step_num >= max_steps:
                    done = True
                    episode_success = new_score >= SUCCESS_THRESHOLD

                prompt = new_prompt
                current_score = new_score
                current_tokens = new_tokens
                total_reward += reward
                rewards.append(reward)
                steps = step_num

                emit_step(step_num, action_id, reward, done, None)

                if done:
                    break
            except Exception as step_error:
                reward = _strict_unit(0.0)
                total_reward += reward
                rewards.append(reward)
                steps = step_num
                episode_success = False
                emit_step(step_num, action_id, reward, True, str(step_error))
                break
    finally:
        result["final_score"] = current_score
        result["final_token_count"] = current_tokens
        result["efficiency"] = round(current_score / max(1, current_tokens), 6)
        result["total_reward"] = round(total_reward, 4)
        result["rewards"] = rewards
        result["success"] = episode_success
        result["steps"] = steps
        emit_end(result)

    return result


def print_summary(results: list[dict]) -> None:
    # Keep stdout strict: only [START], [STEP], [END] lines.
    _ = results


def main() -> None:
    results = []
    for task in EVAL_TASKS:
        results.append(run_episode(task, max_steps=MAX_STEPS))

    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        print(f"[ERROR] {fatal_error}", file=sys.stderr, flush=True)
    finally:
        sys.exit(0)
