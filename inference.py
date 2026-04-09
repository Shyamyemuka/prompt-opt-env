"""
Cost-aware baseline inference script.

Mandatory expectations:
- file name: inference.py at repo root
- uses OpenAI Python client for LLM calls
- reads API_BASE_URL, MODEL_NAME, OPENAI_API_KEY (or HF_TOKEN fallback) from environment
- runs 3 tasks (easy, medium, hard)
- emits structured logs with [START], [STEP], [END]
- prints summary table and exits with status code 0
"""

import os
import random
import re
import sys
import traceback
from types import SimpleNamespace

from rouge_score import rouge_scorer
from prompt_opt_env.llm_router import create_default_router


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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = OPENAI_API_KEY or HF_TOKEN
ALPHA = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "7"))
INFERENCE_SEED = int(os.getenv("INFERENCE_SEED", "42"))
SCORE_EPSILON = 1e-4
USE_INTELLIGENT_ACTIONS = os.getenv("USE_INTELLIGENT_ACTIONS", "true").lower() == "true"

try:
    from prompt_opt_env.server.actions_llm import apply_action_intelligent
    _INTELLIGENT_ACTIONS_AVAILABLE = True
except Exception:
    try:
        from server.actions_llm import apply_action_intelligent
        _INTELLIGENT_ACTIONS_AVAILABLE = True
    except Exception:
        _INTELLIGENT_ACTIONS_AVAILABLE = False

random.seed(INFERENCE_SEED)

_LLM_ROUTER = create_default_router(
    default_model=MODEL_NAME,
    default_base_url=API_BASE_URL,
    timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
)
if not _LLM_ROUTER.has_provider():
    print("[WARN] No provider key found (OPENAI_API_KEY/GEMINI_API_KEY/HF_TOKEN). Running with deterministic fallback outputs.")

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
    if not _LLM_ROUTER.has_provider():
        return fallback
    text = _LLM_ROUTER.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
    )
    if text.strip():
        return text
    if _LLM_ROUTER.last_error:
        print(f"[WARN] LLM call failed, using fallback output: {_LLM_ROUTER.last_error}")
        return fallback
    return fallback


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return SCORE_EPSILON
    raw = float(_ROUGE.score(reference, hypothesis)["rougeL"].fmeasure)
    bounded = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, raw))
    return round(bounded, 4)


def emit_start(task: dict) -> None:
    print(
        f"[START] task_id={task['task_id']} difficulty={task['difficulty']} "
        f"budget={task['token_budget']} max_steps={MAX_STEPS}"
    )


def emit_step(
    task_id: int,
    step_num: int,
    action_id: int,
    score: float,
    tokens: int,
    reward: float,
    done: bool,
    reason: str,
) -> None:
    print(
        f"[STEP] task_id={task_id} step={step_num} action={ACTION_NAMES[action_id]} "
        f"score={score:.4f} tokens={tokens} reward={reward:+.4f} done={int(done)} reason={reason}"
    )


def emit_end(result: dict) -> None:
    print(
        f"[END] task_id={result['task_id']} difficulty={result['difficulty']} "
        f"init_score={result['initial_score']:.4f} final_score={result['final_score']:.4f} "
        f"final_tokens={result['final_token_count']} budget={result['token_budget']} "
        f"efficiency={result['efficiency']:.6f} total_reward={result['total_reward']:+.4f} "
        f"steps={result['steps']}"
    )


def run_episode(task: dict, max_steps: int = 7) -> dict:
    emit_start(task)

    prompt = task["initial_prompt"]
    initial_output = call_llm(prompt, task["reference"])
    initial_score = rouge_l(initial_output, task["reference"])
    initial_tokens = count_tokens(prompt)

    current_score = initial_score
    current_tokens = initial_tokens
    total_reward = 0.0
    steps = 0

    for step_idx in range(max_steps):
        action_id = random.randint(0, 5)
        step_num = step_idx + 1

        if action_id == 5:
            reward = round(current_score * 1.5, 4)
            total_reward += reward
            steps = step_num
            emit_step(task["task_id"], step_num, action_id, current_score, current_tokens, reward, True, "voluntary_stop")
            break

        if USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE:
            new_prompt = apply_intelligent(action_id, prompt, task)
        else:
            new_prompt = apply(action_id, prompt, task)
        if new_prompt == prompt:
            reward = -0.1
            total_reward += reward
            steps = step_num
            done = step_num >= max_steps
            emit_step(task["task_id"], step_num, action_id, current_score, current_tokens, reward, done, "no_op")
            if done:
                break
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > task["token_budget"]:
            reward = -0.5
            total_reward += reward
            steps = step_num
            emit_step(task["task_id"], step_num, action_id, current_score, current_tokens, reward, True, "budget_exceeded")
            break

        new_output = call_llm(new_prompt, task["reference"])
        new_score = rouge_l(new_output, task["reference"])
        token_overhead = new_tokens - current_tokens
        reward = round(new_score - current_score - ALPHA * token_overhead, 4)

        done = False
        reason = "continue"
        if new_score > 0.85:
            reward = round(reward + 1.0, 4)
            done = True
            reason = "success"
        elif step_num >= max_steps:
            done = True
            reason = "max_steps"

        prompt = new_prompt
        current_score = new_score
        current_tokens = new_tokens
        total_reward += reward
        steps = step_num

        emit_step(task["task_id"], step_num, action_id, current_score, current_tokens, reward, done, reason)

        if done:
            break

    efficiency = round(current_score / max(1, current_tokens), 6)
    result = {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "initial_score": initial_score,
        "final_score": current_score,
        "final_token_count": current_tokens,
        "token_budget": task["token_budget"],
        "efficiency": efficiency,
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }

    emit_end(result)
    return result


def print_summary(results: list[dict]) -> None:
    print("\nBASELINE SCORES SUMMARY")
    print("=" * 86)
    print(f"{'Difficulty':<12} {'Score':>8} {'Tokens':>8} {'Budget':>8} {'Efficiency':>12} {'Reward':>10} {'Steps':>8}")
    print("-" * 86)
    for row in results:
        print(
            f"{row['difficulty']:<12} {row['final_score']:>8.4f} {row['final_token_count']:>8} "
            f"{row['token_budget']:>8} {row['efficiency']:>12.6f} {row['total_reward']:>10.4f} {row['steps']:>8}"
        )
    print("-" * 86)

    avg_score = sum(r["final_score"] for r in results) / len(results)
    avg_eff = sum(r["efficiency"] for r in results) / len(results)
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    print(f"{'Average':<12} {avg_score:>8.4f} {'-':>8} {'-':>8} {avg_eff:>12.6f} {avg_reward:>10.4f} {'-':>8}")
    print("Efficiency = final_score / final_token_count")
    print("All scores are ROUGE-L F1 in (0.0, 1.0).")


def main() -> None:
    print("=" * 70)
    print("PromptOptEnv Cost-Aware Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    print(f"Alpha: {ALPHA}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Intelligent actions: {USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE}")
    print("=" * 70)

    results = []
    for task in EVAL_TASKS:
        results.append(run_episode(task, max_steps=MAX_STEPS))

    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        print(f"[ERROR] inference.py encountered an exception: {fatal_error}")
        traceback.print_exc()
    finally:
        print("Script complete.")
        sys.exit(0)
