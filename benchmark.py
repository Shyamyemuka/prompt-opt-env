"""
Benchmark Suite for PromptOptEnv.

Compares multiple agent strategies:
- heuristic: Domain-knowledge rules
- random: Uniform random baseline
- immediate_stop: Tests if editing helps
- always_improve: Tests cost of ignoring budget

Outputs comparison table with efficiency, success rate, and other metrics.
"""

import os
import random
import sys
import traceback
from importlib import import_module
from typing import Callable
from types import SimpleNamespace

# Load .env.local for local runs
if os.path.exists(".env.local"):
    with open(".env.local") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                if key not in os.environ:
                    os.environ[key] = val.strip('"\'')

from agent import HeuristicAgent, RandomAgent, ImmediateStopAgent, AlwaysImproveAgent
from rouge_score import rouge_scorer
from prompt_opt_env.llm_router import create_default_router

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", "7"))
DONE_THRESHOLD = float(os.getenv("DONE_THRESHOLD", "0.85"))
TOKEN_PENALTY_ALPHA = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))
USE_INTELLIGENT = os.getenv("USE_INTELLIGENT", "false").lower() == "true"
USE_INTELLIGENT_ACTIONS = os.getenv("USE_INTELLIGENT_ACTIONS", "true").lower() == "true" or USE_INTELLIGENT

def _load_actions_llm():
    for module_name in ("prompt_opt_env.server.actions_llm", "server.actions_llm"):
        try:
            module = import_module(module_name)
            return module.apply_action_intelligent, module.get_action_llm_stats
        except Exception:
            continue
    raise ModuleNotFoundError("Unable to import actions_llm from known module paths")


try:
    apply_action_intelligent, get_action_llm_stats = _load_actions_llm()
    _INTELLIGENT_ACTIONS_AVAILABLE = True
except Exception:
    _INTELLIGENT_ACTIONS_AVAILABLE = False

_LLM_ROUTER = create_default_router(
    default_model=MODEL_NAME,
    default_base_url=API_BASE_URL,
    timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
)

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
SCORE_EPSILON = 0.11


def strict_unit_interval(value: float) -> float:
    """Keep benchmark-facing scores and rewards away from closed-interval edges."""
    return float(max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, round(float(value), 4))))

# Test tasks (same as inference.py)
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


def apply_action(action_id: int, prompt: str, task: dict) -> str:
    if action_id == 0:
        if task["context"][:20].lower() in prompt.lower():
            return prompt
        return f"{prompt}\nContext: {task['context']}"
    if action_id == 1:
        import re
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
        import re
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


def apply_action_intelligent_for_task(action_id: int, prompt: str, task: dict) -> str:
    """Apply LLM-powered transform with deterministic fallback in module itself."""
    if not _INTELLIGENT_ACTIONS_AVAILABLE or action_id == 5:
        return apply_action(action_id, prompt, task)

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
        print(f"[WARN] LLM call failed: {_LLM_ROUTER.last_error}")
        return fallback
    return fallback


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return SCORE_EPSILON
    raw = float(_ROUGE.score(reference, hypothesis)["rougeL"].fmeasure)
    bounded = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, raw))
    return strict_unit_interval(bounded)


def run_episode(task: dict, agent, max_steps: int = 7) -> dict:
    """Run single episode with given agent."""
    agent.reset()

    prompt = task["initial_prompt"]
    initial_output = call_llm(prompt, task["reference"])
    initial_score = rouge_l(initial_output, task["reference"])
    initial_tokens = count_tokens(prompt)

    current_score = initial_score
    current_tokens = initial_tokens
    total_reward = 0.0
    steps = 0

    for step_idx in range(max_steps):
        action_id, reason = agent.select_action(
            current_score=current_score,
            current_token_count=current_tokens,
            token_budget=task["token_budget"],
            step_count=step_idx,
            max_steps=max_steps,
        )

        # STOP action
        if action_id == 5:
            reward = strict_unit_interval(current_score * 1.5)
            total_reward += reward
            steps = step_idx + 1
            agent.update(action_id, {"done": True})
            break

        # Apply action (intelligent rewrite path preferred when enabled)
        if USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE:
            new_prompt = apply_action_intelligent_for_task(action_id, prompt, task)
        else:
            new_prompt = apply_action(action_id, prompt, task)

        # No-op check
        if new_prompt == prompt:
            reward = strict_unit_interval(-0.1)
            total_reward += reward
            steps = step_idx + 1
            done = step_idx + 1 >= max_steps
            agent.update(action_id, {"done": done})
            if done:
                break
            continue

        # Budget check
        new_tokens = count_tokens(new_prompt)
        if new_tokens > task["token_budget"]:
            reward = strict_unit_interval(-0.5)
            total_reward += reward
            steps = step_idx + 1
            agent.update(action_id, {"done": True})
            break

        # Score new prompt
        new_output = call_llm(new_prompt, task["reference"])
        new_score = rouge_l(new_output, task["reference"])
        token_overhead = new_tokens - current_tokens
        quality_delta = new_score - current_score
        reward = strict_unit_interval(quality_delta - TOKEN_PENALTY_ALPHA * token_overhead)

        done = False
        if new_score > DONE_THRESHOLD:
            reward = strict_unit_interval(reward + 1.0)
            done = True
        elif step_idx + 1 >= max_steps:
            done = True

        prompt = new_prompt
        current_score = new_score
        current_tokens = new_tokens
        total_reward += reward
        steps = step_idx + 1

        agent.update(action_id, {"done": done})

        if done:
            break

    efficiency = round(current_score / max(1, current_tokens), 6)
    budget_exceeded = current_tokens > task["token_budget"]
    success = current_score > DONE_THRESHOLD

    return {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "initial_score": initial_score,
        "final_score": current_score,
        "final_token_count": current_tokens,
        "token_budget": task["token_budget"],
        "efficiency": efficiency,
        "total_reward": strict_unit_interval(total_reward),
        "steps": steps,
        "success": success,
        "budget_exceeded": budget_exceeded,
        "agent_stats": agent.get_stats(),
    }


def benchmark_agent(name: str, agent_factory: Callable, n_episodes: int = 3) -> dict:
    """Run benchmark for a single agent strategy."""
    print(f"\n  Running {name}...", end="", flush=True)
    results = []

    for episode in range(n_episodes):
        for task in EVAL_TASKS:
            agent = agent_factory()
            result = run_episode(task, agent, max_steps=MAX_STEPS)
            result["episode"] = episode + 1
            results.append(result)

    # Calculate aggregates
    n = len(results)
    avg_efficiency = sum(r["efficiency"] for r in results) / n
    avg_score = sum(r["final_score"] for r in results) / n
    avg_tokens = sum(r["final_token_count"] for r in results) / n
    avg_reward = sum(r["total_reward"] for r in results) / n
    avg_steps = sum(r["steps"] for r in results) / n
    success_rate = sum(1 for r in results if r["success"]) / n * 100
    budget_compliance = sum(1 for r in results if not r["budget_exceeded"]) / n * 100

    print(f" Done ({n} runs)")

    return {
        "name": name,
        "n_runs": n,
        "avg_efficiency": avg_efficiency,
        "avg_score": avg_score,
        "avg_tokens": avg_tokens,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
        "budget_compliance": budget_compliance,
        "results": results,
    }


def print_summary(benchmarks: list[dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("PROMPTOPTENV AGENT BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Strategy':<20} {'Efficiency':>12} {'Score':>10} {'Tokens':>8} {'Reward':>10} {'Steps':>6} {'Success%':>10} {'Budget%':>10}")
    print("-" * 100)
    print(
        f"Intelligent actions enabled: {USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE}"
    )
    if USE_INTELLIGENT_ACTIONS and _INTELLIGENT_ACTIONS_AVAILABLE:
        stats = get_action_llm_stats()
        print(
            f"Action-LLM calls: {stats['calls']} (cache hits: {stats['cache_hits']}, cache size: {stats['cache_size']})"
        )

    for bm in benchmarks:
        print(
            f"{bm['name']:<20} "
            f"{bm['avg_efficiency']:>12.6f} "
            f"{bm['avg_score']:>10.4f} "
            f"{bm['avg_tokens']:>8.1f} "
            f"{bm['avg_reward']:>+10.4f} "
            f"{bm['avg_steps']:>6.1f} "
            f"{bm['success_rate']:>9.1f}% "
            f"{bm['budget_compliance']:>9.1f}%"
        )

    print("-" * 100)

    # Find best
    best = max(benchmarks, key=lambda x: x["avg_efficiency"])
    print(f"\nBest Efficiency: {best['name']} ({best['avg_efficiency']:.6f})")

    # Calculate improvement over random
    random_bm = next((b for b in benchmarks if b["name"] == "random"), None)
    heuristic_bm = next((b for b in benchmarks if b["name"] == "heuristic"), None)

    if random_bm and heuristic_bm:
        improvement = (heuristic_bm["avg_efficiency"] - random_bm["avg_efficiency"]) / random_bm["avg_efficiency"] * 100
        print(f"Heuristic vs Random: {improvement:+.1f}% improvement")

    print("\nNotes:")
    print("- Efficiency = final_score / final_token_count (higher is better)")
    print("- Budget% = percentage of episodes staying under token budget")
    print("- Success% = percentage of episodes achieving ROUGE-L > 0.85")


def main():
    print("=" * 70)
    print("PromptOptEnv Benchmark Suite")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    print(f"Max steps: {MAX_STEPS}, Token alpha: {TOKEN_PENALTY_ALPHA}")
    print("=" * 70)

    if not _LLM_ROUTER.has_provider():
        print("\n[WARN] No API key found. Using fallback LLM outputs.")
        print("Results will be deterministic but may not reflect real performance.\n")

    # Define agents to benchmark
    agents = [
        ("heuristic", lambda: HeuristicAgent(random_seed=42)),
        ("random", lambda: RandomAgent(random_seed=42)),
        ("immediate_stop", lambda: ImmediateStopAgent()),
        ("always_improve", lambda: AlwaysImproveAgent(random_seed=42)),
    ]

    # Run benchmarks
    benchmarks = []
    for name, factory in agents:
        try:
            bm = benchmark_agent(name, factory, n_episodes=3)
            benchmarks.append(bm)
        except Exception as e:
            print(f"\n[ERROR] Failed to benchmark {name}: {e}")
            traceback.print_exc()

    # Print results
    if benchmarks:
        print_summary(benchmarks)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

    return 0 if benchmarks else 1


if __name__ == "__main__":
    sys.exit(main())
