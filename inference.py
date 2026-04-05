"""
Cost-aware baseline inference script.

MANDATORY:
  - Name: inference.py (do not rename)
  - Location: repo root (not inside prompt_opt_env/)
  - Uses: openai.OpenAI client for all LLM calls
  - Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from os.environ
  - Covers: 3 tasks (easy, medium, hard), 7 steps each
  - Prints: per-step score, tokens, reward + summary with efficiency metric
  - Runtime: under 20 minutes on 2 vCPU / 8 GB RAM
  - Exit: always code 0

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1/
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export HF_TOKEN=hf_your_token
    python inference.py
"""
import os
import sys
import re
import random

from openai import OpenAI
from rouge_score import rouge_scorer

# Load .env.local variables explicitly so `python inference.py` works out of the box
try:
    env_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
    if os.path.exists(env_local):
        with open(env_local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k not in os.environ:
                        os.environ[k] = v.strip("'\"")
except Exception:
    pass

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")
ALPHA: float      = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not found!")
    print("Please set your HuggingFace token before running. Examples:")
    print('  PowerShell: $env:HF_TOKEN="hf_your_token"')
    print('  Mac/Linux : export HF_TOKEN="hf_your_token"')
    print('Or load it from your .env.local file: uv run --env-file .env.local inference.py')
    sys.exit(1)

_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
_ROUGE  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def count_tokens(text: str) -> int:
    return len(text.split())


EVAL_TASKS = [
    {
        "difficulty": "easy", "token_budget": 80,
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
        "difficulty": "medium", "token_budget": 65,
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
        "difficulty": "hard", "token_budget": 55,
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

FILLER = [r"\bplease\b", r"\bcould you\b", r"\bcan you\b", r"\bi want you to\b"]


def apply(action_id: int, prompt: str, task: dict) -> str:
    if action_id == 0:
        if task["context"][:20].lower() in prompt.lower():
            return prompt
        return f"{prompt}\nContext: {task['context']}"
    elif action_id == 1:
        r = prompt
        for f in FILLER:
            r = re.sub(f, "", r, flags=re.IGNORECASE)
        r = re.sub(r"  +", " ", r).strip()
        return r if r != prompt else prompt
    elif action_id == 2:
        if "Example output format:" in prompt:
            return prompt
        return f"{prompt}\nExample output format: {task['example']}"
    elif action_id == 3:
        r = re.sub(r"^(?:can you|could you|please)\s+(.+?)[\?\.]*$",
                   lambda m: m.group(1).strip().capitalize() + ".",
                   prompt, flags=re.IGNORECASE | re.MULTILINE)
        return r if r != prompt else prompt
    elif action_id == 4:
        if "Requirement:" in prompt:
            return prompt
        return f"{prompt}\nRequirement: {task['constraint']}"
    return prompt


def call_llm(prompt: str) -> str:
    try:
        r = _CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.1, timeout=30,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return ""


def rouge_l(hyp: str, ref: str) -> float:
    if not hyp:
        return 0.0
    return round(_ROUGE.score(ref, hyp)["rougeL"].fmeasure, 4)


def run_episode_with_display(task: dict, max_steps: int = 10) -> dict:
    """Run episode with before/after display format."""
    prompt = task["initial_prompt"]
    init_out = call_llm(prompt)
    init_score = rouge_l(init_out, task["reference"])
    init_tokens = count_tokens(prompt)

    print(f"\n{'='*60}")
    print(f"Task: {task['task_description']}")
    print(f"{'='*60}")

    print(f"\nInitial Prompt:")
    print(f'"{prompt}"')
    print(f"\nInitial Output:")
    print(f'"{init_out[:150]}{"..." if len(init_out) > 150 else ""}"')
    print(f"\nInitial Score: {init_score:.2f}")
    print(f"Initial Tokens: {init_tokens}")

    current_score = init_score
    current_tokens = init_tokens
    total_reward = 0.0
    steps = 0
    final_prompt = prompt
    final_output = init_out

    for step in range(max_steps):
        action_id = random.randint(0, 5)

        if action_id == 5:
            stop_bonus = round(current_score * 1.5, 4)
            total_reward += stop_bonus
            steps += 1
            break

        new_prompt = apply(action_id, prompt, task)
        if new_prompt == prompt:
            total_reward += -0.1
            steps += 1
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > task["token_budget"]:
            total_reward += -0.5
            steps += 1
            break

        new_out = call_llm(new_prompt)
        new_score = rouge_l(new_out, task["reference"])
        overhead = new_tokens - current_tokens
        reward = round(new_score - current_score - ALPHA * overhead, 4)
        total_reward += reward
        steps += 1

        prompt = new_prompt
        current_score = new_score
        current_tokens = new_tokens
        final_prompt = new_prompt
        final_output = new_out

        if current_score > 0.85:
            total_reward += 1.0
            break

    print(f"\n{'-'*60}")
    print(f"After Optimization ({steps} steps):")
    print(f"{'-'*60}")
    print(f"\nOptimized Prompt:")
    print(f'"{final_prompt}"')
    print(f"\nOptimized Output:")
    print(f'"{final_output[:150]}{"..." if len(final_output) > 150 else ""}"')
    print(f"\nFinal Score: {current_score:.2f}")
    print(f"Final Tokens: {current_tokens}")
    print(f"Total Reward: {total_reward:.2f}")

    token_change = current_tokens - init_tokens
    if token_change < 0:
        reduction_pct = abs(token_change) / init_tokens * 100 if init_tokens > 0 else 0
        print(f"Token cost REDUCED by {reduction_pct:.0f}% (saved {abs(token_change)} tokens)")
    elif token_change > 0:
        print(f"Token cost increased by {token_change} tokens")
    else:
        print("Token cost unchanged")

    return {
        "difficulty": task["difficulty"],
        "initial_score": init_score,
        "final_score": current_score,
        "total_reward": round(total_reward, 4),
        "final_token_count": current_tokens,
        "token_budget": task["token_budget"],
        "steps": steps,
    }


def main() -> None:
    print("=" * 66)
    print("PromptOptEnv — Cost-Aware Baseline Inference Script")
    print(f"Model    : {MODEL_NAME}")
    print(f"Endpoint : {API_BASE_URL}")
    print(f"Alpha    : {ALPHA}")
    print("=" * 66)

    results = []
    for task in EVAL_TASKS:
        results.append(run_episode_with_display(task))

    print()
    print("=" * 66)
    print("BASELINE SCORES SUMMARY")
    print("=" * 66)
    print(f"{'Diff':<10} {'Init':>8} {'Final':>8} {'Tokens':>8} {'Budget':>8} {'Reward':>10} {'Steps':>6}")
    print("-" * 66)
    for r in results:
        print(
            f"{r['difficulty']:<10} {r['initial_score']:>8.2f} {r['final_score']:>8.2f} "
            f"{r['final_token_count']:>8} {r['token_budget']:>8} {r['total_reward']:>10.2f} {r['steps']:>6}"
        )
    print("-" * 66)
    avg_init = round(sum(r["initial_score"] for r in results) / len(results), 4)
    avg_final = round(sum(r["final_score"] for r in results) / len(results), 4)
    print(f"{'Average':<10} {avg_init:>8.2f} {avg_final:>8.2f}")
    print()
    print("All scores are ROUGE-L F1 in [0.0, 1.0]. Script complete.")


if __name__ == "__main__":
    main()
