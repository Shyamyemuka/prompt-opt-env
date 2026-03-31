"""
Interactive Prompt Optimizer using PromptOptEnv RL.

Usage:
    python optimize.py

Features:
    - Interactive prompt input for ANY user prompt
    - Intelligent RL-based action selection using LLM
    - Before/after comparison with metrics
    - Token cost analysis and cost-aware optimization
    - Automatic task detection
"""
import os
import sys
import asyncio
from typing import Optional

from openai import OpenAI
from rouge_score import rouge_scorer

# Import the actual PromptOptEnv environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_opt_env import PromptOptEnvEnv, PromptAction
from prompt_opt_env.server.actions import (
    count_tokens, ACTION_NAMES,
    add_context, shorten, add_example, rephrase, add_constraint
)

# Environment setup
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))

if not HF_TOKEN:
    print("[X] Error: HF_TOKEN not set. Please set your HuggingFace token.")
    print("   export HF_TOKEN=hf_your_token_here")
    sys.exit(1)

_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call LLM with the prompt and return output."""
    try:
        r = _CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3, timeout=30,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"   [!] LLM call failed: {e}")
        return ""


def score_output(output: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    if not output or not reference:
        return 0.0
    return round(_ROUGE.score(reference, output)["rougeL"].fmeasure, 4)


def intelligent_action_selection(
    current_prompt: str,
    current_score: float,
    tokens_remaining: int,
    step: int,
    context: dict,
    task_type: str
) -> int:
    """
    Intelligent action selection using LLM-based policy.
    Returns action_id (0-5) based on current state.
    """
    # Build action selection prompt
    action_descriptions = """
Available actions:
0. ADD_CONTEXT - Add relevant context/input text (+10-15 tokens, +quality)
1. SHORTEN - Remove filler words like "please", "could you" (-5-12 tokens, neutral quality)
2. ADD_EXAMPLE - Add example output format (+12-20 tokens, +quality)
3. REPHRASE - Convert to direct imperative, remove politeness (0 tokens, +quality)
4. ADD_CONSTRAINT - Add format/output constraints (+8-12 tokens, +quality)
5. STOP - End optimization, accept current prompt (0 tokens, bonus reward)
"""

    selection_prompt = f"""You are an RL agent optimizing prompts for {task_type}.

Current prompt: "{current_prompt}"
Current quality score: {current_score:.2f}
Tokens remaining in budget: {tokens_remaining}
Step: {step}/7

{action_descriptions}

Analyze the current prompt and select the BEST action to take.
Consider:
- If prompt is vague/indirect → REPHRASE (action 3, free improvement)
- If prompt has filler words → SHORTEN (action 1, reduces tokens)
- If quality is good enough (>0.75) and tokens are low → STOP (action 5)
- If task needs context → ADD_CONTEXT (action 0)
- If output format is unclear → ADD_EXAMPLE (action 2) or ADD_CONSTRAINT (action 4)

Respond with ONLY the action number (0, 1, 2, 3, 4, or 5). No explanation."""

    try:
        response = call_llm(selection_prompt, max_tokens=10)
        # Extract action number
        import re
        match = re.search(r'\b([0-5])\b', response.strip())
        if match:
            return int(match.group(1))
    except Exception:
        pass

    # Fallback: use heuristics
    if current_score > 0.75 and tokens_remaining < 30:
        return 5  # STOP if good quality and low tokens
    if "please" in current_prompt.lower() or "could you" in current_prompt.lower():
        return 1  # SHORTEN if filler words present
    if current_prompt.endswith("?") or "can you" in current_prompt.lower():
        return 3  # REPHRASE if question form
    if tokens_remaining > 40 and step < 4:
        return 0  # ADD_CONTEXT if early and budget available
    if current_score > 0.70:
        return 5  # STOP if quality is good

    return 3  # Default to REPHRASE (safe, no token cost)


def print_header(text: str, char: str = "="):
    """Print a header line."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}")


def print_metric(label: str, before: str, after: str, improvement: Optional[str] = None):
    """Print a before/after metric."""
    print(f"\n  {label}:")
    print(f"    Before: {before}")
    print(f"    After:  {after}")
    if improvement:
        print(f"    → {improvement}")


class IntelligentPromptOptimizer:
    """RL-based prompt optimizer with intelligent action selection."""

    def __init__(self, task_type: str, initial_prompt: str, input_text: str = "",
                 context: str = "", question: str = "", reference: str = "",
                 example: str = "", constraint: str = "", token_budget: int = 80):
        self.task_type = task_type
        self.initial_prompt = initial_prompt
        self.input_text = input_text
        self.context = context
        self.question = question
        self.reference = reference
        self.example = example
        self.constraint = constraint
        self.token_budget = token_budget

        # Build context dict for actions
        self.context_dict = {
            "input_text": input_text,
            "context": context if context else input_text,
            "example": example,
            "constraint": constraint
        }

    def optimize(self, max_steps: int = 7) -> dict:
        """Run RL optimization with intelligent action selection."""

        print_header(f"TASK: {self.task_type.upper()}")

        # Show inputs
        if self.input_text:
            print(f"\n  Input Text:")
            preview = self.input_text[:150] + "..." if len(self.input_text) > 150 else self.input_text
            print(f'  "{preview}"')
        if self.context and self.context != self.input_text:
            print(f"\n  Context:")
            preview = self.context[:150] + "..." if len(self.context) > 150 else self.context
            print(f'  "{preview}"')
        if self.question:
            print(f"\n  Question: {self.question}")

        # Initial state
        current_prompt = self.initial_prompt
        current_output = call_llm(current_prompt)
        current_score = score_output(current_output, self.reference) if self.reference else self._heuristic_score(current_output)
        current_tokens = count_tokens(current_prompt)

        # Store initial values
        initial_prompt = current_prompt
        initial_output = current_output
        initial_score = current_score
        initial_tokens = current_tokens

        print_header(f"✅ SAMPLE 1: {self.task_type}", "-")

        print(f"\n  Initial Prompt:")
        print(f'  "{current_prompt}"')
        print(f"\n  Initial Output:")
        preview = current_output[:200] + "..." if len(current_output) > 200 else current_output
        print(f'  "{preview}"')
        print(f"\n  Initial Reward: {current_score:.2f}")
        print(f"  Token Count: {current_tokens}")

        print("\n  " + "-" * 50)
        print("  OPTIMIZATION PROCESS")
        print("  " + "-" * 50)

        # Optimization loop
        best_prompt = current_prompt
        best_output = current_output
        best_score = current_score
        best_tokens = current_tokens
        total_reward = 0.0
        step_rewards = []

        for step in range(max_steps):
            tokens_remaining = self.token_budget - current_tokens

            # Intelligent action selection
            action_id = intelligent_action_selection(
                current_prompt, current_score, tokens_remaining,
                step, self.context_dict, self.task_type
            )
            action_name = ACTION_NAMES.get(action_id, "UNKNOWN")

            # Handle STOP
            if action_id == 5:
                stop_bonus = round(best_score * 1.5, 4)
                total_reward += stop_bonus
                step_rewards.append(("STOP", stop_bonus))
                print(f"\n    Step {step+1}: [STOP] Agent decided to stop")
                print(f"             Reward: +{stop_bonus:.3f} (score × 1.5)")
                break

            # Apply action directly using individual functions
            new_prompt = current_prompt
            if action_id == 0:  # ADD_CONTEXT
                ctx = self.context_dict.get("input_text") or self.context_dict.get("context") or ""
                if ctx:
                    new_prompt = add_context(current_prompt, ctx[:100])  # Use first 100 chars
            elif action_id == 1:  # SHORTEN
                new_prompt = shorten(current_prompt)
            elif action_id == 2:  # ADD_EXAMPLE
                ex = self.context_dict.get("example")
                if ex:
                    new_prompt = add_example(current_prompt, ex[:100])
            elif action_id == 3:  # REPHRASE
                new_prompt = rephrase(current_prompt)
            elif action_id == 4:  # ADD_CONSTRAINT
                cons = self.context_dict.get("constraint")
                if cons:
                    new_prompt = add_constraint(current_prompt, cons)

            # Check for no-op
            if new_prompt == current_prompt:
                step_rewards.append((action_name, -0.1))
                total_reward -= 0.1
                print(f"    Step {step+1}: [{action_name}] No effect (no-op penalty: -0.1)")
                continue

            new_tokens = count_tokens(new_prompt)

            # Check budget
            if new_tokens > self.token_budget:
                penalty = -0.5
                total_reward += penalty
                step_rewards.append((action_name, penalty))
                print(f"    Step {step+1}: [{action_name}] Budget exceeded! Penalty: -0.5")
                break

            # Get new output and score
            new_output = call_llm(new_prompt)
            new_score = score_output(new_output, self.reference) if self.reference else self._heuristic_score(new_output)

            # Compute reward
            quality_delta = new_score - current_score
            token_overhead = new_tokens - current_tokens
            step_reward = quality_delta - ALPHA * token_overhead
            step_reward = round(max(-2.0, min(2.0, step_reward)), 4)

            total_reward += step_reward
            step_rewards.append((action_name, step_reward))

            # Print step
            action_icons = {
                0: "[CTX]", 1: "[CUT]", 2: "[EX]",
                3: "[REP]", 4: "[CON]", 5: "[STOP]"
            }
            icon = action_icons.get(action_id, "[?]")
            print(f"    Step {step+1}: {icon} {action_name:15} | "
                  f"Score: {new_score:.2f} ({quality_delta:+.2f}) | "
                  f"Tokens: {new_tokens} ({token_overhead:+d}) | "
                  f"Reward: {step_reward:+.3f}")

            # Update best if improved
            if new_score > best_score or (new_score >= best_score - 0.05 and new_tokens < best_tokens):
                best_prompt = new_prompt
                best_output = new_output
                best_score = new_score
                best_tokens = new_tokens

            # Update current state
            current_prompt = new_prompt
            current_output = new_output
            current_score = new_score
            current_tokens = new_tokens

            # Check success threshold
            if best_score > 0.85:
                print(f"\n    [OK] Success threshold reached (score > 0.85)")
                break

        # Final results section
        print("\n  " + "-" * 50)
        print("  FINAL RESULT")
        print("  " + "-" * 50)

        print(f"\n  Optimized Prompt:")
        print(f'  "{best_prompt}"')
        print(f"\n  Optimized Output:")
        preview = best_output[:250] + "..." if len(best_output) > 250 else best_output
        print(f'  "{preview}"')
        print(f"\n  Final Reward: {best_score:.2f}")
        print(f"  Token Count: {best_tokens}")

        # Calculate metrics
        score_improvement = ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        token_change = best_tokens - initial_tokens
        token_reduction_pct = (abs(token_change) / initial_tokens * 100) if initial_tokens > 0 and token_change < 0 else 0

        # Cost-aware net reward gain
        # Quality improvement value minus token cost
        quality_value = (best_score - initial_score) * 100  # Scale to percentage
        token_cost = token_change * ALPHA * 100 if token_change > 0 else 0
        net_reward_gain = quality_value - token_cost

        print("\n  " + "=" * 50)
        print("  METRICS SUMMARY")
        print("  " + "=" * 50)

        if token_change < 0:
            print(f"\n  Improvement: +{score_improvement:.0f}%")
            print(f"  Token Reduction: {token_reduction_pct:.0f}% (saved {abs(token_change)} tokens)")
        elif token_change > 0:
            print(f"\n  Improvement: +{score_improvement:.0f}%")
            print(f"  Token Increase: +{token_change} tokens")
        else:
            print(f"\n  Improvement: +{score_improvement:.0f}%")
            print(f"  Token count unchanged")

        print(f"  Total Reward: {total_reward:.3f}")

        # Show cost-aware insight if relevant
        if token_change < -5 and score_improvement > -5:
            print(f"\n  👉 This shows cost-aware trade-off intelligence!")
            print(f"     Net Reward Gain (cost-aware): +{net_reward_gain:.0f}%")
            print(f"     Maintained quality while reducing tokens.")

        return {
            "initial_prompt": initial_prompt,
            "initial_output": initial_output,
            "initial_score": initial_score,
            "initial_tokens": initial_tokens,
            "final_prompt": best_prompt,
            "final_output": best_output,
            "final_score": best_score,
            "final_tokens": best_tokens,
            "improvement_pct": score_improvement,
            "token_change": token_change,
            "token_reduction_pct": token_reduction_pct,
            "total_reward": total_reward,
            "net_reward_gain": net_reward_gain,
            "steps": len(step_rewards),
            "step_rewards": step_rewards
        }

    def _heuristic_score(self, output: str) -> float:
        """Fallback scoring when no reference."""
        if not output:
            return 0.0
        score = 0.3
        if len(output) > 50:
            score += 0.1
        if len(output) > 100:
            score += 0.1
        if "." in output:
            score += 0.1
        if output.count(".") > 1:
            score += 0.1
        if len(output) > 200:
            score += 0.1
        return round(min(score, 0.8), 2)  # Cap at 0.8 without reference


def detect_task_type(prompt: str, input_text: str) -> str:
    """Auto-detect task type from prompt and input."""
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["summarize", "summary", "summarise", "tldr"]):
        return "Summarization"
    if any(word in prompt_lower for word in ["question", "answer", "qa", "what is", "why", "how"]):
        return "Question Answering"
    if any(word in prompt_lower for word in ["code", "explain this code", "function", "python", "java", "javascript"]):
        return "Code Explanation"
    if any(word in prompt_lower for word in ["steps", "how to", "instruction", "guide", "tutorial"]):
        return "Instruction Following"

    # Check input text
    if input_text:
        input_lower = input_text.lower()
        if "def " in input_lower or "class " in input_lower or "import " in input_lower:
            return "Code Explanation"
        if len(input_text) > 200:
            return "Summarization"

    return "General"


def interactive_mode():
    """Interactive CLI for prompt optimization."""
    print("\n" + ">>>" * 23)
    print("   PROMPT OPTIMIZER - Interactive RL Prompt Enhancement")
    print("   Uses Cost-Aware RL: Quality Improvement - Token Cost")
    print(">>>" * 23)

    print("\n[i] This optimizer uses Reinforcement Learning to:")
    print("    - Intelligently select prompt improvements")
    print("    - Balance quality gains against token costs")
    print("    - Learn when to STOP (voluntary episode end)")

    print("\n[i] Available Task Types:")
    print("   1. Summarization")
    print("   2. Question Answering (QA)")
    print("   3. Code Explanation")
    print("   4. Instruction Following")
    print("   5. Custom (auto-detect)")

    choice = input("\n> Select task type (1-5): ").strip()

    task_types = {
        "1": "Summarization",
        "2": "Question Answering",
        "3": "Code Explanation",
        "4": "Instruction Following",
        "5": "Custom"
    }

    task_type = task_types.get(choice, "Custom")
    print(f"\n[i] Selected: {task_type}")

    # Gather inputs
    input_text = ""
    context = ""
    question = ""
    reference = ""
    example = ""
    constraint = ""

    print(f"\n[!] Enter the content/text to work with (optional).")
    print(f"    Paste text below (press Enter twice when done):")
    lines = []
    blank_count = 0
    while True:
        line = input()
        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
        else:
            blank_count = 0
            lines.append(line)
    input_text = "\n".join(lines)

    if input_text:
        print(f"[i] Received {len(input_text)} characters")

    if task_type == "Question Answering":
        question = input("\n[?] Enter the specific question: ").strip()

    # Initial prompt
    print("\n[!] Enter your initial prompt (e.g., 'Summarize this text.' or 'Answer the question.')")
    print("    This is what the optimizer will improve:")
    initial_prompt = input("> ").strip()

    if not initial_prompt:
        print("[X] Error: Initial prompt is required.")
        return

    # Auto-detect if custom
    if task_type == "Custom":
        task_type = detect_task_type(initial_prompt, input_text)
        print(f"\n[i] Auto-detected task type: {task_type}")

    # Optional reference for scoring
    print("\n[i] For accurate scoring, provide a reference answer (optional):")
    reference = input("    Reference answer: ").strip()

    # Optional example and constraint
    print("\n[i] Optional: Provide an example of desired output format:")
    example = input("    Example: ").strip()

    print("\n[i] Optional: Add constraints (e.g., 'in 2 sentences', 'under 50 words'):")
    constraint = input("    Constraint: ").strip()

    # Determine token budget based on task
    token_budget = 80  # default easy
    if task_type in ["Question Answering", "Code Explanation"]:
        token_budget = 65  # medium

    # Run optimization
    print("\n" + "..." * 23)
    print("   Running RL optimization with intelligent action selection...")
    print("   " + "..." * 23)

    optimizer = IntelligentPromptOptimizer(
        task_type=task_type,
        initial_prompt=initial_prompt,
        input_text=input_text,
        context=input_text,  # Use input as context too
        question=question,
        reference=reference,
        example=example,
        constraint=constraint,
        token_budget=token_budget
    )

    results = optimizer.optimize(max_steps=7)

    # Ask to save
    print("\n")
    save = input("[?] Save results to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Filename (default: optimization_result.txt): ").strip() or "optimization_result.txt"
        with open(filename, 'w') as f:
            f.write(f"Task: {task_type}\n\n")
            f.write(f"Initial Prompt: {results['initial_prompt']}\n")
            f.write(f"Initial Output: {results['initial_output']}\n")
            f.write(f"Initial Score: {results['initial_score']:.2f}\n")
            f.write(f"Initial Tokens: {results['initial_tokens']}\n\n")
            f.write(f"Optimized Prompt: {results['final_prompt']}\n")
            f.write(f"Optimized Output: {results['final_output']}\n")
            f.write(f"Final Score: {results['final_score']:.2f}\n")
            f.write(f"Final Tokens: {results['final_tokens']}\n")
            f.write(f"Improvement: {results['improvement_pct']:.0f}%\n")
            f.write(f"Token Change: {results['token_change']:+d}\n")
            f.write(f"Total Reward: {results['total_reward']:.3f}\n")
        print(f"[OK] Saved to {filename}")

    print("\n" + "***" * 23)
    print("   Optimization Complete!")
    print("***" * 23)


def demo_mode():
    """Run predefined demo examples showing cost-aware optimization."""

    # Demo 1: Summarization with quality improvement
    print("\n" + "=" * 70)
    print("  DEMO MODE: Showing Cost-Aware RL in Action")
    print("=" * 70)

    opt1 = IntelligentPromptOptimizer(
        task_type="Summarization",
        initial_prompt="Summarize this text.",
        input_text="Artificial intelligence is transforming industries by enabling automation, improving decision-making, and creating new opportunities across sectors.",
        reference="AI is transforming industries through automation, enhanced decision-making, and new opportunities.",
        example="AI impacts industries via automation, better decisions, opportunities.",
        constraint="Keep it concise, under 20 words.",
        token_budget=80
    )
    opt1.optimize(max_steps=7)

    input("\n> Press Enter for Demo 2 (Cost-aware trade-off)...")

    # Demo 2: Cost-aware trade-off (token reduction focus)
    opt2 = IntelligentPromptOptimizer(
        task_type="Summarization",
        initial_prompt="Please could you summarize the following paragraph in detail with all possible points included comprehensively.",
        input_text="Machine learning enables computers to learn from data patterns and improve performance without explicit programming.",
        reference="Machine learning allows computers to learn from data and improve without explicit programming.",
        constraint="Be concise. Remove filler words.",
        token_budget=80
    )
    results2 = opt2.optimize(max_steps=7)

    # Highlight cost-aware aspect
    if results2.get('token_reduction_pct', 0) > 20:
        print("\n  👉 This is GOLD - demonstrates cost-aware trade-off intelligence!")
        print("     The agent learned that maintaining quality while reducing tokens")
        print("     produces better net reward (quality per token).")

    input("\n> Press Enter for Demo 3 (QA with context)...")

    # Demo 3: QA with context
    opt3 = IntelligentPromptOptimizer(
        task_type="Question Answering",
        initial_prompt="Answer the question.",
        context="Photosynthesis is the process by which plants convert sunlight into energy through chlorophyll in their leaves.",
        question="What is photosynthesis?",
        reference="Photosynthesis is the process by which plants convert sunlight into energy.",
        example="It is the process where [subject] [action] to [result].",
        constraint="Provide a complete sentence with clarity.",
        token_budget=65
    )
    opt3.optimize(max_steps=7)


def main():
    print("\n" + ">>>" * 23)
    print("   Welcome to PromptOptEnv Interactive Optimizer")
    print("   Cost-Aware RL for Prompt Optimization")
    print(">>>" * 23)

    print("\n[i] This tool uses Reinforcement Learning where:")
    print("    - Actions have token costs (ADD_CONTEXT = +10-15 tokens)")
    print("    - Reward = Quality_Improvement - Token_Cost_Penalty")
    print("    - Agent learns to STOP when quality/cost ratio is optimal")

    print("\n[i] Modes:")
    print("   1. Interactive Mode - Enter your own prompts")
    print("   2. Demo Mode - See predefined examples")

    choice = input("\n> Select mode (1-2): ").strip()

    if choice == "2":
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
