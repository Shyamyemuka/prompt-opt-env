"""
15 tasks across 4 categories with token budgets.
Token budgets: easy=80, medium=65, hard=55.
Tighter budgets for harder tasks — requires more concise language.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Task:
    task_id: int
    category: Literal["summarisation", "qa", "instruction", "code"]
    difficulty: Literal["easy", "medium", "hard"]
    task_description: str
    initial_bad_prompt: str
    reference_answer: str
    example_output: str
    context_sentence: str
    constraint_sentence: str
    token_budget: int  # hard ceiling on prompt word count


TASK_BANK: list[Task] = [
    # ── SUMMARISATION ────────────────────────────────────────────────────────
    Task(
        task_id=0, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Summarise a news article about climate change in 3 bullet points",
        initial_bad_prompt="talk about climate",
        reference_answer=(
            "- Global temperatures have risen 1.1°C since pre-industrial times.\n"
            "- Extreme weather events are becoming more frequent and severe.\n"
            "- Nations must cut emissions by 45% by 2030 to limit warming to 1.5°C."
        ),
        example_output="- Key fact in under 20 words.\n- Impact point.\n- Action needed.",
        context_sentence="Climate change refers to long-term shifts in global temperatures caused primarily by human activities since the 1800s.",
        constraint_sentence="Exactly 3 bullet points, each under 20 words.",
    ),
    Task(
        task_id=1, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Summarise the plot of Romeo and Juliet in exactly 2 sentences",
        initial_bad_prompt="tell me about romeo and juliet",
        reference_answer=(
            "Romeo and Juliet is a tragedy about two young lovers from feuding families in Verona who secretly marry. "
            "Their deaths by suicide ultimately reconcile their families, ending the feud."
        ),
        example_output="[First sentence: setup and conflict]. [Second sentence: resolution].",
        context_sentence="Romeo and Juliet is a Shakespeare tragedy set in Verona, Italy, about feuding families.",
        constraint_sentence="Exactly 2 sentences. No more, no fewer.",
    ),
    Task(
        task_id=2, category="summarisation", difficulty="medium", token_budget=65,
        task_description="Summarise the key risks of investing in cryptocurrency in under 60 words",
        initial_bad_prompt="crypto risks",
        reference_answer=(
            "Cryptocurrency investments carry extreme price volatility, potential total loss, regulatory uncertainty, "
            "security risks from hacks, and illiquidity. Unlike traditional assets, crypto is uninsured and largely "
            "unregulated, making it unsuitable for risk-averse investors."
        ),
        example_output="Crypto risks include: [risk 1], [risk 2], and [risk 3]. Note: [key warning].",
        context_sentence="Cryptocurrency is a digital currency secured by cryptography, with Bitcoin and Ethereum as major examples.",
        constraint_sentence="Under 60 words total.",
    ),
    Task(
        task_id=3, category="summarisation", difficulty="medium", token_budget=65,
        task_description="Summarise the French Revolution timeline in chronological bullet points",
        initial_bad_prompt="french revolution summary",
        reference_answer=(
            "- 1789: Bastille stormed; National Assembly formed.\n"
            "- 1791: Constitutional monarchy established.\n"
            "- 1792: Republic declared.\n"
            "- 1793–1794: Reign of Terror.\n"
            "- 1799: Napoleon seizes power."
        ),
        example_output="- [Year]: [Key event in one line].",
        context_sentence="The French Revolution (1789–1799) was a period of radical political transformation in France.",
        constraint_sentence="Bullet points with years. At least 5 events in chronological order.",
    ),
    Task(
        task_id=4, category="summarisation", difficulty="easy", token_budget=80,
        task_description="Explain what machine learning is to a 10-year-old",
        initial_bad_prompt="explain machine learning",
        reference_answer=(
            "Machine learning is when computers learn from examples, just like how you learned to recognise cats "
            "by seeing many cats. The computer looks at lots of data and figures out patterns by itself."
        ),
        example_output="Machine learning is like [simple child analogy]. The computer [what it does simply].",
        context_sentence="Machine learning is a type of AI where computers learn patterns from data without being explicitly programmed.",
        constraint_sentence="Simple words only. No jargon. Write for a 10-year-old.",
    ),
    # ── QA ───────────────────────────────────────────────────────────────────
    Task(
        task_id=5, category="qa", difficulty="medium", token_budget=65,
        task_description="Answer: What is the time complexity of binary search and why?",
        initial_bad_prompt="binary search complexity",
        reference_answer=(
            "Binary search has O(log n) time complexity. With each comparison it eliminates half the remaining elements. "
            "After k steps n/2^k = 1, so k = log₂(n) steps in the worst case."
        ),
        example_output="Binary search is O([notation]) because [explanation in 2 sentences].",
        context_sentence="Binary search finds a target in a sorted array by repeatedly halving the search space.",
        constraint_sentence="Include the Big O notation and explain why that complexity holds.",
    ),
    Task(
        task_id=6, category="qa", difficulty="medium", token_budget=65,
        task_description="Answer: What causes inflation and how does the central bank control it?",
        initial_bad_prompt="inflation",
        reference_answer=(
            "Inflation occurs when money supply grows faster than output or when demand exceeds supply. "
            "Central banks control it by raising interest rates, which reduces borrowing and slows price increases."
        ),
        example_output="Inflation is caused by [cause]. Central banks respond by [mechanism].",
        context_sentence="Inflation is the rate at which general price levels rise, eroding purchasing power over time.",
        constraint_sentence="Cover both causes and the central bank's primary control tool.",
    ),
    Task(
        task_id=7, category="qa", difficulty="easy", token_budget=80,
        task_description="Answer: What is the difference between RAM and ROM?",
        initial_bad_prompt="RAM ROM difference",
        reference_answer=(
            "RAM is volatile memory that temporarily stores data the computer is currently using and is erased when power is lost. "
            "ROM is non-volatile memory that permanently stores firmware and retains data without power."
        ),
        example_output="RAM is [description]. ROM is [description]. Key difference: [one sentence].",
        context_sentence="RAM and ROM are both types of computer memory serving different purposes in a system.",
        constraint_sentence="Define both clearly. State the single most important difference.",
    ),
    Task(
        task_id=8, category="qa", difficulty="easy", token_budget=80,
        task_description="Answer: Why does the sky appear blue during the day and red at sunset?",
        initial_bad_prompt="sky color why",
        reference_answer=(
            "Sunlight contains all colours. Earth's atmosphere scatters shorter blue wavelengths in all directions "
            "(Rayleigh scattering), making the sky appear blue. At sunset, sunlight travels through more atmosphere, "
            "scattering blue away and leaving longer red and orange wavelengths."
        ),
        example_output="Blue sky: [reason]. Red sunset: [reason].",
        context_sentence="This is explained by Rayleigh scattering, where atmospheric particles scatter light wavelengths differently.",
        constraint_sentence="Explain both daytime blue and sunset red in one coherent answer.",
    ),
    # ── INSTRUCTION ──────────────────────────────────────────────────────────
    Task(
        task_id=9, category="instruction", difficulty="easy", token_budget=80,
        task_description="Write step-by-step instructions to make a cup of tea",
        initial_bad_prompt="how to make tea",
        reference_answer=(
            "1. Boil water in a kettle.\n2. Place a tea bag in your cup.\n"
            "3. Pour hot water over the tea bag.\n4. Steep 3–5 minutes.\n"
            "5. Remove the tea bag.\n6. Add milk or sugar to taste."
        ),
        example_output="1. [Action verb]. 2. [Action verb]. (continue...)",
        context_sentence="Making tea involves boiling water and steeping a tea bag for the correct amount of time.",
        constraint_sentence="Numbered steps. Include timing. Start each step with an action verb.",
    ),
    Task(
        task_id=10, category="instruction", difficulty="medium", token_budget=65,
        task_description="Explain how to set up a Python virtual environment on Windows",
        initial_bad_prompt="python venv windows",
        reference_answer=(
            "1. Open Command Prompt.\n2. Navigate to project: cd path\\to\\project\n"
            "3. Create venv: python -m venv venv\n4. Activate: venv\\Scripts\\activate\n"
            "5. Install packages: pip install package\n6. Deactivate: deactivate"
        ),
        example_output="Step 1: Open [tool]. Step 2: Run `[command]`. (continue...)",
        context_sentence="A Python virtual environment isolates packages from your system Python installation.",
        constraint_sentence="Include exact commands in code format. Cover creation, activation, and deactivation.",
    ),
    Task(
        task_id=11, category="instruction", difficulty="hard", token_budget=55,
        task_description="Describe the steps to resolve a Git merge conflict",
        initial_bad_prompt="git merge conflict fix",
        reference_answer=(
            "1. Run git merge to trigger the conflict.\n"
            "2. Open conflicted file — Git marks sections with <<<<<<, =======, >>>>>>>.\n"
            "3. Edit file to keep correct code and remove markers.\n"
            "4. Stage resolved file: git add filename\n"
            "5. Commit: git commit\n6. Push: git push"
        ),
        example_output="1. [Trigger]. 2. [Find markers <<<<<<, =======, >>>>>>>]. 3. [Edit]. 4. [git add]. 5. [git commit].",
        context_sentence="A Git merge conflict occurs when two branches changed the same lines and Git cannot auto-resolve.",
        constraint_sentence="Include the conflict markers (<<<<<<, =======, >>>>>>>). Cover all steps through push.",
    ),
    # ── CODE ─────────────────────────────────────────────────────────────────
    Task(
        task_id=12, category="code", difficulty="medium", token_budget=65,
        task_description="Explain what a Python list comprehension does, with an example",
        initial_bad_prompt="list comprehension",
        reference_answer=(
            "A list comprehension creates a list concisely: [expression for item in iterable if condition]. "
            "Example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81."
        ),
        example_output="A list comprehension [definition]. Example: `[x**2 for x in range(10)]` produces [result].",
        context_sentence="List comprehensions are a Python feature offering a concise alternative to for loops for creating lists.",
        constraint_sentence="Include a runnable code example. Explain what the example produces.",
    ),
    Task(
        task_id=13, category="code", difficulty="medium", token_budget=65,
        task_description="Explain Big O notation using a simple code example",
        initial_bad_prompt="big o notation",
        reference_answer=(
            "Big O notation describes how runtime grows with input size. "
            "O(1) = constant (dict lookup). O(n) = linear (single loop). O(n²) = quadratic (nested loops)."
        ),
        example_output="Big O [definition]. Example: `[code]` is O([notation]) because [reason].",
        context_sentence="Big O notation describes the upper bound of an algorithm's time or space complexity.",
        constraint_sentence="Give 2+ examples with different complexities. Include code for each.",
    ),
    Task(
        task_id=14, category="code", difficulty="easy", token_budget=80,
        task_description="Explain what recursion is with a simple Python example",
        initial_bad_prompt="what is recursion",
        reference_answer=(
            "Recursion is when a function calls itself to solve a smaller version of the same problem. "
            "Example: def factorial(n): return 1 if n==0 else n*factorial(n-1). factorial(5) = 120."
        ),
        example_output="Recursion is [definition]. Example: `[code]` — base case: [base], recursive case: [recursive].",
        context_sentence="Recursion solves problems by breaking them into smaller instances of the same problem.",
        constraint_sentence="Show the base case and recursive case explicitly. Show sample input/output.",
    ),
]
