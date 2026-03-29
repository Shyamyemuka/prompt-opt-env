"""
Complete task bank. 15 tasks across 4 categories.

Each task has:
  task_id, category, task_description, initial_bad_prompt,
  reference_answer, example_output, context_sentence,
  constraint_sentence, difficulty.

These fields are used by the action functions and the grader.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Task:
    task_id: int
    category: Literal["summarisation", "qa", "instruction", "code"]
    task_description: str
    initial_bad_prompt: str       # Deliberately vague starting prompt
    reference_answer: str         # Gold answer for ROUGE scoring
    example_output: str           # Used by ADD_EXAMPLE action
    context_sentence: str         # Used by ADD_CONTEXT action
    constraint_sentence: str      # Used by ADD_CONSTRAINT action
    difficulty: Literal["easy", "medium", "hard"]


TASK_BANK: list[Task] = [
    Task(
        task_id=0,
        category="summarisation",
        task_description="Summarise a 200-word news article about climate change in 3 bullet points",
        initial_bad_prompt="talk about climate",
        reference_answer=(
            "- Global temperatures have risen 1.1°C since pre-industrial times.\n"
            "- Extreme weather events are becoming more frequent and severe.\n"
            "- Nations must cut emissions by 45% by 2030 to limit warming to 1.5°C."
        ),
        example_output=(
            "- Point one about key fact.\n"
            "- Point two about impact.\n"
            "- Point three about action needed."
        ),
        context_sentence=(
            "Climate change refers to long-term shifts in global temperatures and weather patterns, "
            "primarily caused by human activities since the 1800s."
        ),
        constraint_sentence="Respond with exactly 3 bullet points, each under 20 words.",
        difficulty="easy",
    ),
    Task(
        task_id=1,
        category="summarisation",
        task_description="Summarise the plot of Romeo and Juliet in exactly 2 sentences",
        initial_bad_prompt="tell me about romeo and juliet",
        reference_answer=(
            "Romeo and Juliet is a tragedy about two young lovers from feuding families in Verona "
            "who secretly marry. Their deaths by suicide ultimately reconcile their families, ending the feud."
        ),
        example_output="[First sentence: setup and conflict]. [Second sentence: resolution and outcome].",
        context_sentence=(
            "Romeo and Juliet is a play by William Shakespeare written around 1594–1596, set in Verona, Italy."
        ),
        constraint_sentence="Your answer must be exactly 2 sentences. No more, no fewer.",
        difficulty="easy",
    ),
    Task(
        task_id=2,
        category="summarisation",
        task_description="Summarise the key risks of investing in cryptocurrency in under 60 words",
        initial_bad_prompt="crypto risks",
        reference_answer=(
            "Cryptocurrency investments carry extreme price volatility, potential for total loss, "
            "regulatory uncertainty across jurisdictions, security risks from hacks and scams, and "
            "illiquidity in smaller markets. Unlike traditional assets, crypto is uninsured and "
            "unregulated, making it unsuitable for risk-averse investors."
        ),
        example_output="Crypto risks include: [risk 1], [risk 2], and [risk 3]. Investors should note [key warning].",
        context_sentence=(
            "Cryptocurrency is a digital or virtual currency secured by cryptography, "
            "with Bitcoin and Ethereum being the most prominent examples."
        ),
        constraint_sentence="Your response must be under 60 words total.",
        difficulty="medium",
    ),
    Task(
        task_id=3,
        category="summarisation",
        task_description="Summarise the French Revolution timeline in chronological bullet points",
        initial_bad_prompt="french revolution summary",
        reference_answer=(
            "- 1789: Estates-General convened; Third Estate forms National Assembly; Bastille stormed.\n"
            "- 1791: Constitutional monarchy established.\n"
            "- 1792: War declared on Austria; First French Republic proclaimed.\n"
            "- 1793-1794: Reign of Terror under Robespierre.\n"
            "- 1799: Napoleon Bonaparte seizes power in coup."
        ),
        example_output="- [Year]: [Key event description].",
        context_sentence=(
            "The French Revolution (1789–1799) was a period of radical political and social "
            "transformation in France."
        ),
        constraint_sentence="Use bullet points with years. List at least 5 key events chronologically.",
        difficulty="medium",
    ),
    Task(
        task_id=4,
        category="summarisation",
        task_description="Summarise what machine learning is for a 10-year-old",
        initial_bad_prompt="explain machine learning",
        reference_answer=(
            "Machine learning is when computers learn from examples, just like how you learned to "
            "recognise cats by seeing many cats. Instead of being told exact rules, the computer "
            "looks at lots of data and figures out the patterns by itself."
        ),
        example_output=(
            "Machine learning is like [simple analogy a child understands]. "
            "The computer [simple description of what it does]."
        ),
        context_sentence=(
            "Machine learning is a type of artificial intelligence where computers learn patterns "
            "from data without being explicitly programmed for each task."
        ),
        constraint_sentence="Use simple words. No technical jargon. Write as if explaining to a 10-year-old.",
        difficulty="easy",
    ),
    Task(
        task_id=5,
        category="qa",
        task_description="Answer: What is the time complexity of binary search and why?",
        initial_bad_prompt="binary search complexity",
        reference_answer=(
            "Binary search has O(log n) time complexity. With each comparison, it eliminates half "
            "the remaining elements. Starting with n elements: after 1 step n/2, after 2 steps n/4, "
            "after k steps n/2^k = 1, so k = log₂(n) steps in the worst case."
        ),
        example_output="Binary search is O([complexity]) because [explanation of why].",
        context_sentence=(
            "Binary search is a search algorithm that finds the position of a target value "
            "within a sorted array."
        ),
        constraint_sentence=(
            "Include the Big O notation and a brief explanation of why that complexity is correct."
        ),
        difficulty="medium",
    ),
    Task(
        task_id=6,
        category="qa",
        task_description="Answer: What causes inflation and how does the central bank control it?",
        initial_bad_prompt="inflation",
        reference_answer=(
            "Inflation occurs when the money supply grows faster than economic output, when demand "
            "exceeds supply (demand-pull), or when production costs rise (cost-push). Central banks "
            "control it primarily by raising interest rates, which reduces borrowing and spending, "
            "slowing down price increases."
        ),
        example_output=(
            "Inflation is caused by [cause 1] and [cause 2]. Central banks respond by [mechanism]."
        ),
        context_sentence=(
            "Inflation is the rate at which the general level of prices for goods and services rises, "
            "eroding purchasing power."
        ),
        constraint_sentence=(
            "Cover both the causes of inflation and the central bank's primary tool for controlling it."
        ),
        difficulty="medium",
    ),
    Task(
        task_id=7,
        category="qa",
        task_description="Answer: What is the difference between RAM and ROM?",
        initial_bad_prompt="RAM ROM difference",
        reference_answer=(
            "RAM (Random Access Memory) is volatile memory that temporarily stores data the computer "
            "is currently using; it is erased when power is lost. ROM (Read-Only Memory) is non-volatile "
            "memory that permanently stores firmware and boot instructions; it retains data without power."
        ),
        example_output="RAM is [description]. ROM is [description]. The key difference is [key difference].",
        context_sentence=(
            "RAM and ROM are both types of computer memory that serve different purposes in a computer system."
        ),
        constraint_sentence=(
            "Define both RAM and ROM clearly, then state the single most important difference between them."
        ),
        difficulty="easy",
    ),
    Task(
        task_id=8,
        category="qa",
        task_description="Answer: Why does the sky appear blue during the day and red at sunset?",
        initial_bad_prompt="sky color why",
        reference_answer=(
            "Sunlight contains all colours. Earth's atmosphere scatters shorter blue wavelengths in all "
            "directions (Rayleigh scattering), making the sky appear blue. At sunset, sunlight travels "
            "through more atmosphere, scattering away most blue light and leaving longer red and orange "
            "wavelengths visible."
        ),
        example_output=(
            "During the day the sky looks blue because [reason]. At sunset it turns red because [reason]."
        ),
        context_sentence=(
            "This phenomenon is explained by Rayleigh scattering, where atmospheric particles scatter "
            "different wavelengths of light differently."
        ),
        constraint_sentence=(
            "Explain both the daytime blue and the sunset red in a single coherent answer."
        ),
        difficulty="easy",
    ),
    Task(
        task_id=9,
        category="instruction",
        task_description="Write step-by-step instructions to make a cup of tea",
        initial_bad_prompt="how to make tea",
        reference_answer=(
            "1. Boil water in a kettle.\n"
            "2. Place a tea bag in your cup.\n"
            "3. Pour the hot water over the tea bag.\n"
            "4. Wait 3–5 minutes for the tea to steep.\n"
            "5. Remove the tea bag.\n"
            "6. Add milk or sugar to taste.\n"
            "7. Stir and enjoy."
        ),
        example_output="1. [First step]. 2. [Second step]. 3. [Continue...]",
        context_sentence=(
            "Making tea is a simple process involving boiling water and steeping tea leaves or a tea bag."
        ),
        constraint_sentence=(
            "Write numbered steps. Include specific timing (how long to steep). "
            "Start each step with an action verb."
        ),
        difficulty="easy",
    ),
    Task(
        task_id=10,
        category="instruction",
        task_description="Explain how to set up a Python virtual environment on Windows",
        initial_bad_prompt="python venv windows",
        reference_answer=(
            "1. Open Command Prompt.\n"
            "2. Navigate to your project folder: cd path\\to\\project\n"
            "3. Create the virtual environment: python -m venv venv\n"
            "4. Activate it: venv\\Scripts\\activate\n"
            "5. Install packages: pip install package_name\n"
            "6. Deactivate when done: deactivate"
        ),
        example_output="Step 1: Open [tool]. Step 2: Run `[command]`. Step 3: [Continue...]",
        context_sentence=(
            "A Python virtual environment is an isolated directory containing a specific Python version "
            "and installed packages, separate from your system Python."
        ),
        constraint_sentence=(
            "Include the exact commands to type. Use code formatting for commands. "
            "Cover creation, activation, and deactivation."
        ),
        difficulty="medium",
    ),
    Task(
        task_id=11,
        category="instruction",
        task_description="Describe the steps to resolve a Git merge conflict",
        initial_bad_prompt="git merge conflict fix",
        reference_answer=(
            "1. Run git merge or git pull to trigger the conflict.\n"
            "2. Open the conflicted file — Git marks conflicts with <<<<<<, =======, and >>>>>>>.\n"
            "3. Edit the file to keep the correct code, removing the conflict markers.\n"
            "4. Stage the resolved file: git add filename\n"
            "5. Commit the merge: git commit\n"
            "6. Push if needed: git push"
        ),
        example_output=(
            "1. [Trigger conflict]. 2. [Identify conflict markers]. "
            "3. [Resolve]. 4. [Stage]. 5. [Commit]."
        ),
        context_sentence=(
            "A Git merge conflict occurs when two branches have made different changes to the same "
            "part of the same file and Git cannot automatically determine which version to use."
        ),
        constraint_sentence=(
            "Include what the conflict markers look like (<<<<<<, =======, >>>>>>>). "
            "Cover all steps from conflict to resolution."
        ),
        difficulty="hard",
    ),
    Task(
        task_id=12,
        category="code",
        task_description="Explain what a Python list comprehension does, with an example",
        initial_bad_prompt="list comprehension",
        reference_answer=(
            "A list comprehension is a concise way to create a list in Python. Instead of a for loop, "
            "you write the logic inline: [expression for item in iterable if condition]. "
            "Example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81."
        ),
        example_output=(
            "A list comprehension [definition]. Example: `[code example]` which [what it produces]."
        ),
        context_sentence=(
            "List comprehensions are a Python feature inspired by set-builder notation in mathematics, "
            "offering a more readable alternative to for loops for creating lists."
        ),
        constraint_sentence=(
            "Include a concrete, runnable code example. Explain what the example produces."
        ),
        difficulty="medium",
    ),
    Task(
        task_id=13,
        category="code",
        task_description="Explain Big O notation using a simple code example",
        initial_bad_prompt="big o notation",
        reference_answer=(
            "Big O notation describes how an algorithm's runtime grows relative to input size. "
            "O(1) is constant time (e.g. dict lookup). O(n) means runtime grows linearly "
            "(e.g. a single for loop over n items). O(n²) means nested loops — runtime grows quadratically."
        ),
        example_output="Big O notation [definition]. For example: `[code]` is O([complexity]) because [reason].",
        context_sentence=(
            "Big O notation is a mathematical notation used in computer science to describe the upper "
            "bound of an algorithm's time or space complexity."
        ),
        constraint_sentence=(
            "Give at least 2 examples with different complexities. Include code snippets for each."
        ),
        difficulty="medium",
    ),
    Task(
        task_id=14,
        category="code",
        task_description="Explain what recursion is with a simple Python example",
        initial_bad_prompt="what is recursion",
        reference_answer=(
            "Recursion is when a function calls itself to solve a smaller version of the same problem. "
            "It needs a base case (stopping condition) and a recursive case. "
            "Example: def factorial(n): return 1 if n == 0 else n * factorial(n-1). "
            "factorial(5) = 5×4×3×2×1 = 120."
        ),
        example_output=(
            "Recursion is [definition]. Example: `[code]` works by [explanation of recursive steps]."
        ),
        context_sentence=(
            "Recursion is a programming technique where a function solves a problem by breaking it "
            "into smaller instances of the same problem."
        ),
        constraint_sentence=(
            "Include a working Python code example. Show the base case and recursive case explicitly. "
            "Show a sample input/output."
        ),
        difficulty="easy",
    ),
]
