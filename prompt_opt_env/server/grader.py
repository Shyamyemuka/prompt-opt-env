"""
Grading logic for PromptRL.
ALL LLM calls use the OpenAI Python client — required by hackathon rules.
Never use httpx or requests for LLM calls.

Modes:
  'rouge'         — uses DUMMY_OUTPUTS[task_id], no API, always works
  'openai_client' — calls LLM via OpenAI client, falls back to dummy on error
"""
import os
import sys
from openai import OpenAI
from rouge_score import rouge_scorer


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Pre-written canned outputs per task_id for ROUGE mode.
# Quality is intentionally moderate — represents what a model
# would return for the initial bad prompt, giving room for improvement.
DUMMY_OUTPUTS: dict[int, str] = {
    0: (
        "- Global temperatures are rising due to human emissions.\n"
        "- Extreme weather is becoming more common.\n"
        "- Countries need to cut emissions urgently."
    ),
    1: (
        "Romeo and Juliet are young lovers from rival families who secretly marry. "
        "Their tragic deaths reconcile their feuding families."
    ),
    2: (
        "Crypto risks include extreme volatility, potential total loss, "
        "regulatory uncertainty, and security vulnerabilities."
    ),
    3: (
        "- 1789: Bastille stormed.\n- 1791: Constitutional monarchy.\n"
        "- 1792: Republic declared.\n- 1793: Reign of Terror.\n"
        "- 1799: Napoleon takes power."
    ),
    4: (
        "Machine learning is when computers learn patterns from data, "
        "like learning to recognise images by seeing many examples."
    ),
    5: (
        "Binary search has O(log n) complexity because it halves the "
        "search space with each comparison."
    ),
    6: (
        "Inflation is caused by excess money supply or high demand. "
        "Central banks raise interest rates to slow price increases."
    ),
    7: (
        "RAM is volatile temporary storage erased on power loss. "
        "ROM is permanent non-volatile memory storing firmware."
    ),
    8: (
        "The sky is blue due to Rayleigh scattering of short wavelengths. "
        "At sunset, longer red wavelengths dominate after more atmosphere."
    ),
    9: (
        "1. Boil water. 2. Add tea bag. 3. Pour water. "
        "4. Steep 3-5 minutes. 5. Remove bag. 6. Add milk or sugar."
    ),
    10: (
        "Run python -m venv venv to create. Activate with "
        "venv\\Scripts\\activate. Install with pip. Type deactivate to exit."
    ),
    11: (
        "1. Run git merge. 2. Open file with <<<<<<, =======, >>>>>>> markers. "
        "3. Edit to resolve. 4. git add. 5. git commit. 6. git push."
    ),
    12: (
        "A list comprehension creates lists concisely: "
        "[x**2 for x in range(10)] produces [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]."
    ),
    13: (
        "Big O describes runtime growth. O(n) is a single loop. "
        "O(n²) is nested loops. Example: for i in range(n) is O(n)."
    ),
    14: (
        "Recursion is when a function calls itself. "
        "def factorial(n): return 1 if n==0 else n*factorial(n-1). "
        "factorial(5) = 120."
    ),
}


def _make_client() -> OpenAI:
    """Create OpenAI client from environment variables."""
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
        api_key=os.getenv("HF_TOKEN", ""),
    )


class Grader:
    """
    Reward computation for PromptRL.
    Uses OpenAI Python client for all LLM calls (hackathon requirement).

    Args:
        grader_type: 'rouge' (no API) or 'openai_client' (real LLM).
    """

    def __init__(self, grader_type: str = "rouge", **kwargs) -> None:
        self.grader_type = grader_type
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    def score(self, prompt: str, reference_answer: str, task_id: int) -> tuple[float, str]:
        """
        Score a prompt against the reference answer.

        Returns:
            (rouge_l_f1, llm_output) — score is float in [0.0, 1.0].
        """
        llm_output = self._get_output(prompt, task_id)
        rouge_l = self._compute_rouge(llm_output, reference_answer)
        return rouge_l, llm_output

    def _get_output(self, prompt: str, task_id: int) -> str:
        """Get LLM output. Uses OpenAI client if grader_type='openai_client', else dummy."""
        if self.grader_type == "openai_client":
            try:
                return self._call_llm(prompt)
            except Exception as e:
                print(f"[PromptRL] [LLM] Fallback to dummy (error: {e})", file=sys.stderr)
                return DUMMY_OUTPUTS.get(task_id, "")
        return DUMMY_OUTPUTS.get(task_id, "")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using the OpenAI Python client.
        MANDATORY approach — no raw HTTP to LLM endpoints.
        """
        client = _make_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
            timeout=30,
        )
        return response.choices[0].message.content or ""

    def _compute_rouge(self, hypothesis: str, reference: str) -> float:
        """ROUGE-L F1 between hypothesis and reference. Returns float in [0.0, 1.0]."""
        if not hypothesis or not reference:
            return 0.0
        scores = _ROUGE.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)

    @staticmethod
    def clip_reward(reward: float) -> float:
        """Clip to [-2.0, +2.0]. Wider range than pure quality env to accommodate STOP bonus."""
        return round(max(-2.0, min(2.0, reward)), 4)
