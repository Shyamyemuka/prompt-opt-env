"""
Grading logic for PromptOptEnv.
ALL LLM calls use the OpenAI Python client — required by hackathon rules.
Never use httpx or requests for LLM calls.

Modes:
  'rouge'         — uses DUMMY_OUTPUTS[task_id], no API, always works
  'openai_client' — calls LLM via OpenAI client, falls back to dummy on error
"""
import os
import sys
import math
from rouge_score import rouge_scorer

try:
    from ..llm_router import create_default_router
except (ModuleNotFoundError, ImportError):
    from llm_router import create_default_router


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
SCORE_EPSILON = 0.05

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

class Grader:
    """
    Reward computation for PromptOptEnv.
    Uses OpenAI Python client for all LLM calls (hackathon requirement).

    Args:
        grader_type: 'rouge' (no API) or 'openai_client' (real LLM).
    """

    def __init__(self, grader_type: str = "rouge", **kwargs) -> None:
        self.grader_type = grader_type
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self._router = create_default_router(
            default_model=self.model_name,
            default_base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/"),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

    def score(self, prompt: str, reference_answer: str, task_id: int) -> tuple[float, str]:
        """
        Score a prompt against the reference answer.

        Returns:
            (rouge_l_f1, llm_output) — score is float in (0.0, 1.0).
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
                print(f"[PromptOptEnv] [LLM] Fallback to dummy (error: {e})", file=sys.stderr)
                return DUMMY_OUTPUTS.get(task_id, "")
        return DUMMY_OUTPUTS.get(task_id, "")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using the provider router (OpenAI-compatible clients).
        MANDATORY approach — no raw HTTP to LLM endpoints.
        """
        text = self._router.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        if not text:
            raise RuntimeError(self._router.last_error or "llm_call_failed")
        return text

    def _compute_rouge(self, hypothesis: str, reference: str) -> float:
        """ROUGE-L F1 between hypothesis and reference. Returns float in (0.0, 1.0)."""
        if not hypothesis or not reference:
            return SCORE_EPSILON
        scores = _ROUGE.score(reference, hypothesis)
        raw = float(scores["rougeL"].fmeasure)
        if not math.isfinite(raw):
            return SCORE_EPSILON
        # OpenEnv evaluator requires strict bounds: 0.0 < score < 1.0.
        bounded = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, raw))
        rounded = round(bounded, 4)
        return float(max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, rounded)))

    @staticmethod
    def clip_reward(reward: float) -> float:
        """Clip to [-2.0, +2.0]. Wider range than pure quality env to accommodate STOP bonus."""
        return round(max(-2.0, min(2.0, reward)), 4)
