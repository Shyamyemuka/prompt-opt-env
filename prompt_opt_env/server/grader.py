"""
Grading logic for the PromptOptEnv RL environment.

Computes ROUGE-L score between LLM output and a reference answer.
Supports two modes:
  - 'rouge':  uses dummy canned outputs (no external API needed)
  - 'hf_api': calls HuggingFace Inference API for real LLM output,
              falls back to dummy output on failure.
"""

import os
import httpx
from rouge_score import rouge_scorer as rouge_scorer_module


# Module-level ROUGE scorer instance — reused across all calls
ROUGE_SCORER = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

# Dummy outputs keyed by task_id for when HF API is not available.
# These are keyword-matched canned responses that give the grader something to score.
DUMMY_OUTPUTS: dict[int, str] = {
    0: "Climate change is causing rising temperatures and extreme weather.",
    1: "Romeo and Juliet are young lovers from rival families who die together.",
    2: "Crypto is risky due to volatility and lack of regulation.",
    3: "The French Revolution started in 1789 and ended Napoleon's rise in 1799.",
    4: "Machine learning is when computers learn from data.",
    5: "Binary search is O(log n) because it halves the search space each time.",
    6: "Inflation is caused by too much money. Banks raise interest rates to control it.",
    7: "RAM is temporary storage. ROM is permanent. RAM is erased when power is off.",
    8: "The sky is blue because of light scattering. It is red at sunset due to atmosphere.",
    9: "Boil water, add tea bag, steep, remove bag, add milk.",
    10: "Run python -m venv venv, then venv\\Scripts\\activate on Windows.",
    11: "Open conflicted file, resolve markers, git add, git commit.",
    12: "List comprehension: [x**2 for x in range(10)] makes a list of squares.",
    13: "Big O describes growth rate. O(n) is linear, O(n²) is quadratic.",
    14: "Recursion calls itself. factorial(n) = n * factorial(n-1). Base case: factorial(0) = 1.",
}


class Grader:
    """
    Handles reward computation for the prompt optimization environment.

    Supports two grader modes:
      - 'rouge':   Pure ROUGE-L scoring against dummy canned outputs (no API needed).
      - 'hf_api':  Calls HuggingFace Inference API for real LLM output, scored with ROUGE-L.
                   Falls back to dummy output if API fails.
    """

    def __init__(
        self,
        grader_type: str = "rouge",
        hf_token: str = "",
        hf_model: str = "",
    ):
        """
        Initialise the grader.

        Args:
            grader_type: 'rouge' or 'hf_api'. Default: 'rouge'.
            hf_token: HuggingFace API token (required for hf_api mode).
            hf_model: HF model identifier. Default: Mistral-7B-Instruct-v0.2.
        """
        self.grader_type = grader_type
        self.hf_token = hf_token
        self.hf_model = hf_model or "mistralai/Mistral-7B-Instruct-v0.2"
        self._api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"

    def score(self, prompt: str, reference_answer: str, task_id: int) -> tuple[float, str]:
        """
        Score the given prompt against the reference answer.

        Gets LLM output (real or dummy), computes ROUGE-L F1 against reference.

        Args:
            prompt: The current prompt string to score.
            reference_answer: Gold-standard answer from the task bank.
            task_id: Integer task ID (used for selecting dummy output).

        Returns:
            Tuple of (rouge_l_score: float, llm_output_text: str).
        """
        llm_output = self._get_output(prompt, task_id)
        rouge_score = self._compute_rouge(llm_output, reference_answer)
        return rouge_score, llm_output

    def _get_output(self, prompt: str, task_id: int) -> str:
        """
        Get LLM output for the prompt. Falls back to dummy if API unavailable.

        Args:
            prompt: Prompt string to send to the LLM.
            task_id: Used to select the appropriate dummy output.

        Returns:
            LLM output string (real or dummy).
        """
        if self.grader_type == "hf_api" and self.hf_token:
            try:
                return self._call_hf_api(prompt)
            except Exception:
                # Fall back to dummy output silently
                return DUMMY_OUTPUTS.get(task_id, "No output available.")
        return DUMMY_OUTPUTS.get(task_id, "No output available.")

    def _call_hf_api(self, prompt: str) -> str:
        """
        Call HuggingFace Inference API synchronously.

        Args:
            prompt: Prompt string to send to the model.

        Returns:
            Generated text from the model.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP response.
            httpx.TimeoutException: If request takes longer than 10 seconds.
        """
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "temperature": 0.1},
        }
        response = httpx.post(
            self._api_url,
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)

    def _compute_rouge(self, hypothesis: str, reference: str) -> float:
        """
        Compute ROUGE-L F1 score between hypothesis and reference.

        Args:
            hypothesis: Generated text to evaluate.
            reference: Gold-standard reference answer.

        Returns:
            ROUGE-L F1 score as a float in [0.0, 1.0]. Returns 0.0 if either input is empty.
        """
        if not hypothesis or not reference:
            return 0.0
        scores = ROUGE_SCORER.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)

    @staticmethod
    def clip_reward(reward: float) -> float:
        """
        Clip reward to [-1.0, 2.0].

        Upper bound is 2.0 (not 1.0) to accommodate the success bonus (+1.0).

        Args:
            reward: Raw reward value.

        Returns:
            Reward clipped to the range [-1.0, 2.0].
        """
        return max(-1.0, min(2.0, reward))
