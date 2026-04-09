"""
Heuristic Policy Agent for PromptOptEnv.

This agent encodes domain knowledge about optimal prompt optimization:
- Use high-impact actions early (ADD_EXAMPLE, ADD_CONTEXT)
- Use cheap actions for refinement (REPHRASE, SHORTEN)
- STOP when quality is good enough or budget is tight

The policy is interpretable and deterministic - perfect for demonstrating
that the environment has learnable structure.
"""

from dataclasses import dataclass, field
from typing import Literal
import random


@dataclass
class AgentState:
    """Track episode state for the agent."""
    episode_id: str = ""
    step_count: int = 0
    actions_used: set[int] = field(default_factory=set)
    last_action: int | None = None
    stuck_count: int = 0


class HeuristicAgent:
    """
    Rule-based agent that optimizes prompts using domain knowledge.

    Decision hierarchy (first rule that matches wins):
    1. STOP if score > 0.70 and budget healthy
    2. STOP if budget critically low (<15% remaining)
    3. SHORTEN if budget tight (<25% remaining)
    4. ADD_EXAMPLE if score < 0.40 (high impact)
    5. ADD_CONTEXT if score < 0.60 (moderate impact)
    6. REPHRASE if not yet used (free token-wise)
    7. ADD_CONSTRAINT if score < 0.70
    8. STOP as fallback

    Args:
        score_threshold: Score above which to consider STOP (default: 0.70)
        budget_critical: Budget % below which to STOP immediately (default: 0.15)
        budget_tight: Budget % below which to SHORTEN (default: 0.25)
        random_seed: For reproducible tie-breaking
    """

    ACTION_NAMES = {
        0: "ADD_CONTEXT",
        1: "SHORTEN",
        2: "ADD_EXAMPLE",
        3: "REPHRASE",
        4: "ADD_CONSTRAINT",
        5: "STOP",
    }

    def __init__(
        self,
        score_threshold: float = 0.70,
        budget_critical: float = 0.15,
        budget_tight: float = 0.25,
        random_seed: int = 42,
    ):
        self.score_threshold = score_threshold
        self.budget_critical = budget_critical
        self.budget_tight = budget_tight
        self.rng = random.Random(random_seed)
        self.state = AgentState()

    def reset(self, episode_id: str = "") -> None:
        """Reset agent state for new episode."""
        self.state = AgentState(
            episode_id=episode_id,
            step_count=0,
            actions_used=set(),
            last_action=None,
            stuck_count=0,
        )

    def select_action(
        self,
        current_score: float,
        current_token_count: int,
        token_budget: int,
        step_count: int,
        max_steps: int = 7,
    ) -> tuple[int, str]:
        """
        Select action based on current state.

        Args:
            current_score: ROUGE-L score (0.0-1.0)
            current_token_count: Current prompt token count
            token_budget: Hard token limit for this task
            step_count: Current step number (0-indexed)
            max_steps: Max steps per episode

        Returns:
            (action_id, reason) tuple
        """
        self.state.step_count = step_count
        tokens_remaining = token_budget - current_token_count
        budget_remaining_pct = tokens_remaining / token_budget if token_budget > 0 else 0

        # Rule 1: Success - good score with healthy budget
        if current_score > self.score_threshold and budget_remaining_pct > 0.20:
            return 5, f"success_stop (score={current_score:.3f} > {self.score_threshold})"

        # Rule 2: Critical budget - must stop
        if budget_remaining_pct < self.budget_critical:
            return 5, f"critical_budget_stop (remaining={budget_remaining_pct:.1%})"

        # Rule 3: Tight budget - try to reduce
        if budget_remaining_pct < self.budget_tight:
            # Only SHORTEN if we haven't already (avoid stuck detection)
            if 1 not in self.state.actions_used:
                return 1, f"tight_budget_shorten (remaining={budget_remaining_pct:.1%})"
            return 5, f"tight_budget_stop (remaining={budget_remaining_pct:.1%})"

        # Rule 4: Low score - high impact action needed
        if current_score < 0.40 and 2 not in self.state.actions_used:
            return 2, f"low_score_example (score={current_score:.3f})"

        # Rule 5: Medium-low score - moderate impact
        if current_score < 0.60 and 0 not in self.state.actions_used:
            return 0, f"medium_score_context (score={current_score:.3f})"

        # Rule 6: Free win - REPHRASE if not used (net 0 tokens)
        if 3 not in self.state.actions_used:
            return 3, "rephrase_free (first use)"

        # Rule 7: Approaching threshold - add constraint for final boost
        if current_score < self.score_threshold and 4 not in self.state.actions_used:
            return 4, f"final_boost_constraint (score={current_score:.3f})"

        # Rule 8: Late in episode - consider STOP
        if step_count >= max_steps - 2:
            return 5, f"late_stop (step={step_count}/{max_steps})"

        # Rule 9: Random among remaining useful actions (avoid getting stuck)
        available = [a for a in [0, 1, 2, 3, 4] if a not in self.state.actions_used]
        if available:
            action = self.rng.choice(available)
            return action, f"unused_action_fallback (action={self.ACTION_NAMES[action]})"

        # Rule 10: Fallback to STOP
        return 5, "fallback_stop (no useful actions remaining)"

    def update(self, action_id: int, observation: dict) -> None:
        """
        Update internal state after taking action.

        Args:
            action_id: Action that was taken
            observation: Observation dict from environment
        """
        self.state.actions_used.add(action_id)

        # Track stuck detection
        if action_id == self.state.last_action:
            self.state.stuck_count += 1
        else:
            self.state.stuck_count = 0
        self.state.last_action = action_id

    def get_stats(self) -> dict:
        """Return agent statistics for this episode."""
        return {
            "actions_used": len(self.state.actions_used),
            "action_history": [self.ACTION_NAMES[a] for a in self.state.actions_used],
            "stuck_count": self.state.stuck_count,
        }


class RandomAgent:
    """Baseline agent that selects actions uniformly at random."""

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)
        self.action_history: list[int] = []

    def reset(self) -> None:
        """Reset for new episode."""
        self.action_history = []

    def select_action(self, **kwargs) -> tuple[int, str]:
        """Select random action 0-5."""
        action = self.rng.randint(0, 5)
        return action, "random"

    def update(self, action_id: int, observation: dict) -> None:
        """Track action history."""
        self.action_history.append(action_id)

    def get_stats(self) -> dict:
        """Return random agent stats."""
        return {
            "actions_taken": len(self.action_history),
            "action_history": self.action_history,
        }


class ImmediateStopAgent:
    """Baseline that stops immediately - tests if editing helps at all."""

    def __init__(self):
        self.stopped_at: int | None = None

    def reset(self) -> None:
        """Reset for new episode."""
        self.stopped_at = None

    def select_action(self, **kwargs) -> tuple[int, str]:
        """Always STOP on first step."""
        self.stopped_at = 0
        return 5, "immediate_stop"

    def update(self, action_id: int, observation: dict) -> None:
        """No state to track."""
        pass

    def get_stats(self) -> dict:
        """Return stats."""
        return {"stopped_at": self.stopped_at}


class AlwaysImproveAgent:
    """
    Baseline that never STOPs, always tries to improve quality.
    Tests cost of ignoring token budget.
    """

    ACTION_CYCLE = [2, 0, 3, 4, 1]  # EXAMPLE, CONTEXT, REPHRASE, CONSTRAINT, SHORTEN

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)
        self.step_count = 0

    def reset(self) -> None:
        """Reset for new episode."""
        self.step_count = 0

    def select_action(self, **kwargs) -> tuple[int, str]:
        """Cycle through improving actions, never STOP."""
        action = self.ACTION_CYCLE[self.step_count % len(self.ACTION_CYCLE)]
        self.step_count += 1
        return action, f"improve_cycle (step={self.step_count})"

    def update(self, action_id: int, observation: dict) -> None:
        """No additional state."""
        pass

    def get_stats(self) -> dict:
        """Return stats."""
        return {"steps_taken": self.step_count}


# Export all agents
AGENTS = {
    "heuristic": HeuristicAgent,
    "random": RandomAgent,
    "immediate_stop": ImmediateStopAgent,
    "always_improve": AlwaysImproveAgent,
}
