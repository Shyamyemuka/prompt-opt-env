# Agent Implementation Plan

## Current Gap Analysis

### Problem
The PromptOptEnv environment is fully functional but lacks demonstration that:
1. The action space enables intelligent decision-making
2. Cost-aware strategies outperform naive approaches
3. The STOP mechanic and token penalties create meaningful tradeoffs

For Round 1 selection, judges need evidence that the environment has "learnable structure" - not just that it runs.

### Solution
Implement a **Heuristic Agent** that encodes domain knowledge about optimal prompt optimization strategy, plus a **Benchmark Suite** that proves this strategy outperforms random action selection.

---

## Implementation Needs

### 1. Heuristic Policy Agent (`agent.py`)

**Core Insight:** Optimal prompt optimization follows predictable patterns:
- Early: Quality is low → use high-impact actions (ADD_EXAMPLE, ADD_CONTEXT)
- Mid: Quality improving → use cheap actions (REPHRASE, SHORTEN) to refine
- Late: Quality good or budget tight → STOP

**Decision Rules:**
```
IF current_score > 0.70 AND tokens_used < 60% budget → STOP (success)
IF tokens_remaining < 15% budget → SHORTEN or STOP (conserve)
IF current_score < 0.40 → ADD_EXAMPLE (biggest quality boost)
IF current_score < 0.60 → ADD_CONTEXT (moderate boost)
IF REPHRASE hasn't been used this episode → REPHRASE (free win)
ELSE → ADD_CONSTRAINT (last resort)
```

**Requirements:**
- Stateless (per-episode memory only)
- Deterministic (reproducible)
- Fast (no model loading)
- Explains its decisions (for debugging/demo)

### 2. Benchmark Suite (`benchmark.py`)

**Comparisons to Run:**
| Strategy | Description |
|----------|-------------|
| Random | Uniform random over 6 actions |
| Immediate STOP | Baseline: STOP on first step |
| Always Improve | Never STOP, always improving actions |
| Heuristic Agent | Rule-based policy |

**Metrics to Track:**
- `efficiency` = final_score / final_token_count
- `success_rate` = % episodes hitting score > 0.85
- `budget_compliance` = % episodes not exceeding token budget
- `avg_steps` = average episode length
- `avg_reward` = cumulative reward per episode

**Output:**
- Summary table (markdown)
- Per-task breakdown (optional)
- Statistical significance (heuristic vs random)

### 3. Documentation Updates

**README Additions:**
- "Benchmark Results" section with comparison table
- Explanation of heuristic strategy
- Key insight: cost-aware RL enables 30%+ efficiency gains

---

## Improvement Methodologies

### Phase 1: Baseline (Current)
- Random agent: proves environment works
- No comparison: fails to demonstrate value

### Phase 2: Heuristic Agent (This Implementation)
**Advantages:**
- Immediate working policy
- Interpretable decisions
- Proves environment structure exists
- Fast to implement (45 mins)

**Limitations:**
- Hardcoded rules don't adapt to tasks
- May be suboptimal on edge cases
- Not "learning" in ML sense

### Phase 3: Trained RL Agent (Future Work)
Once selected for Round 2, implement:
- **Q-Learning:** Tabular, state = (score_bucket, token_bucket, last_action)
- **Policy Gradient:** REINFORCE with small neural net
- **Comparison:** Show trained > heuristic > random

---

## Success Criteria

The implementation succeeds if:
1. Heuristic agent achieves >20% better efficiency than random
2. Heuristic uses STOP action intelligently (not too early, not too late)
3. Budget compliance >80% (vs random's ~50%)
4. Benchmark runs in <5 minutes
5. Results are reproducible (seeded RNG)

---

## Files to Create

```
prompt-opt-env/
├── agent.py              # HeuristicAgent class
├── benchmark.py          # Run comparisons, output table
└── AGENT.md              # This file
```

## Files to Modify

```
prompt-opt-env/
└── README.md             # Add "Benchmark Results" section
```
