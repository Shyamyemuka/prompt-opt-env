# IMPLEMENTATION_PLAN.md — Step-by-Step Build Plan
# Project: PromptRL — Cost-Aware Task-Adaptive Prompt Optimization
# Deadline: April 7, 2026, 11:59 PM IST
# Today: March 30, 2026 | Days remaining: 9
# Cross-reference: BACKEND_STRUCTURE.md (all code), TECH_STACK.md
# Version: FINAL
# Last updated: 2026-03-30

---

## Schedule

| Day | Date | Phase | Focus | Hours |
|---|---|---|---|---|
| 1 | Mar 30 (Sun) | 0 + 1 | Prerequisites + scaffold | 3–4 h |
| 2 | Mar 31 (Mon) | 2 | models.py + task_bank.py | 2–3 h |
| 3 | Apr 1 (Tue) | 3 | actions.py + tests | 2–3 h |
| 4 | Apr 2 (Wed) | 4 | grader.py + tests | 2–3 h |
| 5 | Apr 3 (Thu) | 5 | prompt_rl_environment.py | 3–4 h |
| 6 | Apr 4 (Fri) | 6 | inference.py + integration test | 3–4 h |
| 7 | Apr 5 (Sat) | 7 | Docker build + fix | 2–3 h |
| 8 | Apr 6 (Sun) | 8 | HF deploy + pre-validation | 2–3 h |
| 9 | Apr 7 (Mon) | 9 | README + polish + SUBMIT | 2–3 h |

**Total: ~25 hours over 9 days.**
**Hard deadline: April 7, 2026, 11:59 PM IST.**

---

## Phase 0: Prerequisites (Day 1 — first 30 minutes)

### 0.1 Register and join Discord
```
1. https://www.scaler.com/school-of-technology/meta-pytorch-hackathon → Register for FREE
2. Discord: https://discord.gg/Dedhy5pkWD (all announcements here)
```

### 0.2 HuggingFace account and token
```
1. Sign up at https://huggingface.co/
2. Settings → Access Tokens → New token → Read permission
3. Copy: hf_xxxxxxxxxxxxxxxx
```

### 0.3 Install requirements
```bash
python --version     # Must be 3.11.x
# If not: https://www.python.org/downloads/release/python-3119/

pip install uv
# Docker Desktop: https://www.docker.com/products/docker-desktop/
docker --version     # Verify

pip install openenv-core openai
openenv --help
python -c "from openai import OpenAI; print('OpenAI OK')"
```

### 0.4 GitHub setup
```bash
# Create public repo: prompt-rl on GitHub
git clone https://github.com/{username}/prompt-rl.git
cd prompt-rl
```

---

## Phase 1: Scaffold (Day 1 — remaining time)

### 1.1 Run openenv init
```bash
openenv init prompt_rl
ls prompt_rl/
# Verify: __init__.py  client.py  models.py  openenv.yaml  pyproject.toml  server/  README.md
```

### 1.2 Create missing files
```bash
# Additional server files
touch prompt_rl/server/actions.py
touch prompt_rl/server/grader.py
touch prompt_rl/server/task_bank.py

# Test files
mkdir -p prompt_rl/tests
touch prompt_rl/tests/__init__.py
touch prompt_rl/tests/test_actions.py
touch prompt_rl/tests/test_grader.py
touch prompt_rl/tests/test_environment.py

# MANDATORY: inference.py at repo root
touch inference.py

# Config templates
touch .env.example
touch .gitignore
```

### 1.3 Write .env.example and .gitignore
```bash
cat > .env.example << 'EOF'
API_BASE_URL=https://api-inference.huggingface.co/v1/
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
MAX_STEPS=7
DONE_THRESHOLD=0.85
TOKEN_PENALTY_ALPHA=0.02
GRADER=rouge
TASK_SEED=
ENABLE_WEB_INTERFACE=false
EOF

cp .env.example .env
# Edit .env: replace hf_xxxxxxxx with your real token

cat > .gitignore << 'EOF'
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
dist/
*.egg-info/
.venv/
venv/
uv.lock
.DS_Store
EOF
```

### 1.4 Update pyproject.toml
Replace stub with content from TECH_STACK.md Section 9.
Key: `openai==1.30.1` is present. `httpx` is absent.

### 1.5 Install dependencies
```bash
cd prompt_rl
uv pip install -e ".[dev]"

python -c "import openenv; print('openenv OK')"
python -c "from openai import OpenAI; print('openai OK')"
python -c "from rouge_score import rouge_scorer; print('rouge OK')"
python -c "import pydantic; print(f'pydantic {pydantic.VERSION} OK')"
```

### 1.6 Initial commit
```bash
cd ..
git add .
git commit -m "feat: scaffold with openenv init, inference.py placeholder, .env.example"
git push origin main
```

---

## Phase 2: Data Layer (Day 2)

### 2.1 Write prompt_rl/models.py
Copy from BACKEND_STRUCTURE.md Section 2.1.
Two classes: `PromptAction` (action_id 0–5) and `PromptObservation` (ALL fields including the new token fields).

Verify:
```bash
cd prompt_rl
python -c "
from models import PromptAction, PromptObservation
a = PromptAction(action_id=5)  # STOP action
assert a.action_id == 5
print('PromptAction OK — including STOP')
"
```

### 2.2 Write server/task_bank.py
Copy from BACKEND_STRUCTURE.md Section 2.2.
All 15 tasks. All fields including `token_budget` per task.
Token budgets: easy=80, medium=65, hard=55.

Verify:
```python
python -c "
from server.task_bank import TASK_BANK
assert len(TASK_BANK) == 15
assert all(hasattr(t, 'token_budget') for t in TASK_BANK)
easy = [t for t in TASK_BANK if t.difficulty == 'easy']
medium = [t for t in TASK_BANK if t.difficulty == 'medium']
hard = [t for t in TASK_BANK if t.difficulty == 'hard']
assert all(t.token_budget == 80 for t in easy)
assert all(t.token_budget == 65 for t in medium)
assert all(t.token_budget == 55 for t in hard)
print(f'task_bank OK — {len(easy)} easy, {len(medium)} medium, {len(hard)} hard')
"
```

### 2.3 Commit
```bash
cd ..
git add .
git commit -m "feat: models.py (6 actions incl STOP, all token fields) + task_bank.py (15 tasks with token budgets)"
```

---

## Phase 3: Actions (Day 3)

### 3.1 Write server/actions.py
Copy from BACKEND_STRUCTURE.md Section 2.3.
Five edit functions + `count_tokens()` + `apply_action()` dispatcher.
Action 5 (STOP) is handled in the environment, not here.

### 3.2 Write tests/test_actions.py
```python
import pytest
from prompt_rl.server.actions import (
    add_context, shorten, add_example, rephrase, add_constraint,
    apply_action, count_tokens, ACTION_NAMES
)
from prompt_rl.server.task_bank import TASK_BANK


def test_count_tokens():
    assert count_tokens("hello world") == 2
    assert count_tokens("one two three four five") == 5
    assert count_tokens("") == 0


def test_add_context_appends_and_increases_tokens():
    original = "Tell me about dogs."
    result = add_context(original, "Dogs are domesticated mammals.")
    assert "Context:" in result
    assert count_tokens(result) > count_tokens(original)


def test_add_context_no_duplicate():
    p = "Tell me.\nContext: Dogs are domesticated mammals."
    assert add_context(p, "Dogs are domesticated mammals.") == p


def test_shorten_reduces_tokens():
    p = "Please could you explain machine learning."
    result = shorten(p)
    assert count_tokens(result) < count_tokens(p)
    assert "please" not in result.lower()
    assert "could you" not in result.lower()


def test_shorten_no_filler_is_noop():
    p = "Explain machine learning."
    assert shorten(p) == p


def test_add_example_appends():
    result = add_example("Summarise this.", "[example]")
    assert "Example output format:" in result


def test_add_example_no_duplicate():
    p = "Summarise.\nExample output format: [example]"
    assert add_example(p, "[example]") == p


def test_rephrase_net_zero_tokens():
    """REPHRASE should have near-zero net token change."""
    p = "Can you explain recursion?"
    result = rephrase(p)
    delta = abs(count_tokens(result) - count_tokens(p))
    assert delta <= 2, f"REPHRASE changed token count by {delta}, expected <=2"


def test_add_constraint_appends():
    result = add_constraint("Summarise.", "Under 50 words.")
    assert "Requirement:" in result


def test_apply_action_all_ids_0_to_4():
    task = TASK_BANK[0]
    for action_id in range(5):
        result = apply_action(action_id, "simple prompt about the topic", task)
        assert isinstance(result, str)


def test_apply_action_5_raises():
    """Action 5 (STOP) must not be passed to apply_action — handled by env."""
    with pytest.raises(ValueError):
        apply_action(5, "prompt", TASK_BANK[0])


def test_apply_action_invalid_raises():
    with pytest.raises(ValueError):
        apply_action(99, "prompt", TASK_BANK[0])


def test_action_names_complete():
    assert set(ACTION_NAMES.keys()) == {0, 1, 2, 3, 4, 5}
    assert ACTION_NAMES[5] == "STOP"
```

### 3.3 Run tests
```bash
cd prompt_rl
python -m pytest tests/test_actions.py -v
# ALL must pass
```

### 3.4 Commit
```bash
cd ..
git add .
git commit -m "feat: actions.py (5 edit functions + count_tokens) + passing tests"
```

---

## Phase 4: Grader (Day 4)

### 4.1 Write server/grader.py
Copy from BACKEND_STRUCTURE.md Section 2.4.
DUMMY_OUTPUTS for all 15 tasks. OpenAI client. Fallback on error.

### 4.2 Write tests/test_grader.py
```python
import pytest
from unittest.mock import patch
from prompt_rl.server.grader import Grader, DUMMY_OUTPUTS


def test_dummy_outputs_cover_all_tasks():
    for task_id in range(15):
        assert task_id in DUMMY_OUTPUTS
        assert len(DUMMY_OUTPUTS[task_id]) > 10


def test_rouge_grader_returns_float_in_range():
    grader = Grader(grader_type="rouge")
    score, output = grader.score("prompt", "reference answer", task_id=14)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_rouge_grader_never_calls_llm():
    grader = Grader(grader_type="rouge")
    with patch.object(grader, "_call_llm") as mock:
        grader.score("prompt", "reference", task_id=0)
        mock.assert_not_called()


def test_openai_client_grader_calls_llm():
    grader = Grader(grader_type="openai_client")
    with patch.object(grader, "_call_llm", return_value="A test output") as mock:
        score, output = grader.score("What is recursion?", "Recursion calls itself.", task_id=14)
        mock.assert_called_once()
        assert isinstance(score, float)


def test_openai_client_fallback_on_error():
    grader = Grader(grader_type="openai_client")
    with patch.object(grader, "_call_llm", side_effect=Exception("down")):
        score, output = grader.score("prompt", "ref", task_id=0)
    assert isinstance(score, float)  # must not crash


def test_clip_reward():
    assert Grader.clip_reward(0.5) == 0.5
    assert Grader.clip_reward(-3.0) == -2.0   # clipped at -2
    assert Grader.clip_reward(3.0) == 2.0     # clipped at +2
    assert Grader.clip_reward(1.125) == 1.125 # stop bonus within range
```

### 4.3 Run tests
```bash
python -m pytest tests/test_grader.py -v
```

### 4.4 Commit
```bash
cd ..
git add .
git commit -m "feat: grader.py (ROUGE + OpenAI client, fallback) + passing tests"
```

---

## Phase 5: Core Environment (Day 5)

### 5.1 Write server/prompt_rl_environment.py
Copy from BACKEND_STRUCTURE.md Section 2.5.
This is the most important file — every path documented in APP_FLOW.md Section 4 must be implemented.

### 5.2 Write server/app.py
Three lines. Copy from BACKEND_STRUCTURE.md Section 2.6.

### 5.3 Update prompt_rl/__init__.py
Copy from BACKEND_STRUCTURE.md Section 2.9.

### 5.4 Write tests/test_environment.py
```python
import pytest
from prompt_rl.server.prompt_rl_environment import PromptRLEnvironment
from prompt_rl.models import PromptAction


def test_reset_all_fields_present():
    env = PromptRLEnvironment()
    obs = env.reset()
    required = [
        "task_description", "current_prompt", "previous_prompt",
        "current_score", "previous_score", "current_token_count",
        "previous_token_count", "token_budget", "tokens_remaining",
        "token_overhead", "reward", "done", "step_count", "reference_answer", "info",
    ]
    for f in required:
        assert hasattr(obs, f), f"Missing field: {f}"
    required_info = ["grader_used", "action_applied", "stuck_count",
                     "termination_reason", "llm_output_preview", "no_op"]
    for k in required_info:
        assert k in obs.info, f"Missing info key: {k}"


def test_reset_initial_values():
    env = PromptRLEnvironment()
    obs = env.reset()
    assert obs.step_count == 0
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.previous_prompt == ""
    assert obs.token_overhead == 0
    assert obs.token_budget > 0
    assert obs.current_token_count > 0
    assert obs.tokens_remaining == obs.token_budget - obs.current_token_count


def test_step_returns_valid_observation():
    env = PromptRLEnvironment()
    env.reset()
    obs = env.step(PromptAction(action_id=0))
    assert obs.step_count == 1
    assert isinstance(obs.reward, float)
    assert isinstance(obs.done, bool)


def test_stop_action_terminates():
    env = PromptRLEnvironment()
    env.reset()
    obs = env.step(PromptAction(action_id=5))
    assert obs.done is True
    assert obs.info["action_applied"] == "STOP"
    assert obs.info["termination_reason"] == "voluntary_stop"
    assert obs.reward >= 0.0


def test_stop_reward_equals_score_times_1_5():
    env = PromptRLEnvironment()
    reset_obs = env.reset()
    current_score = reset_obs.current_score
    stop_obs = env.step(PromptAction(action_id=5))
    expected = round(current_score * 1.5, 4)
    assert abs(stop_obs.reward - expected) < 0.001


def test_stuck_terminates_with_penalty():
    env = PromptRLEnvironment()
    env.reset()
    obs = None
    for _ in range(5):
        obs = env.step(PromptAction(action_id=0))
        if obs.done: break
    assert obs.done is True
    assert obs.reward == -0.5
    assert obs.info["termination_reason"] == "stuck"


def test_token_counting_is_consistent():
    env = PromptRLEnvironment()
    obs = env.reset()
    expected = len(obs.current_prompt.split())
    assert obs.current_token_count == expected


def test_tokens_remaining_is_correct():
    env = PromptRLEnvironment()
    obs = env.reset()
    assert obs.tokens_remaining == obs.token_budget - obs.current_token_count


def test_max_steps_terminates():
    env = PromptRLEnvironment()
    env.reset()
    obs = None
    for i in range(7):
        obs = env.step(PromptAction(action_id=i % 5))
        if obs.done: break
    assert obs.done is True


def test_all_five_edit_actions_work():
    for action_id in range(5):
        env = PromptRLEnvironment()
        env.reset()
        obs = env.step(PromptAction(action_id=action_id))
        assert obs is not None
        assert isinstance(obs.reward, float)


def test_state_returns_episode_id():
    env = PromptRLEnvironment()
    env.reset()
    state = env.state
    assert state.episode_id != ""
    assert state.step_count == 0
```

### 5.5 Run ALL tests
```bash
cd prompt_rl
python -m pytest tests/ -v --tb=short
# Every test must pass before proceeding
```

### 5.6 Commit
```bash
cd ..
git add .
git commit -m "feat: complete PromptRLEnvironment (STOP, budget, stuck, all paths) + all tests passing"
```

---

## Phase 6: inference.py + Integration Test (Day 6)

### 6.1 Write inference.py at repo root
Copy from BACKEND_STRUCTURE.md Section 2.10. This is the mandatory baseline script.

Key checks before running:
- File is at repo root (not inside prompt_rl/)
- Uses `os.environ["API_BASE_URL"]` (not os.getenv)
- Uses `openai.OpenAI` client — no httpx
- Covers 3 tasks: easy (task_id=4), medium (task_id=5), hard (task_id=11)
- All exceptions wrapped in try/except
- Prints efficiency metric (final_score / final_token_count)

### 6.2 Set env vars and test inference.py
```bash
# Linux/Mac:
export API_BASE_URL=https://api-inference.huggingface.co/v1/
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
export HF_TOKEN=hf_your_real_token

python inference.py
# Must print summary table with 3 rows and efficiency column
# Must exit with code 0
# Must complete in under 20 minutes
```

### 6.3 Start server and run full episode
```bash
# Terminal 1:
cd prompt_rl
GRADER=rouge uv run server

# Terminal 2:
python -c "
import asyncio, random
from prompt_rl import PromptRLEnv, PromptAction

async def main():
    async with PromptRLEnv(base_url='ws://localhost:8000') as client:
        result = await client.reset()
        print(f'Task: {result.observation.task_description}')
        print(f'Budget: {result.observation.token_budget} tokens')
        while not result.done:
            obs = result.observation
            print(f'Tokens: {obs.current_token_count}/{obs.token_budget} | score={obs.current_score:.4f}')
            result = await client.step(PromptAction(action_id=random.randint(0, 5)))
            print(f'Step {obs.step_count+1}: {result.observation.info[\"action_applied\"]} reward={result.reward:.4f} done={result.done}')
        print(f'Done: {result.observation.info[\"termination_reason\"]}')

asyncio.run(main())
"
```

### 6.4 Verify JSON completeness
```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}" | python -m json.tool
# Confirm ALL fields in FRONTEND_GUIDELINES.md Section 4 are present
# Pay attention to: current_token_count, token_budget, tokens_remaining, token_overhead
```

### 6.5 Commit
```bash
cd ..
git add .
git commit -m "feat: inference.py complete (cost-aware summary), full integration test passing"
```

---

## Phase 7: Docker (Day 7)

### 7.1 Write server/requirements.txt and Dockerfile
Copy from BACKEND_STRUCTURE.md Sections 2.8 and 2.7.

### 7.2 Build Docker image
```bash
# If openenv-base:latest not found locally:
git clone https://github.com/meta-pytorch/OpenEnv.git /tmp/openenv
cd /tmp/openenv && docker build -t openenv-base:latest . && cd -

# Build env image:
docker build -t prompt-rl:latest -f prompt_rl/server/Dockerfile .
# Must complete with no errors
```

### 7.3 Run and test Docker container
```bash
docker run -d -p 8000:8000 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1/ \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
  -e HF_TOKEN=hf_your_token \
  -e GRADER=rouge \
  -e TOKEN_PENALTY_ALPHA=0.02 \
  prompt-rl:latest

sleep 5
curl http://localhost:8000/health
# Must return: {"status": "ok"}

curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"
# Must return full JSON with token fields
```

### 7.4 Run openenv validate
```bash
cd prompt_rl
openenv validate
# Must pass with no errors
```

### 7.5 Commit
```bash
cd ..
git add .
git commit -m "feat: Docker build clean, openenv validate passes"
```

---

## Phase 8: HF Deploy + Pre-Validation (Day 8)

### 8.1 Login and deploy
```bash
pip install huggingface-hub
huggingface-cli login

cd prompt_rl
openenv push --repo-id {username}/prompt-rl
# ~5 minutes to build and start
```

### 8.2 Set secrets in HF Spaces UI
```
https://huggingface.co/spaces/{username}/prompt-rl
Settings → Repository secrets:

  API_BASE_URL = https://api-inference.huggingface.co/v1/
  MODEL_NAME   = mistralai/Mistral-7B-Instruct-v0.2
  HF_TOKEN     = hf_your_actual_token
```

### 8.3 Verify live
```bash
curl https://{username}-prompt-rl.hf.space/health
# {"status": "ok"}

curl -s -X POST https://{username}-prompt-rl.hf.space/reset \
  -H "Content-Type: application/json" -d "{}"
# Full JSON with all token fields
```

### 8.4 Run pre-submission validation script
```bash
# Download from dashboard (Pre Validation Script button)
python validate.py
# Must show ALL PASS — fix any failures before continuing
```

### 8.5 Commit and tag
```bash
git add .
git commit -m "deploy: live on HF Spaces, all pre-validation checks passing"
git tag v1.0.0-round1
git push origin main --tags
```

---

## Phase 9: Final Polish + Submit (Day 9 — April 7)

### 9.1 Write README.md (2 hours)
Follow FRONTEND_GUIDELINES.md Section 5. All 14 sections required.
Lead with the differentiation paragraph — this is what judges read first.
Include the worked episode example showing cost-aware trade-offs.
Fill in actual numbers from running `python inference.py`.

### 9.2 Finalise openenv.yaml
Copy from TECH_STACK.md Section 11 exactly. The description field is what the LLM judge reads — do not shorten it.

### 9.3 Final pre-submission checklist

Tick every item before submitting:

```
[ ] openenv validate — no errors
[ ] docker build succeeds from repo root
[ ] docker run — health check returns 200
[ ] reset() returns PromptObservation with ALL fields (including all token fields)
[ ] step() for action_ids 0–5 all return valid float reward
[ ] action_id=5 (STOP) returns done=True, reward=current_score*1.5
[ ] token budget enforcement works — exceeding returns done=True, reward=-0.5
[ ] state() returns episode_id and step_count
[ ] inference.py at repo root (not inside prompt_rl/)
[ ] python inference.py — prints cost-aware summary with efficiency column, exits 0
[ ] inference.py uses os.environ["API_BASE_URL"], ["MODEL_NAME"], ["HF_TOKEN"]
[ ] All LLM calls use openai.OpenAI — no httpx/requests to LLM endpoints
[ ] HF Spaces URL returns 200 and responds to reset()
[ ] API_BASE_URL, MODEL_NAME, HF_TOKEN set as HF Spaces secrets
[ ] GitHub repo is public
[ ] openenv.yaml: name, description (full paragraph), tags, all env_vars
[ ] README: differentiation paragraph, all 14 sections, baseline scores with efficiency
[ ] .env NOT committed to GitHub
[ ] Pre-submission validation script passes ALL checks
[ ] 3+ tasks verified to return scores in [0.0, 1.0]
[ ] TOKEN_PENALTY_ALPHA documented in README and openenv.yaml
```

### 9.4 Submit
```
https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard
→ Submit your Assessment
→ Paste: https://{username}-prompt-rl.hf.space
→ Submit before April 7, 2026, 11:59 PM IST
```

---

## Post-Submission: Round 2 Prep (April 8–24)

- Complete all 4 prep course modules at the dashboard
- Attend bootcamp April 18–19 (online)
- Prototype exact BPE token counting with tiktoken
- Design adaptive α: penalty increases as budget depletes
- Sketch multi-objective Pareto front tracking (quality vs. cost frontier)
- Prototype LLM-as-judge grader replacement for ROUGE-L
- Prepare 3-minute pitch: "Why existing prompt optimizers fail in production, and how PromptRL fixes that"
