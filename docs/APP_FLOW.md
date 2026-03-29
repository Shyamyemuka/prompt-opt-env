# APP_FLOW.md — Application Flow
# Project: PromptOptEnv
# Cross-reference: PRD.md Section 6, BACKEND_STRUCTURE.md

---

## Overview

PromptOptEnv has no frontend pages. It is a server-side RL environment exposed via WebSocket and HTTP. "Flow" in this context means the sequence of API calls, internal method calls, data transformations, and state transitions that happen across a full episode. Every possible path is documented here.

---

## 1. System Boot Flow

```
Developer runs: uv run server
    |
    ├── server/app.py loads
    |       ├── Reads env vars (MAX_STEPS, GRADER, HF_TOKEN, etc.)
    |       ├── Instantiates PromptOptEnvironment (or factory if HF_TOKEN present)
    |       └── Calls create_app(PromptOptEnvironment, PromptAction, PromptObservation)
    |
    ├── FastAPI app starts on 0.0.0.0:8000
    |       ├── WebSocket endpoint registered at /ws
    |       ├── HTTP endpoint registered at /reset (POST)
    |       ├── HTTP endpoint registered at /step (POST)
    |       ├── HTTP endpoint registered at /state (GET)
    |       └── Health check registered at /health (GET)
    |
    └── Server is ready to accept connections
```

---

## 2. Client Connection Flow

```
External code (RL trainer or test script):

    from prompt_opt_env import PromptOptEnv, PromptAction

    async with PromptOptEnv(base_url="ws://localhost:8000") as client:
        |
        ├── EnvClient.__init__ called
        |       └── Stores base_url, sets mode=async
        |
        ├── __aenter__ called
        |       └── Opens WebSocket connection to ws://localhost:8000/ws
        |               └── Server logs: "New session connected: {session_id}"
        |
        [... episode runs ...]
        |
        └── __aexit__ called
                └── Closes WebSocket connection
                        └── Server logs: "Session disconnected: {session_id}"

# Synchronous alternative:
with PromptOptEnv(base_url="ws://localhost:8000").sync() as client:
    [same flow, blocking calls]
```

---

## 3. Episode Start Flow — reset()

```
Client calls: result = await client.reset()
    |
    ├── EnvClient sends WebSocket message: {"type": "reset", "payload": {}}
    |
    ├── Server receives message
    |       └── Routes to PromptOptEnvironment.reset()
    |
    ├── PromptOptEnvironment.reset() executes:
    |       ├── Generates new episode_id = str(uuid4())
    |       ├── Resets step_count = 0
    |       ├── Resets stuck_count = 0
    |       ├── Resets last_action = None
    |       |
    |       ├── Task selection:
    |       |       ├── If TASK_SEED is set → task_id = TASK_SEED % 15
    |       |       └── Else → task_id = random.randint(0, 14)
    |       |
    |       ├── Loads task from TASK_BANK[task_id]:
    |       |       ├── task_description
    |       |       ├── initial_bad_prompt → set as current_prompt
    |       |       └── reference_answer
    |       |
    |       ├── Computes initial score:
    |       |       ├── Calls grader.score(initial_bad_prompt, reference_answer)
    |       |       |       ├── [ROUGE path] Calls _get_dummy_output(initial_bad_prompt)
    |       |       |       |       └── Returns a template response (no API call at reset)
    |       |       |       └── Returns ROUGE-L F1 score as float
    |       |       └── Stores as self.previous_score = initial_score
    |       |
    |       └── Returns PromptObservation:
    |               ├── task_description = task.task_description
    |               ├── current_prompt = task.initial_bad_prompt
    |               ├── previous_prompt = ""  (empty at start)
    |               ├── current_score = initial_score
    |               ├── previous_score = 0.0
    |               ├── reward = 0.0
    |               ├── done = False
    |               ├── step_count = 0
    |               ├── reference_answer = task.reference_answer
    |               └── info = {"grader_used": "rouge", "action_applied": None, "stuck_count": 0}
    |
    ├── Server serialises PromptObservation to JSON
    |
    └── Client receives StepResult:
            ├── result.observation → PromptObservation
            ├── result.reward → 0.0
            └── result.done → False
```

---

## 4. Episode Step Flow — step(action)

### 4A. Happy path — valid action, episode continues

```
Client calls: result = await client.step(PromptAction(action_id=2))
    |
    ├── EnvClient sends: {"type": "step", "payload": {"action_id": 2}}
    |
    ├── Server receives, routes to PromptOptEnvironment.step(action)
    |
    ├── Input validation:
    |       ├── action.action_id must be int in {0, 1, 2, 3, 4}
    |       └── If invalid → raises ValueError, server returns error message, episode NOT terminated
    |
    ├── Stuck detection check:
    |       ├── If action.action_id == self.last_action:
    |       |       self.stuck_count += 1
    |       └── Else:
    |               self.stuck_count = 0
    |               self.last_action = action.action_id
    |
    ├── If stuck_count >= 3:
    |       → Jump to Section 4C (stuck termination)
    |
    ├── Apply action transformation:
    |       ├── action_id=0 (ADD_CONTEXT):
    |       |       new_prompt = actions.add_context(current_prompt, task_description)
    |       |       → Appends: "\nContext: [domain-specific sentence derived from task_description]"
    |       |
    |       ├── action_id=1 (SHORTEN):
    |       |       new_prompt = actions.shorten(current_prompt)
    |       |       → Removes: filler phrases ("please", "could you", "I would like")
    |       |       → Removes: redundant clauses (sentences that repeat the main ask)
    |       |       → If prompt is already short (<50 chars): returns unchanged + reward=-0.1
    |       |
    |       ├── action_id=2 (ADD_EXAMPLE):
    |       |       new_prompt = actions.add_example(current_prompt, task_description)
    |       |       → Appends: "\nExample output format: [pre-written example from task bank]"
    |       |
    |       ├── action_id=3 (REPHRASE):
    |       |       new_prompt = actions.rephrase(current_prompt)
    |       |       → Converts passive to active voice (rule-based regex)
    |       |       → Converts questions to imperative ("Can you explain X" → "Explain X")
    |       |
    |       └── action_id=4 (ADD_CONSTRAINT):
    |               new_prompt = actions.add_constraint(current_prompt, task_description)
    |               → Appends: "\nRequirement: [constraint string from task bank]"
    |
    ├── No-op detection:
    |       └── If new_prompt == current_prompt:
    |               reward = -0.1
    |               → Skip grader call
    |               → Continue episode (no-op does NOT terminate episode)
    |
    ├── Grader call (only if new_prompt != current_prompt):
    |       ├── [ROUGE grader — default]:
    |       |       ├── calls grader.get_llm_output(new_prompt)
    |       |       |       ├── Uses HF Inference API if HF_TOKEN set
    |       |       |       |       POST https://api-inference.huggingface.co/models/{HF_MODEL}
    |       |       |       |       headers: {"Authorization": "Bearer {HF_TOKEN}"}
    |       |       |       |       body: {"inputs": new_prompt, "parameters": {"max_new_tokens": 200}}
    |       |       |       |       timeout: 10 seconds
    |       |       |       |       on failure → falls back to dummy output
    |       |       |       └── If no HF_TOKEN → uses _get_dummy_output(new_prompt)
    |       |       |               → Returns keyword-matched canned response from task bank
    |       |       |               → Ensures grader always has something to score
    |       |       |
    |       |       ├── calls rouge_scorer.score(reference_answer, llm_output)
    |       |       |       → Uses rouge_score.RougeScorer(['rougeL'], use_stemmer=True)
    |       |       |       → Extracts fmeasure from rougeL result
    |       |       |
    |       |       └── current_score = rougeL_fmeasure (float 0.0 to 1.0)
    |       |
    |       └── reward = clip(current_score - previous_score, -1.0, +1.0)
    |
    ├── Step count update:
    |       self.step_count += 1
    |       self.current_prompt = new_prompt
    |       self.previous_score = current_score
    |
    ├── Termination checks:
    |       ├── If current_score > DONE_THRESHOLD (0.85):
    |       |       done = True
    |       |       reward += 1.0  (success bonus, then clip to max 2.0)
    |       |
    |       └── If step_count >= MAX_STEPS (5):
    |               done = True
    |
    └── Returns PromptObservation:
            ├── task_description = (unchanged)
            ├── current_prompt = new_prompt
            ├── previous_prompt = old current_prompt
            ├── current_score = current_score
            ├── previous_score = self.previous_score (before this step)
            ├── reward = reward
            ├── done = done
            ├── step_count = self.step_count
            ├── reference_answer = task.reference_answer
            └── info = {
                    "grader_used": "rouge" or "hf_api",
                    "action_applied": action_name_string,
                    "stuck_count": self.stuck_count,
                    "llm_output_preview": first_100_chars_of_output
                }
```

### 4B. Success termination path

```
During step() — current_score > 0.85:
    |
    ├── reward += 1.0 (success bonus on top of delta reward)
    ├── done = True
    └── Returns observation with done=True
            → Client training loop should call reset() to start new episode
```

### 4C. Stuck termination path

```
During step() — stuck_count reaches 3:
    |
    ├── reward = -0.5 (stuck penalty, overrides delta reward)
    ├── done = True
    └── Returns observation with done=True, reward=-0.5
            → info["stuck_count"] = 3
```

### 4D. Max steps termination path

```
During step() — step_count reaches MAX_STEPS:
    |
    ├── reward = (normal delta reward, no penalty unless score is bad)
    ├── done = True
    └── Returns observation with done=True
```

---

## 5. State Query Flow — state()

```
Client calls: state = await client.state()
    |
    ├── EnvClient sends: {"type": "state"}
    |
    ├── Server returns PromptOptEnvironment.state property:
    |       └── State(
    |               episode_id=self._state.episode_id,
    |               step_count=self._state.step_count,
    |               task_id=self._state.task_id
    |           )
    |
    └── Client receives State object
```

---

## 6. Full Episode Example (happy path, 3 steps)

```
reset()
  → task: "Summarise the plot of Romeo and Juliet in exactly 2 sentences"
  → initial_prompt: "tell me about romeo and juliet"
  → initial_score: 0.12
  → reward: 0.0

step(action_id=3)  ← REPHRASE
  → new_prompt: "Summarise the plot of Romeo and Juliet."
  → current_score: 0.28
  → reward: +0.16
  → done: False

step(action_id=2)  ← ADD_EXAMPLE
  → new_prompt: "Summarise the plot of Romeo and Juliet.\nExample output format: [2 concise sentences covering key events]"
  → current_score: 0.51
  → reward: +0.23
  → done: False

step(action_id=4)  ← ADD_CONSTRAINT
  → new_prompt: "Summarise the plot of Romeo and Juliet.\nExample output format: [2 concise sentences covering key events]\nRequirement: Exactly 2 sentences, no more."
  → current_score: 0.72
  → reward: +0.21
  → done: False

step(action_id=1)  ← SHORTEN
  → new_prompt: "Summarise Romeo and Juliet in exactly 2 sentences.\nRequirement: Exactly 2 sentences, no more."
  → current_score: 0.81
  → reward: +0.09
  → done: False

step(action_id=0)  ← ADD_CONTEXT  [step 5 = MAX_STEPS]
  → new_prompt: "Summarise Romeo and Juliet in exactly 2 sentences.\nRequirement: Exactly 2 sentences, no more.\nContext: Shakespeare tragedy about two feuding families."
  → current_score: 0.83
  → reward: +0.02
  → done: True  ← MAX_STEPS reached

Total cumulative reward: 0.16 + 0.23 + 0.21 + 0.09 + 0.02 = 0.71
```

---

## 7. Error Paths

| Error | Trigger | Server behaviour | Client receives |
|---|---|---|---|
| Invalid action_id | action_id not in {0,1,2,3,4} | Raises ValueError, does not advance state | Error message in WebSocket response |
| HF API timeout | API call takes >10s | Falls back to dummy output, continues episode | Normal observation, info["grader_used"]="rouge_fallback" |
| HF API 429 rate limit | Too many requests | Falls back to dummy output | Same as above |
| step() called after done=True | Client error | Environment resets automatically, logs warning | reset() observation returned |
| WebSocket disconnects mid-episode | Network issue | Session cleaned up server-side, state lost | Connection error on client |

---

## 8. Deployment Flow (openenv push)

```
Developer runs: openenv push --repo-id {hf_username}/prompt-opt-env

    ├── openenv CLI reads openenv.yaml
    ├── Builds Docker image using server/Dockerfile
    ├── Pushes image to Hugging Face Spaces registry
    ├── Spaces starts the container (port 7860 exposed)
    └── Environment is live at:
            WebSocket: wss://{hf_username}-prompt-opt-env.hf.space/ws
            HTTP:      https://{hf_username}-prompt-opt-env.hf.space/

Client connects remotely:
    async with PromptOptEnv(base_url="wss://{hf_username}-prompt-opt-env.hf.space") as client:
        [same flow as local]
```
