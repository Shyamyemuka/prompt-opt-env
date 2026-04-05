"""
Web front-end for PromptOptEnv - Interactive Prompt Optimizer

Run:
    python web_app.py

The server starts automatically, loads .env.local, and opens your browser.
"""
import os
import sys
import threading
import webbrowser

# ── Resolve project root relative to this file ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Auto-load .env.local (HF_TOKEN, MODEL_NAME, etc.) ────────────────────────
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(BASE_DIR, ".env.local")
    if os.path.exists(_env_file):
        load_dotenv(_env_file, override=True)
        print(f"[INFO] Loaded environment from {_env_file}")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell env

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import optimizer components
from prompt_opt_env.server.actions import (
    count_tokens, ACTION_NAMES,
    add_context, shorten, add_example, rephrase, add_constraint
)
from openai import OpenAI
from rouge_score import rouge_scorer

# ── Environment variables (loaded from .env.local above) ─────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ALPHA: float = float(os.getenv("TOKEN_PENALTY_ALPHA", "0.02"))

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN not set. Add it to .env.local as HF_TOKEN=hf_...")

_CLIENT = None
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_client():
    """Lazy load OpenAI client."""
    global _CLIENT
    if _CLIENT is None and HF_TOKEN:
        _CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _CLIENT


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call LLM with the prompt and return output."""
    client = get_client()
    if not client:
        return "[Error: HF_TOKEN not configured]"
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3, timeout=30,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"[Error: {str(e)}]"


def score_output(output: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    if not output or not reference:
        return 0.0
    return round(_ROUGE.score(reference, output)["rougeL"].fmeasure, 4)


def intelligent_action_selection(current_prompt: str, current_score: float,
                                 tokens_remaining: int, step: int) -> int:
    """Select best action based on current state."""
    # Heuristic-based selection
    if current_score > 0.75 and tokens_remaining < 30:
        return 5  # STOP
    if "please" in current_prompt.lower() or "could you" in current_prompt.lower():
        return 1  # SHORTEN
    if current_prompt.endswith("?") or "can you" in current_prompt.lower():
        return 3  # REPHRASE
    if tokens_remaining > 40 and step < 4:
        return 0  # ADD_CONTEXT
    if current_score > 0.70:
        return 5  # STOP
    return 3  # Default REPHRASE


def optimize_prompt_with_state(state: dict) -> dict:
    """Optimize a prompt using RL-based action selection with state object."""

    # Extract from state
    task = state.get("task", "General")
    prompt = state.get("prompt", "")
    input_data = state.get("input", "")
    goal = state.get("goal", "Balanced")

    # Build full prompt with input
    full_prompt = prompt
    if input_data:
        full_prompt = f"{prompt}\n\n{input_data}"

    # Adjust token budget based on goal
    token_budget = 80
    if goal == "High Quality":
        token_budget = 100
    elif goal == "Low Cost":
        token_budget = 60

    # Run optimization using existing logic
    current_prompt = full_prompt
    current_output = call_llm(current_prompt)
    current_score = 0.5  # Default score
    current_tokens = count_tokens(current_prompt)

    # Store initial values
    initial_prompt = current_prompt
    initial_output = current_output
    initial_score = current_score
    initial_tokens = current_tokens

    # Optimization loop
    best_prompt = current_prompt
    best_output = current_output
    best_score = current_score
    best_tokens = current_tokens
    total_reward = 0.0

    max_steps = 7
    for step in range(max_steps):
        tokens_remaining = token_budget - current_tokens
        action_id = intelligent_action_selection(current_prompt, current_score, tokens_remaining, step)

        if action_id == 5:  # STOP
            stop_bonus = round(best_score * 1.5, 4)
            total_reward += stop_bonus
            break

        # Apply action
        new_prompt = current_prompt
        if action_id == 0 and input_data:
            new_prompt = add_context(current_prompt, input_data[:100])
        elif action_id == 1:
            new_prompt = shorten(current_prompt)
        elif action_id == 2:
            new_prompt = add_example(current_prompt, "Provide a clear, concise response.")
        elif action_id == 3:
            new_prompt = rephrase(current_prompt)
        elif action_id == 4:
            constraint = "Be concise and direct." if goal == "Low Cost" else "Provide a detailed response."
            new_prompt = add_constraint(current_prompt, constraint)

        if new_prompt == current_prompt:
            total_reward -= 0.1
            continue

        new_tokens = count_tokens(new_prompt)
        if new_tokens > token_budget:
            total_reward -= 0.5
            break

        new_output = call_llm(new_prompt)
        new_score = 0.5  # Placeholder - would use actual scoring

        quality_delta = new_score - current_score
        token_overhead = new_tokens - current_tokens
        step_reward = quality_delta - ALPHA * token_overhead
        step_reward = round(max(-2.0, min(2.0, step_reward)), 4)
        total_reward += step_reward

        if new_score > best_score or (new_score >= best_score - 0.05 and new_tokens < best_tokens):
            best_prompt = new_prompt
            best_output = new_output
            best_score = new_score
            best_tokens = new_tokens

        current_prompt = new_prompt
        current_output = new_output
        current_score = new_score
        current_tokens = new_tokens

        if best_score > 0.85:
            break

    # Calculate metrics
    improvement_pct = ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
    token_change = best_tokens - initial_tokens
    token_reduction_pct = (abs(token_change) / initial_tokens * 100) if initial_tokens > 0 and token_change < 0 else 0

    return {
        "initial_prompt": initial_prompt,
        "initial_output": initial_output,
        "initial_score": initial_score,
        "initial_tokens": initial_tokens,
        "final_prompt": best_prompt,
        "final_output": best_output,
        "final_score": best_score,
        "final_tokens": best_tokens,
        "improvement_pct": improvement_pct,
        "token_change": token_change,
        "token_reduction_pct": token_reduction_pct,
        "total_reward": total_reward,
    }


# Keep for backward compatibility
def optimize_prompt(initial_prompt: str, context: str = "", reference: str = "",
                   example: str = "", constraint: str = "", max_steps: int = 7) -> dict:
    """Optimize a prompt using RL-based action selection."""
    state = {
        "task": "General",
        "prompt": initial_prompt,
        "input": context,
        "goal": "Balanced"
    }
    return optimize_prompt_with_state(state)


# ── Ensure templates directory and HTML files exist ──────────────────────────
def _write_templates():
    """Write HTML templates to disk (always, so they stay in sync)."""
    tpl_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    _write_index_html(tpl_dir)
    _write_results_html(tpl_dir)


# Create FastAPI app
TPL_DIR = os.path.join(BASE_DIR, "templates")
app = FastAPI(title="PromptOptEnv Web")
templates = Jinja2Templates(directory=TPL_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Default prompts for each task
DEFAULT_PROMPTS = {
    "Summarization": "Summarize the following text concisely.",
    "Question Answering": "Answer the question based on the given context.",
    "Paraphrasing": "Paraphrase the following sentence.",
    "Instruction Following": "Follow the instruction and generate an appropriate response."
}


@app.post("/optimize", response_class=HTMLResponse)
async def optimize(
    request: Request,
    task: str = Form("Summarization"),
    input_text: str = Form(""),
    context: str = Form(""),
    question: str = Form(""),
    user_prompt: str = Form(""),
    prompt: str = Form(""),
    goal: str = Form("Balanced")
):
    """Handle optimization form submission with structured task input."""

    # Backward compatibility:
    # - accept legacy `prompt` field used by older UI versions
    # - default unknown/missing task to Summarization
    selected_task = task if task in DEFAULT_PROMPTS else "Summarization"
    provided_prompt = user_prompt.strip() if user_prompt.strip() else prompt.strip()

    # Construct state object
    state = {
        "task": selected_task,
        "input": "",
        "prompt": provided_prompt if provided_prompt else DEFAULT_PROMPTS[selected_task],
        "goal": goal
    }

    # Process input based on task
    if selected_task == "Question Answering":
        state["input"] = f"Context: {context}\nQuestion: {question}" if context or question else ""
    else:
        state["input"] = input_text

    # Pass state to optimizer (using existing backend logic)
    results = optimize_prompt_with_state(state)
    return templates.TemplateResponse("results.html", {
        "request": request,
        **results
    })


def _write_index_html(tpl_dir: str):
    """Write index.html into tpl_dir."""
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PromptOptEnv - RL Prompt Optimizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: rgba(255,255,255,0.8);
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        textarea, input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            background: white;
        }
        textarea:focus, input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea { min-height: 100px; resize: vertical; }
        .hint {
            font-size: 13px;
            color: #666;
            margin-top: 5px;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .optional-fields {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .optional-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PromptOptEnv</h1>
        <p class="subtitle">RL-Powered Prompt Optimization with Cost Awareness</p>

        <div class="card">
            <form action="/optimize" method="post">
                <div class="form-group">
                    <label for="task">Select Task</label>
                    <select name="task" id="task">
                        <option value="Summarization">Summarization</option>
                        <option value="Question Answering">Question Answering</option>
                        <option value="Paraphrasing">Paraphrasing</option>
                        <option value="Instruction Following">Instruction Following</option>
                    </select>
                </div>

                <div class="form-group" id="generic-input-group">
                    <label for="input_text" id="generic-input-label">Input Text</label>
                    <textarea name="input_text" id="input_text" placeholder="Paste the paragraph you want to summarize..."></textarea>
                </div>

                <div id="qa-input-group" class="hidden">
                    <div class="form-group">
                        <label for="context">Context</label>
                        <textarea name="context" id="context" placeholder="Enter context passage..."></textarea>
                    </div>

                    <div class="form-group">
                        <label for="question">Question</label>
                        <textarea name="question" id="question" placeholder="Enter your question..."></textarea>
                    </div>
                </div>

                <div class="optional-fields">
                    <p class="optional-title">Optional Configuration</p>

                    <div class="form-group">
                        <label for="user_prompt">Initial Prompt (Optional)</label>
                        <textarea
                            name="user_prompt"
                            id="user_prompt"
                            placeholder="e.g., Summarize the following text clearly and concisely"
                        ></textarea>
                        <p class="hint">Leave blank to use a task-specific default prompt</p>
                    </div>

                    <div class="form-group">
                        <label for="goal">Optimization Goal</label>
                        <select name="goal" id="goal">
                            <option value="Balanced" selected>Balanced</option>
                            <option value="High Quality">High Quality</option>
                            <option value="Low Cost">Low Cost</option>
                        </select>
                    </div>
                </div>

                <button type="submit">Optimize Prompt</button>
            </form>
        </div>
    </div>
    <script>
        const taskField = document.getElementById("task");
        const genericInputGroup = document.getElementById("generic-input-group");
        const genericInputLabel = document.getElementById("generic-input-label");
        const genericInput = document.getElementById("input_text");
        const qaInputGroup = document.getElementById("qa-input-group");

        const inputConfig = {
            "Summarization": {
                label: "Input Text",
                placeholder: "Paste the paragraph you want to summarize..."
            },
            "Paraphrasing": {
                label: "Sentence",
                placeholder: "Enter a sentence to paraphrase..."
            },
            "Instruction Following": {
                label: "Instruction",
                placeholder: "e.g., Explain photosynthesis in simple terms"
            }
        };

        function updateTaskInputs() {
            const selectedTask = taskField.value;

            if (selectedTask === "Question Answering") {
                genericInputGroup.classList.add("hidden");
                qaInputGroup.classList.remove("hidden");
                return;
            }

            qaInputGroup.classList.add("hidden");
            genericInputGroup.classList.remove("hidden");
            const config = inputConfig[selectedTask];
            genericInputLabel.textContent = config.label;
            genericInput.placeholder = config.placeholder;
        }

        taskField.addEventListener("change", updateTaskInputs);
        updateTaskInputs();
    </script>
</body>
</html>'''

    with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)


def _write_results_html(tpl_dir: str):
    """Write results.html into tpl_dir."""
    results_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Results - PromptOptEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 30px; font-size: 2em; }
        .card {
            background: white; border-radius: 16px; padding: 30px;
            margin-bottom: 25px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .section-title {
            color: #667eea; font-size: 18px; font-weight: 700; margin-bottom: 15px;
            text-transform: uppercase; letter-spacing: 0.5px;
            border-bottom: 2px solid #667eea; padding-bottom: 10px;
        }
        .prompt-box {
            background: #f8f8f8; padding: 15px; border-radius: 8px;
            border-left: 4px solid #667eea; margin-bottom: 15px;
            font-family: monospace; font-size: 14px; white-space: pre-wrap;
        }
        .output-box {
            background: #f8f8f8; padding: 15px; border-radius: 8px;
            border: 1px solid #ddd; min-height: 80px; color: #555;
            white-space: pre-wrap;
        }
        .metric { display: inline-block; margin-right: 20px; margin-top: 10px; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
        .metric-value { font-size: 20px; font-weight: 700; color: #333; }
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px; margin-top: 20px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 12px; text-align: center;
        }
        .metric-card .value { font-size: 32px; font-weight: 700; }
        .metric-card .label { font-size: 14px; opacity: 0.9; margin-top: 5px; }
        .back-btn {
            display: inline-block; margin-top: 20px; padding: 12px 30px;
            background: white; color: #667eea; text-decoration: none;
            border-radius: 8px; font-weight: 600; transition: transform 0.2s;
        }
        .back-btn:hover { transform: translateY(-2px); }
        .arrow { text-align: center; font-size: 30px; color: #667eea; margin: 20px 0; }
        @media (max-width: 768px) { .comparison { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Results</h1>

        <!-- Initial State -->
        <div class="card">
            <div class="section-title">Initial Prompt</div>
            <div class="prompt-box">{{ initial_prompt }}</div>

            <div class="section-title">Initial Output</div>
            <div class="output-box">{{ initial_output or "[No output generated]" }}</div>

            <div style="margin-top: 15px;">
                <span class="metric">
                    <div class="metric-label">Initial Reward</div>
                    <div class="metric-value">{{ "%.2f"|format(initial_score) }}</div>
                </span>
                <span class="metric">
                    <div class="metric-label">Token Count</div>
                    <div class="metric-value">{{ initial_tokens }}</div>
                </span>
            </div>
        </div>

        <!-- Arrow -->
        <div class="arrow">&#8595;</div>

        <!-- Optimized State -->
        <div class="card">
            <div class="section-title">Optimized Prompt</div>
            <div class="prompt-box" style="border-left-color: #4caf50;">{{ final_prompt }}</div>

            <div class="section-title">Optimized Output</div>
            <div class="output-box">{{ final_output or "[No output generated]" }}</div>

            <div style="margin-top: 15px;">
                <span class="metric">
                    <div class="metric-label">Final Reward</div>
                    <div class="metric-value">{{ "%.2f"|format(final_score) }}</div>
                </span>
                <span class="metric">
                    <div class="metric-label">Token Count</div>
                    <div class="metric-value">{{ final_tokens }}</div>
                </span>
            </div>
        </div>

        <!-- Metrics Summary -->
        <div class="card">
            <div class="section-title">Metrics Summary</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value">{{ "%.0f"|format(improvement_pct) }}%</div>
                    <div class="label">Improvement</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ "%.0f"|format(token_reduction_pct) }}%</div>
                    <div class="label">Token Reduction</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ "%.3f"|format(total_reward) }}</div>
                    <div class="label">Total Reward</div>
                </div>
            </div>

            {% if token_change < 0 %}
            <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 8px; color: #2e7d32;">
                <strong>Cost-Aware Success!</strong> Reduced tokens by {{ abs(token_change) }} while maintaining quality.
            </div>
            {% endif %}
        </div>

        <div style="text-align: center;">
            <a href="/" class="back-btn">&#8592; Optimize Another Prompt</a>
        </div>
    </div>
</body>
</html>'''

    with open(os.path.join(tpl_dir, "results.html"), "w", encoding="utf-8") as f:
        f.write(results_html)


if __name__ == "__main__":
    PORT = 5000
    URL = f"http://localhost:{PORT}"

    # Write/refresh templates from embedded HTML
    _write_templates()

    print("=" * 60)
    print("  PromptOptEnv Web Server Starting")
    print("=" * 60)
    print(f"\n  Open your browser to: {URL}")
    print(f"  Templates written to: {os.path.join(BASE_DIR, 'templates')}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser after a short delay so the server has time to start
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT)
