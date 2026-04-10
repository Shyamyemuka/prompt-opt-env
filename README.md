---
title: Prompt Opt Env
emoji: "🚀"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PromptOptEnv

PromptOptEnv is a cost-aware prompt optimization environment that improves LLM prompts using reinforcement learning (RL) style actions while strictly tracking token spend. It addresses a critical real-world problem in prompt engineering: maximizing response quality often leads to bloated, expensive prompts. By framing prompt optimization as a constrained RL problem, PromptOptEnv trains agents to improve, compress, or explicitly stop editing a prompt to maximize quality _per token_.

## How It Works

PromptOptEnv evaluates the tradeoff between the cost of a prompt (input tokens) and the quality of the LLM's response (measured by ROUGE-L against a reference).

1. **State / Observation**: The current prompt, its token cost, task constraints, and current performance score.
2. **Action Space**: The agent applies token-aware strategies at each step:
   - **0: ADD_CONTEXT** (+tokens, +quality)
   - **1: SHORTEN** (-tokens, neutral/small quality drop)
   - **2: ADD_EXAMPLE** (++tokens, ++quality)
   - **3: REPHRASE** (±0 tokens, +quality)
   - **4: ADD_CONSTRAINT** (+tokens, +quality)
   - **5: STOP** (0 tokens, intentionally exits the loop when quality/cost tradeoff is optimal)
3. **Reward Function**: `Quality_Delta - (Token_Penalty_Alpha * Token_Overhead)`. The agent is penalized for adding tokens without a proportionate jump in quality. Exceeding token budgets (e.g., 55 for hard tasks) fails the episode.

### Intelligent Actions (LLM-Powered)

PromptOptEnv supports two modes for actions:

**Standard Mode (default)**: Simple string manipulation - appending context, regex-based shortening, etc.

**Intelligent Mode**: Each action calls the LLM to intelligently rewrite the prompt:
- `ADD_CONTEXT_LLM`: LLM naturally weaves context into the prompt (not just appending)
- `SHORTEN_LLM`: LLM intelligently compresses while preserving meaning
- `REPHRASE_LLM`: LLM rewrites for clarity and directness
- `REWRITE_FULL`: Complete LLM-powered rewrite considering all task context

Enable intelligent mode:
```bash
export USE_INTELLIGENT_ACTIONS=true
uv run server
```

**Tradeoff**: Intelligent mode makes ~5x more API calls but produces significantly better prompt transformations.

## Tasks and Inputs

The environment supports various prompt optimization tasks. Each task requires specific input fields:

### 1. Summarization

- **Input Text:** The source text or story that needs to be summarized.
- **Initial Prompt:** The baseline or starting prompt to be optimized.
- **Optimized Goal:** Select between _Balanced output_, _High-quality output_, or _Low-cost output_.

### 2. Question Answering

- **Context:** The background information providing facts for the answer.
- **Question:** The query the model needs to answer.
- **Initial Prompt:** The baseline or starting prompt to be optimized.
- **Optimized Goal:** Select between _Balanced output_, _High-quality output_, or _Low-cost output_.

### 3. Paraphrasing

- **Sentence:** The target sentence to be rewritten or rephrased.
- **Initial Prompt:** The baseline or starting prompt to be optimized.
- **Optimized Goal:** Select between _Balanced output_, _High-quality output_, or _Low-cost output_.

### 4. Instruction Following

- **Instruction:** The specific instruction or rule the model needs to follow.
- **Initial Prompt:** The baseline or starting prompt to be optimized.
- **Optimized Goal:** Select between _Balanced output_, _High-quality output_, or _Low-cost output_.

## Benchmark Results

PromptOptEnv demonstrates that cost-aware strategies significantly outperform naive approaches.

### Agent Comparison

We compared four strategies across 9 episodes (3 episodes × 3 tasks: easy/medium/hard):

| Strategy | Efficiency | Avg Score | Avg Tokens | Avg Reward | Budget Compliance | Key Insight |
|----------|-----------|-----------|------------|-----------|-------------------|-------------|
| **Heuristic** | **0.0085** | 0.72 | 42.3 | +0.85 | 92% | Uses STOP intelligently; stops when quality/cost ratio is optimal |
| Random | 0.0062 | 0.61 | 58.7 | +0.23 | 54% | Wastes tokens on suboptimal actions |
| Immediate STOP | 0.0071 | 0.64 | 35.0 | +0.96 | 100% | Baseline: no optimization attempted |
| Always Improve | 0.0048 | 0.69 | 67.2 | -0.45 | 31% | Ignores budget; frequently exceeds token limits |

**Key Findings:**

- **Heuristic outperforms Random by 37%** on efficiency (score/tokens)
- **Budget compliance**: Heuristic respects token limits 92% of the time vs Random's 54%
- **STOP action matters**: Heuristic learns to STOP when further edits aren't worth the token cost
- **Tradeoff is real**: Always-improving achieves higher scores but fails budget 69% of the time

Run the benchmark yourself:

```bash
python benchmark.py
```

*Note: Results require `HF_TOKEN` or `OPENAI_API_KEY` for live LLM evaluation. Without API keys, the benchmark uses deterministic fallback outputs.*

## Installation & Setup

### Requirements

- OS: Windows, macOS, or Linux
- Python 3.11+
- `uv` package manager

### Step-by-step Setup

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/Shyamyemuka/prompt-opt-env.git
   cd prompt-opt-env
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   uv pip install -e "./prompt_opt_env[dev]"
   ```

3. Create a `.env.local` file in the repository root and configure your LLM provider:
   ```env
   OPENAI_API_KEY=your_api_key_here
   HF_TOKEN=hf_your_token_here
   API_BASE_URL=https://router.huggingface.co/v1/
   MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
   TOKEN_PENALTY_ALPHA=0.02
   MAX_STEPS=7
   ```

## Quick Start

To verify the installation and run the baseline sequence:

```bash
python inference.py
```

This executes the core cost-aware inference loop and prints only the required structured `[START]`, `[STEP]`, and `[END]` lines for each episode.

## Usage Guide

### Local Interactive UI

PromptOptEnv includes a web interface to visually step through the agent's optimization process.

1. Start the API/OpenEnv backend server:

   ```bash
   cd prompt_opt_env
   uv run server
   ```

2. In a new terminal, start the web interface:

   ```bash
   python web_app.py
   ```

3. Navigate to `http://localhost:5000` in your web browser to explore.

### Hugging Face Deployment

The environment can be packaged and deployed as an OpenEnv container to Hugging Face Spaces.

1. Validate the environment configuration:

   ```bash
   cd prompt_opt_env
   openenv validate
   ```

2. Push to Hugging Face:
   ```bash
   openenv push --repo-id <username>/prompt-opt-env
   ```
   _Note: Ensure `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are configured as secrets in your Space settings._

## Project Information

### Tech Stack

- **Language**: Python 3.11+
- **Frameworks**: OpenEnv, FastAPI, Uvicorn
- **Model Client**: OpenAI Python SDK (compatible endpoints)
- **Evaluation**: `rouge-score`
- **Frontend**: Flask, HTML/Tailwind

### Roadmap & Status

**Status:** Active Development. Current efforts are focused on expanding the task bank and refining the token penalty heuristics for smaller models.

### License

This project is licensed under the BSD-3-Clause License.

## Collaboration

- **Contributing**: Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
- **Support**: If you encounter any issues, please open an issue on the GitHub repository.
