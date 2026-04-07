# PromptOptEnv ⚡️

**A Reinforcement Learning-powered Prompt Optimization Environment built for the Meta x Scaler OpenEnv Hackathon.**

---

## 🌟 Overview

Writing the perfect prompt is difficult. `PromptOptEnv` eliminates the guesswork by turning your basic, naive instructions into highly optimized, task-specific expert prompts that extract maximum performance and cost efficiency from Language Models.

By placing an LLM inside an **OpenEnv Reinforcement Learning loop**, the environment learns to iteratively apply prompting best practices (like providing context, appending few-shot examples, or adding system constraints) to maximize a specific objective function (like ROUGE string matching against a minimized token-cost baseline).

![Web UI Demo](placeholder_for_demo.gif)

## ✨ Key Features

- **Dynamic Task Architecture**: Engineered with tailored state-handling and grading mechanisms for `Summarization`, `Paraphrasing`, `Question Answering`, and `Instruction Following`.
- **Cost-Aware Reinforcement Learning**: The RL reward function inherently penalizes token bloat using a tunable Alpha penalty. "Better" prompts are only accepted if the quality delta outweighs the generation cost.
- **OpenEnv Native**: Built to comply entirely with the OpenEnv validation specification and is 1-click deployable to HuggingFace containers.
- **Beautiful Web Playground**: A next-generation, dark-themed `TailwindCSS` web interface that visually breaks down prompt permutations in a sleek "Before vs. After" layout.

---

## 🛠️ System Architecture

1. **State**: The current permutation of the prompt, the task definition, the input data, and the running token count.
2. **Action Space**:
   - `0`: Provide Context
   - `1`: Shorten Prompt
   - `2`: Add Example Structure
   - `3`: Rephrase for Clarity
   - `4`: Add System Constraint
   - `5`: STOP
3. **Reward Function**: Calculates the ROUGE-L semantic overlap against an expert generated baseline, offset by the token length overhead (`Reward = QualityDelta - (Alpha * TokenOverhead)`).

---

## 🚀 Running the Project

Please refer to the [RUN.md](RUN.md) file for a complete, step-by-step sequential guide on how to setup your environment variables, test the baseline script, run the Web UI, and deploy the OpenEnv container!

---

## 📸 Screenshots

*(Hackathon tip: Swap out these placeholder images with actual screenshots or GIFs before submitting!)*

### Landing Page
![Landing Page Details](placeholder_landing.png)

### Optimizer Results
![Optimizer Before/After](placeholder_results.png)
