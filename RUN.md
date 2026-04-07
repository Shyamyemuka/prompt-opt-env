# 🚀 Running the Application (Sequential Guide)

This guide provides a step-by-step, sequential workflow to set up, run, and test the `prompt-opt-env` application. Follow these steps in order to get the project running locally and understand how the different components fit together.

---

## Step 1: Install Dependencies
**Description:** Set up an isolated Python virtual environment and install the required packages using `uv`. This ensures your system Python remains clean and the application has exactly what it needs.

```bash
uv venv
uv pip install -e "./prompt_opt_env[dev]"
```

## Step 2: Configure Environment Variables
**Description:** Create a configuration file named `.env.local` in the root directory. This file stores your API keys and configuration settings so the application can communicate with the AI models.

```bash
# Create a .env.local file in the project root and add the following:
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1/
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TOKEN_PENALTY_ALPHA=0.02
MAX_STEPS=7
DONE_THRESHOLD=0.85
LLM_TIMEOUT_SECONDS=10
```

## Step 3: Run the OpenEnv API Server (Backend)
**Description:** Start the core reinforcement learning environment server. This API acts as the brain/backend for the prompt optimization process. **You should leave this terminal running.**

```bash
cd prompt_opt_env
uv run server
```
*Health Check:* You can verify the backend is running by executing `curl http://localhost:8000/health` in a new terminal window.

## Step 4: Run the Web UI (Prompt Optimizer)
**Description:** Start the user-friendly web interface. This communicates with the API server created in Step 3 so you can visually interact with the optimizer. **Open a new terminal for this step.**

```bash
python web_app.py
```
*Access:* Open your web browser and navigate to `http://localhost:5000` to use the application.

## Step 5: Test the Baseline Inference Script
**Description:** Run the mandatory OpenEnv script. This executes a programmatic reinforcement learning loop against the environment to verify everything is working under the hood. **You can run this in a new terminal window.**

```bash
python inference.py
```

---

# 🐳 Alternative: Running with Docker

If you prefer to run the backend API server using Docker instead of natively via `uv`, you can follow these sequential steps (This acts as an alternative to **Step 3** above).

## 1. Build the Docker Image
**Description:** Build the container image for the API server based on your current code.

```bash
docker build -t prompt-opt-env-web:latest -f prompt_opt_env/server/Dockerfile prompt_opt_env
```

## 2. Run the Docker Container
**Description:** Start the backend server inside a Docker container in the background, injecting your environment variables.

```powershell
docker run -d --name prompt-opt-env-web-run -p 8000:8000 `
  -e API_BASE_URL=https://router.huggingface.co/v1/ `
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct `
  -e HF_TOKEN=hf_your_token_here `
  -e GRADER=rouge `
  prompt-opt-env-web:latest
```

## 3. Verify the Container Logs
**Description:** Check the Docker logs to ensure the API server started correctly without any errors.

```powershell
docker logs --tail 40 prompt-opt-env-web-run
```

## 4. Stop and Clean Up The Container
**Description:** Once you are done testing with Docker, you can stop and clean up the container with this command.

```bash
docker rm -f prompt-opt-env-web-run
```

---

# 🚀 Deployment: Validating and Deploying (OpenEnv Submit)

This section is for when you are fully done and ready to submit to the Hackathon.

## 1. Validate and Push to HuggingFace
**Description:** Use the `openenv` CLI to validate your environment setup and push it to a HuggingFace Space.

```bash
cd prompt_opt_env
openenv validate
openenv push --repo-id <hf-username>/prompt-opt-env
```
