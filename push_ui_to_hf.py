"""
push_ui_to_hf.py
----------------
After running `openenv push`, this script pushes our custom Dark UI Dockerfile
directly to the Hugging Face Space repo, overriding the one openenv injected.

Usage:
    python push_ui_to_hf.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env.local for HF_TOKEN
try:
    from dotenv import load_dotenv
    env_file = os.path.join(BASE_DIR, ".env.local")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"[INFO] Loaded {env_file}")
except ImportError:
    pass

from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
REPO_ID = "shyamyemuka/prompt-opt-env"
REPO_TYPE = "space"

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not found. Add it to .env.local")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

print(f"[INFO] Uploading files to {REPO_ID}...")

# 1. Upload our custom Dockerfile (overrides what openenv injected)
dockerfile_src = os.path.join(BASE_DIR, "prompt_opt_env", "Dockerfile.ui")
upload_file(
    path_or_fileobj=dockerfile_src,
    path_in_repo="Dockerfile",        # HF reads this as the root Dockerfile
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    token=HF_TOKEN,
    commit_message="Override Dockerfile with Custom Dark UI",
)
print("[OK] Dockerfile uploaded")

# 2. Upload web_ui.py (our full UI logic)
web_ui_src = os.path.join(BASE_DIR, "prompt_opt_env", "web_ui.py")
upload_file(
    path_or_fileobj=web_ui_src,
    path_in_repo="web_ui.py",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    token=HF_TOKEN,
    commit_message="Upload Custom Dark UI (web_ui.py)",
)
print("[OK] web_ui.py uploaded")

# 3. Upload the templates folder if it exists
import glob
templates_dir = os.path.join(BASE_DIR, "templates")
if os.path.isdir(templates_dir):
    for fpath in glob.glob(os.path.join(templates_dir, "*.html")):
        fname = os.path.basename(fpath)
        upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"templates/{fname}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            commit_message=f"Upload template: {fname}",
        )
        print(f"[OK] templates/{fname} uploaded")

print(f"\n[DONE] All files pushed. Hugging Face will rebuild the Space now.")
print(f"       Visit: https://huggingface.co/spaces/{REPO_ID}")
