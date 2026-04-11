"""
Push a curated PromptOptEnv Space bundle to Hugging Face.

This keeps the Space repo clean by staging only the files required for the app
and by deleting remote junk patterns in the same commit. Local files are never
deleted.

Usage:
    python push_ui_to_hf.py
    python push_ui_to_hf.py --repo-id your-name/prompt-opt-env
    python push_ui_to_hf.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

BASE_DIR = Path(__file__).resolve().parent
ENV_DIR = BASE_DIR / "prompt_opt_env"
DEFAULT_REPO_ID = "shyamyemuka/prompt-opt-env"
REPO_TYPE = "space"

REMOTE_DELETE_PATTERNS = [
    "*.log",
    "*.txt",
    ".pytest_cache",
    ".pytest_cache/**",
    "__pycache__",
    "**/__pycache__",
    ".venv",
    ".venv/**",
    ".uvcache2",
    ".uvcache2/**",
    ".uv-cache",
    ".uv-cache/**",
    # Clean stale mirrored package tree from older deployment mode.
    # If this lingers, evaluators may import outdated grader paths.
    "prompt_opt_env",
    "prompt_opt_env/**",
]

DEPLOY_FILE_MAP = {
    BASE_DIR / "README.md": "README.md",
    BASE_DIR / "benchmark.py": "benchmark.py",
    BASE_DIR / "demo_examples.py": "demo_examples.py",
    BASE_DIR / "inference.py": "inference.py",
    BASE_DIR / "optimize.py": "optimize.py",
    ENV_DIR / ".dockerignore": ".dockerignore",
    ENV_DIR / "client.py": "client.py",
    ENV_DIR / "Dockerfile.ui": "Dockerfile",
    ENV_DIR / "llm_router.py": "llm_router.py",
    ENV_DIR / "models.py": "models.py",
    ENV_DIR / "openenv.yaml": "openenv.yaml",
    ENV_DIR / "pyproject.toml": "pyproject.toml",
    ENV_DIR / "uv.lock": "uv.lock",
    ENV_DIR / "web_ui.py": "web_ui.py",
}

DEPLOY_DIRS = {
    ENV_DIR / "server": "server",
    ENV_DIR / "templates": "templates",
}


def _load_env_local() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = BASE_DIR / ".env.local"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"[INFO] Loaded {env_file}")


def stage_hf_bundle(staging_dir: Path) -> list[str]:
    staged: list[str] = []
    staging_dir.mkdir(parents=True, exist_ok=True)

    for source, destination in DEPLOY_FILE_MAP.items():
        if not source.exists():
            continue
        target = staging_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged.append(destination.replace("\\", "/"))

    for source, destination in DEPLOY_DIRS.items():
        if not source.exists():
            continue
        target = staging_dir / destination
        shutil.copytree(source, target, dirs_exist_ok=True)
        staged.append(destination.replace("\\", "/") + "/")

    return sorted(staged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_local()

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN not found. Add it to the environment or .env.local.")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "hf_space"
        staged = stage_hf_bundle(staging_dir)

        print(f"[INFO] Prepared {len(staged)} deploy entries for {args.repo_id}:")
        for item in staged:
            print(f"  - {item}")

        if args.dry_run:
            print("[INFO] Dry run complete. No remote changes made.")
            return 0

        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            delete_patterns=REMOTE_DELETE_PATTERNS,
            commit_message="Sync PromptOptEnv Space bundle",
        )

    print(f"[DONE] Space synced: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
