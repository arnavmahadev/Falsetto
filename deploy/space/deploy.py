#!/usr/bin/env python
"""Push FALSETTO Studio to a Hugging Face Space.

Prereqs (one time):
    pip install -U huggingface_hub
    hf auth login            # paste a WRITE token from https://hf.co/settings/tokens

Then:
    python deploy/space/deploy.py <your-hf-username>/falsetto-studio

It creates the Space (if needed), stages app.py + requirements + README + the
pre-built demo assets, and uploads. Assets are built automatically if missing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder, whoami

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
ASSETS = REPO_ROOT / "demo_assets"
SPACE_FILES = ["app.py", "requirements.txt", "README.md", ".gitattributes"]


def _ensure_assets() -> None:
    if (ASSETS / "fusion_demo.pt").exists():
        return
    print("demo_assets missing — building them (downloads MERT, ~2 min)...")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_demo.py")], check=True)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python deploy/space/deploy.py <hf-username>/falsetto-studio", file=sys.stderr)
        return 2
    repo_id = argv[0]

    try:
        user = whoami()["name"]
    except Exception:
        print("Not logged in. Run:  hf auth login   (needs a WRITE token)", file=sys.stderr)
        return 1
    print(f"Logged in as {user}. Target Space: {repo_id}")

    _ensure_assets()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for name in SPACE_FILES:
            shutil.copy2(HERE / name, staging / name)
        shutil.copytree(ASSETS, staging / "demo_assets")

        create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        print("Uploading (LFS handles the model + wavs)...")
        upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=str(staging),
            commit_message="Deploy FALSETTO Studio",
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDone. Building now — watch it come up at:\n  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
