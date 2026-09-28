"""Colab helpers: find a persistent place for checkpoints so runs survive disconnects.

On Colab, outputs go to Google Drive (``MyDrive/stego-runs/<name>``); anywhere else
they go to ``./runs/<name>``. Training scripts resume from the latest checkpoint there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DRIVE_ROOT = Path("/content/drive")


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def mount_drive() -> Path:
    """Mount Google Drive (asks for permission the first time) and return MyDrive."""
    from google.colab import drive  # only importable on Colab

    if not (DRIVE_ROOT / "MyDrive").exists():
        drive.mount(str(DRIVE_ROOT))
    return DRIVE_ROOT / "MyDrive"


def run_dir(name: str, root: str | Path | None = None) -> Path:
    """Directory for one experiment's checkpoints and outputs, created if missing."""
    if root is None:
        root = mount_drive() / "stego-runs" if in_colab() else Path("runs")
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_checkpoint(path: str | Path) -> Path | None:
    """Newest ``checkpoint-<step>`` folder (the Hugging Face Trainer layout), if any."""
    steps = []
    for child in Path(path).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if m and child.is_dir():
            steps.append((int(m.group(1)), child))
    return max(steps)[1] if steps else None
