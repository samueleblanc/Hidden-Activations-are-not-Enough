"""Resume / state checkpointing for the km-feature-viz pipeline.

Each step writes its set of completed (model, class_id, image_id) keys to
state_path(step_name). On restart, the script loads the set and skips
already-done entries.

Failures append to errors.json with traceback for post-hoc inspection.
"""
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Set


def load_completed(state_file: Path) -> Set[str]:
    """Return the set of completed sample keys, or empty set if file missing."""
    if not state_file.exists():
        return set()
    with state_file.open() as f:
        data = json.load(f)
    return set(data)


def save_completed(state_file: Path, keys: Set[str]) -> None:
    """Atomic write of the completed-keys set."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=state_file.parent, suffix=".tmp"
    ) as tmp:
        json.dump(sorted(keys), tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, state_file)


def mark_completed(state_file: Path, key: str) -> None:
    """Add key to the completed set and persist."""
    keys = load_completed(state_file)
    keys.add(key)
    save_completed(state_file, keys)


def log_error(
    errors_file: Path,
    step: str,
    sample_id: str,
    error_type: str,
    message: str,
    tb: str = "",
) -> None:
    """Append one error entry to errors.json. Atomic per-write."""
    errors_file.parent.mkdir(parents=True, exist_ok=True)
    if errors_file.exists():
        with errors_file.open() as f:
            data = json.load(f)
    else:
        data = []
    data.append(
        {
            "step": step,
            "sample_id": sample_id,
            "error_type": error_type,
            "message": message,
            "traceback": tb,
        }
    )
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=errors_file.parent, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, errors_file)


def capture_traceback() -> str:
    """Convenience: return current exception traceback as a string."""
    return traceback.format_exc()
