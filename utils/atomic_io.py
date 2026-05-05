"""Atomic file write utilities: write to .tmp then rename.

Prevents corrupt files from interrupted writes (OOM kill, SIGKILL, wall-time).

Both helpers accept the legacy ``(data, path)`` argument order used throughout
the original repo (training.py, legacy/*) AND the newer ``(path, data)`` order
adopted by the cka_similarity workers and bin/calibrate.py. The two are
disambiguated by detecting which positional argument is path-like (str /
``os.PathLike``); a dict / tensor / other Python object is treated as data.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _split_path_data(a: Any, b: Any) -> "tuple[Path, Any]":
    """Resolve (path, data) regardless of which positional order the caller used.

    Returns ``(Path, data)``. Raises TypeError if neither argument looks
    path-like, or if both do (ambiguous).
    """
    a_is_path = isinstance(a, (str, os.PathLike))
    b_is_path = isinstance(b, (str, os.PathLike))
    if a_is_path and not b_is_path:
        return Path(a), b
    if b_is_path and not a_is_path:
        return Path(b), a
    if a_is_path and b_is_path:
        # Both look like paths; assume modern (path, data) order with a string payload.
        return Path(a), b
    raise TypeError(
        "atomic_io: could not identify a path argument; expected one of the two "
        "positional args to be str or os.PathLike."
    )


def atomic_torch_save(arg1, arg2):
    """Save a PyTorch object atomically via tmp+rename.

    Accepts both ``(obj, path)`` (legacy) and ``(path, obj)`` (new) orders.
    """
    path, obj = _split_path_data(arg1, arg2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_json_dump(arg1, arg2, indent: int = 2):
    """Write JSON atomically via tmp+rename.

    Accepts both ``(data, path)`` (legacy) and ``(path, data)`` (new) orders.
    """
    path, data = _split_path_data(arg1, arg2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_json_load(path, default=None):
    """Read a JSON file, returning ``default`` if it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)
