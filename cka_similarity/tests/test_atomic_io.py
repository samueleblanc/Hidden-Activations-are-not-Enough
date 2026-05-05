import json, os, tempfile
import pytest
from utils.atomic_io import atomic_json_dump


def test_atomic_json_dump_creates_file(tmp_path):
    path = tmp_path / "out.json"
    atomic_json_dump(str(path), {"distances": [1.0, 2.0]})
    assert path.exists()
    assert json.loads(path.read_text()) == {"distances": [1.0, 2.0]}


def test_atomic_json_dump_overwrites_atomically(tmp_path):
    path = tmp_path / "out.json"
    atomic_json_dump(str(path), {"distances": [1.0]})
    atomic_json_dump(str(path), {"distances": [1.0, 2.0]})
    assert json.loads(path.read_text()) == {"distances": [1.0, 2.0]}


def test_atomic_json_dump_no_partial_on_failure(tmp_path, monkeypatch):
    """If write fails, the destination file must remain unchanged (or absent)."""
    path = tmp_path / "out.json"
    atomic_json_dump(str(path), {"a": 1})
    original = path.read_text()

    # Simulate a write error after the temp file is created
    real_replace = os.replace
    def fake_replace(*args, **kwargs):
        raise IOError("simulated failure")
    monkeypatch.setattr("os.replace", fake_replace)

    with pytest.raises(IOError):
        atomic_json_dump(str(path), {"a": 99})
    assert path.read_text() == original
