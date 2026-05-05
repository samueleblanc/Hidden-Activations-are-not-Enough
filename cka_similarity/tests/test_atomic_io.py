import json, os, tempfile
import torch
import pytest
from utils.atomic_io import atomic_json_dump, atomic_torch_save


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


def test_atomic_json_dump_rejects_ambiguous_str_str(tmp_path):
    """When both positional args are str/PathLike, refuse rather than guess.

    A legacy caller passing ``atomic_json_dump('payload_text', '/path/file.json')``
    used to silently create ``./payload_text`` with ``/path/file.json`` as content.
    That silent-failure mode is now a hard TypeError.
    """
    path = tmp_path / "out.json"
    with pytest.raises(TypeError, match="ambiguous"):
        # Both args are str -- neither is unambiguously data.
        atomic_json_dump(str(path), "some-string-payload")
    # And neither file should have been created.
    assert not path.exists()
    assert not (tmp_path / "some-string-payload").exists()


def test_atomic_torch_save_path_data_order(tmp_path):
    """Pin the modern (path, tensor) ordering for atomic_torch_save."""
    path = tmp_path / "tensor.pt"
    tensor = torch.randn(4, 4)
    # New (path, data) order:
    atomic_torch_save(str(path), tensor)
    assert path.exists()
    loaded = torch.load(str(path), weights_only=True)
    assert torch.equal(tensor, loaded)

    # And the legacy (data, path) order must still work for backward compat:
    path2 = tmp_path / "tensor2.pt"
    atomic_torch_save(tensor, str(path2))
    assert path2.exists()
    loaded2 = torch.load(str(path2), weights_only=True)
    assert torch.equal(tensor, loaded2)
