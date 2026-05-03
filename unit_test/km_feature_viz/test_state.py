"""Tests for state checkpointing helpers."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from km_feature_viz import state


class TestState(unittest.TestCase):

    def test_load_state_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            state_file = Path(d) / "missing.json"
            self.assertEqual(state.load_completed(state_file), set())

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            state_file = Path(d) / "s.json"
            keys = {"a", "b", "c"}
            state.save_completed(state_file, keys)
            self.assertEqual(state.load_completed(state_file), keys)

    def test_mark_completed_appends(self):
        with tempfile.TemporaryDirectory() as d:
            state_file = Path(d) / "s.json"
            state.save_completed(state_file, {"a"})
            state.mark_completed(state_file, "b")
            self.assertEqual(state.load_completed(state_file), {"a", "b"})

    def test_mark_completed_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            state_file = Path(d) / "s.json"
            state.mark_completed(state_file, "a")
            state.mark_completed(state_file, "a")
            self.assertEqual(state.load_completed(state_file), {"a"})

    def test_log_error_appends(self):
        with tempfile.TemporaryDirectory() as d:
            errors_file = Path(d) / "errors.json"
            state.log_error(errors_file, step="01", sample_id="x", error_type="OOM", message="cuda")
            state.log_error(errors_file, step="02", sample_id="y", error_type="NaN", message="bad")
            data = json.loads(errors_file.read_text())
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["step"], "01")
            self.assertEqual(data[1]["sample_id"], "y")


if __name__ == "__main__":
    unittest.main()
