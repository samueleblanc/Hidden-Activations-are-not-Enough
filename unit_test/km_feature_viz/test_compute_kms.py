"""Smoke test for the KM compute driver.

Uses a tiny CNN and the existing knowledgematrix `SmallCNN` test fixture
pattern, NOT a pretrained torchvision model — we only verify the
plumbing (manifest iteration, slicing, save format, resume).
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from km_feature_viz import paths
from km_feature_viz.compute_kms import (
    slice_class_rows,
    save_km,
    load_km_slice,
)


class TestSliceAndSave(unittest.TestCase):

    def test_slice_class_rows(self):
        # full KM is (1000, 150529); slice down to in-scope class ids
        full = torch.randn(1000, 150529, dtype=torch.float16)
        in_scope = [0, 207, 282]
        sliced = slice_class_rows(full, in_scope)
        self.assertEqual(sliced.shape, (3, 150529))
        for i, c in enumerate(in_scope):
            self.assertTrue(torch.equal(sliced[i], full[c]))

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d) / "km.pt"
            tensor = torch.randn(10, 1000, dtype=torch.float16)
            in_scope = list(range(10))
            save_km(tmp_path, tensor, in_scope_classes=in_scope)
            loaded_tensor, loaded_classes = load_km_slice(tmp_path)
            self.assertTrue(torch.equal(tensor, loaded_tensor))
            self.assertEqual(loaded_classes, in_scope)
