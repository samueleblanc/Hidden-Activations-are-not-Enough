"""Tests for KM dictionary / PCA / NMF."""
import tempfile
import unittest
from pathlib import Path

import torch

from km_feature_viz.dictionary import (
    stack_class_rows,
    pca_top_k,
    nmf_top_k,
)


class TestDictionary(unittest.TestCase):

    def test_stack_class_rows_shape(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # 3 fake KMs each with 5 in-scope classes, 100-dim
            for i in range(3):
                fpath = tmp / f"sample_{i}.pt"
                torch.save(
                    {"km": torch.randn(5, 100, dtype=torch.float16), "classes": [0, 1, 2, 3, 4]},
                    fpath,
                )
            stacked = stack_class_rows(
                km_paths=sorted(tmp.glob("*.pt")),
                target_class_idx_in_slice=2,
            )
            self.assertEqual(stacked.shape, (3, 100))

    def test_pca_top_k_returns_components(self):
        X = torch.randn(20, 50)
        components, explained = pca_top_k(X, k=5)
        self.assertEqual(components.shape, (5, 50))
        self.assertEqual(explained.shape, (5,))

    def test_nmf_top_k_returns_components(self):
        X = torch.rand(20, 50).abs()
        components, _ = nmf_top_k(X, k=5)
        self.assertEqual(components.shape, (5, 50))
        self.assertTrue((components >= 0).all())
