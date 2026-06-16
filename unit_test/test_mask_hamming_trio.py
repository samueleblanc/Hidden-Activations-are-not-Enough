"""Tests for mask_hamming_trio.py (Plan B / D4).

Two locally-runnable test classes:

  * TestReluCoverage — the KEY correctness gate (Critic 4's flag). For each
    trio arch it builds the KM wrapper, runs a save=True forward on a small
    3D ImageNet-shaped input, and asserts EVERY nn.ReLU in model.layers has a
    collected sign pattern (no silently-undercounted branch ReLUs). The local
    Mac has internet + cached torchvision weights, so build_model can fetch
    DEFAULT weights offline. This is the gate that decides which archs are
    supported.

  * TestPerSampleExtraction — synthetic per-sample smoke. Builds ONE model
    once, fabricates two random 3D "endpoint" tensors, runs the per-sample
    extraction, and asserts finite d_f/d_M/A and integer relu/maxpool counts
    in sane ranges. No pairs.pth needed.

  * TestPartialSpearman — unit test of the size-controlled crossing-mechanism
    statistic against an independent reference (residualize-then-correlate).

These run on CPU under the repo's env/ (Python 3.11, torch 2.6.0). The full
per-(arch,attack) run needs the A2 pairs.pth on the cluster and is not
exercised here.
"""
import math
import unittest

import numpy as np
import torch
from torch import nn

import mask_hamming_trio as mh

# The trio. resnet152 is the heaviest build (~150 ReLUs); keep CPU.
TRIO = ["resnet152", "densenet121", "googlenet"]


class TestReluCoverage(unittest.TestCase):
    """Branch-ReLU coverage gate — the headline correctness check."""

    def test_every_relu_covered_per_arch(self):
        for arch in TRIO:
            with self.subTest(arch=arch):
                model = mh.build_model(arch, "cpu")
                report = mh.check_relu_coverage(model, input_shape=(3, 224, 224))
                # Must have ReLUs and every one must be covered.
                self.assertGreater(report["n_relu"], 0, f"{arch}: no ReLUs found")
                self.assertEqual(
                    report["n_relu"], report["n_relu_covered"],
                    f"{arch}: {report['n_relu'] - report['n_relu_covered']} "
                    f"uncovered ReLU(s) — silent undercount risk",
                )
                self.assertTrue(report["full_coverage"], f"{arch}: not fully covered")
                # Maxpool indices must also be fully populated (used by the
                # maxpool-mismatch statistic).
                self.assertEqual(
                    report["n_maxpool"], report["n_maxpool_covered"],
                    f"{arch}: uncovered maxpool layer(s)",
                )
                del model


class TestPerSampleExtraction(unittest.TestCase):
    """Synthetic per-sample smoke on one real model."""

    @classmethod
    def setUpClass(cls):
        # googlenet is the smallest/fastest trio build and exercises both
        # branch ReLUs AND maxpools — good single-model smoke target.
        cls.model = mh.build_model("googlenet", "cpu")
        cls.kmc = mh.make_kmc(cls.model, device="cpu", batch_size=4096)

    def test_per_sample_returns_sane_record(self):
        torch.manual_seed(0)
        # Fabricated 3D endpoints (NOT a real attack — just two points).
        x_clean = torch.randn(3, 224, 224)
        x_adv = x_clean + 0.05 * torch.randn(3, 224, 224)
        rec = mh.per_sample_record(self.kmc, self.model, x_clean, x_adv,
                                    device="cpu", idx=7)
        # Distances finite and non-negative.
        for k in ("d_f", "d_M", "A"):
            self.assertIn(k, rec)
            self.assertTrue(math.isfinite(rec[k]), f"{k} not finite: {rec[k]}")
            self.assertGreaterEqual(rec[k], 0.0, f"{k} negative")
        # Hamming / pattern integers with sane ranges.
        self.assertIsInstance(rec["relu_pattern_len"], int)
        self.assertGreater(rec["relu_pattern_len"], 0)
        self.assertIsInstance(rec["relu_hamming"], int)
        self.assertGreaterEqual(rec["relu_hamming"], 0)
        self.assertLessEqual(rec["relu_hamming"], rec["relu_pattern_len"])
        # Maxpool counts (googlenet has maxpools).
        self.assertIsInstance(rec["maxpool_total_positions"], int)
        self.assertGreater(rec["maxpool_total_positions"], 0)
        self.assertIsInstance(rec["maxpool_mismatches"], int)
        self.assertGreaterEqual(rec["maxpool_mismatches"], 0)
        self.assertLessEqual(rec["maxpool_mismatches"], rec["maxpool_total_positions"])
        self.assertEqual(rec["idx"], 7)

    def test_identical_endpoints_zero_drift(self):
        # Same point twice: d_f, d_M, hamming, maxpool mismatch all 0.
        torch.manual_seed(1)
        x = torch.randn(3, 224, 224)
        rec = mh.per_sample_record(self.kmc, self.model, x, x.clone(),
                                   device="cpu", idx=0)
        self.assertAlmostEqual(rec["d_f"], 0.0, places=6)
        self.assertAlmostEqual(rec["d_M"], 0.0, places=6)
        self.assertEqual(rec["relu_hamming"], 0)
        self.assertEqual(rec["maxpool_mismatches"], 0)
        # A is guarded when d_M == 0 (returns None). Check None FIRST so the
        # `or` short-circuits before math.isfinite, which raises on None.
        self.assertTrue(rec["A"] is None or rec["A"] == 0.0
                        or math.isfinite(rec["A"]))


class TestPartialSpearman(unittest.TestCase):
    """partial_spearman(A, H | d_f): rank-residualize then correlate."""

    def _reference(self, A, H, z):
        """Independent reference: Spearman partial via rank-residualization.

        Rank all three, OLS-regress rank(A) and rank(H) on [1, rank(z)],
        Pearson-correlate the residuals.
        """
        from scipy.stats import rankdata
        ra = rankdata(A); rh = rankdata(H); rz = rankdata(z)
        X = np.column_stack([np.ones_like(rz), rz])
        def resid(y):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        ea, eh = resid(ra), resid(rh)
        return float(np.corrcoef(ea, eh)[0, 1])

    def test_matches_reference_on_random_data(self):
        rng = np.random.RandomState(3)
        z = rng.rand(200)
        A = 2.0 * z + 0.3 * rng.randn(200)   # both correlated with z
        H = -1.5 * z + 0.3 * rng.randn(200)
        got = mh.partial_spearman(A, H, z)
        want = self._reference(A, H, z)
        self.assertAlmostEqual(got, want, places=6)

    def test_negative_when_conditional_anticorrelated(self):
        # Construct A and H anticorrelated AFTER removing the shared z drive
        # (the crossing-mechanism prediction): partial corr should be < 0.
        rng = np.random.RandomState(4)
        z = rng.rand(300)
        common = 3.0 * z
        e = rng.randn(300)
        A = common + e
        H = common - e          # residuals exactly opposite
        rho = mh.partial_spearman(A, H, z)
        self.assertLess(rho, 0.0)

    def test_handles_degenerate_input(self):
        # Too few points / constant arrays -> None, not a crash.
        self.assertIsNone(mh.partial_spearman([1.0], [2.0], [3.0]))
        self.assertIsNone(
            mh.partial_spearman([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0])
        )


if __name__ == "__main__":
    unittest.main()
