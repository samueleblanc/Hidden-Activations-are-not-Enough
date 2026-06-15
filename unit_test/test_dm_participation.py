"""Tests for dm_participation.py (Plan B / D5).

All math-correctness tests run on SMALL SYNTHETIC matrices — NO heavy real
models (a concurrent test is building the real trio; these stay light and
CPU-only under env/, Python 3.11). The headline correctness concern is the
C×C-Gram-not-full-SVD route, exercised in TestGramVsSvd.

  * TestColumnConcentration — PR_col / top1_col_frac / nonzero_columns against
    hand computation, including the single-pixel sparse limit.
  * TestSpectralConcentration — PR_spec / sigma_top10 against hand computation.
  * TestGramVsSvd — the KEY correctness test: eigenvalues of ΔM ΔMᵀ equal the
    squared singular values from a direct torch.linalg.svd; AND the production
    code path uses the Gram (eigvalsh), never a full-matrix svd.
  * TestVisibleInvisibleSplit — the Pythagoras decomposition (d_M², d_f²/(d+1),
    invisible_mass, A), including the roundoff clamp.
  * TestParticipationRecord — the assembled per-pair schema.
"""
import inspect
import math
import unittest

import torch

import dm_participation as dp


def _pr(weights):
    """Independent participation-ratio reference: (Σw)² / Σw²."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


class TestColumnConcentration(unittest.TestCase):
    """PR_col / top1_col_frac / nonzero_columns on known ΔM."""

    def test_two_equal_columns(self):
        # ΔM with two columns of equal mass and one zero column.
        # col masses: [1, 1, 0]  → PR_col = (2)²/2 = 2, top1 = 0.5, nnz = 2.
        dM = torch.tensor([[1.0, 1.0, 0.0]])  # 1 x 3
        out = dp.column_concentration(dM)
        self.assertAlmostEqual(out["PR_col"], 2.0, places=10)
        self.assertAlmostEqual(out["top1_col_frac"], 0.5, places=10)
        self.assertEqual(out["nonzero_columns"], 2)

    def test_matches_handcomputed_general(self):
        torch.manual_seed(0)
        dM = torch.randn(6, 10, dtype=torch.float64)
        col_mass = (dM ** 2).sum(dim=0)
        want_pr = _pr(col_mass)
        want_top1 = float(col_mass.max() / col_mass.sum())
        want_nnz = int((col_mass > 0).sum())
        out = dp.column_concentration(dM)
        self.assertAlmostEqual(out["PR_col"], want_pr, places=9)
        self.assertAlmostEqual(out["top1_col_frac"], want_top1, places=9)
        self.assertEqual(out["nonzero_columns"], want_nnz)

    def test_single_pixel_sparse_limit(self):
        # All mass in ONE column (sparse / one-pixel limit):
        # PR_col ≈ 1, top1_col_frac ≈ 1, nonzero_columns == 1.
        dM = torch.zeros(6, 10, dtype=torch.float64)
        dM[:, 3] = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
                                dtype=torch.float64)
        out = dp.column_concentration(dM)
        self.assertAlmostEqual(out["PR_col"], 1.0, places=10)
        self.assertAlmostEqual(out["top1_col_frac"], 1.0, places=10)
        self.assertEqual(out["nonzero_columns"], 1)

    def test_zero_matrix(self):
        out = dp.column_concentration(torch.zeros(4, 5))
        self.assertEqual(out["PR_col"], 0.0)
        self.assertEqual(out["top1_col_frac"], 0.0)
        self.assertEqual(out["nonzero_columns"], 0)


class TestSpectralConcentration(unittest.TestCase):
    """PR_spec / sigma_top10 on known ΔM."""

    def test_rank_one_matrix(self):
        # Rank-1 ΔM = u vᵀ: one nonzero singular value σ = ||u|| ||v||,
        # all others 0 → PR_spec ≈ 1.
        u = torch.tensor([[1.0], [2.0], [2.0]], dtype=torch.float64)  # 3x1
        v = torch.tensor([[3.0, 0.0, 4.0, 0.0]], dtype=torch.float64)  # 1x4
        dM = u @ v  # 3 x 4, rank 1
        out = dp.spectral_concentration(dM)
        self.assertAlmostEqual(out["PR_spec"], 1.0, places=8)
        sigma_expected = float(torch.linalg.norm(u) * torch.linalg.norm(v))
        self.assertAlmostEqual(out["sigma_top10"][0], sigma_expected, places=8)
        # Remaining reported singular values are ~0. NOTE: eigvalsh returns the
        # near-zero eigenvalues as ~1e-15 noise and sqrt amplifies that to ~1e-8
        # (and the magnitude differs between torch 2.2.2 / 2.6.0). Assert they
        # are negligible RELATIVE to the leading σ (~15) rather than at a fixed
        # 1e-8 absolute floor that the sqrt-of-roundoff can breach.
        for s in out["sigma_top10"][1:]:
            self.assertLess(s, 1e-5 * sigma_expected)

    def test_pr_spec_matches_eigen_reference(self):
        torch.manual_seed(1)
        dM = torch.randn(6, 10, dtype=torch.float64)
        eig = torch.linalg.eigvalsh(dM @ dM.T).clamp(min=0.0)
        want = _pr(eig)
        out = dp.spectral_concentration(dM)
        self.assertAlmostEqual(out["PR_spec"], want, places=9)

    def test_sigma_top10_length_and_order(self):
        torch.manual_seed(2)
        dM = torch.randn(6, 10, dtype=torch.float64)
        out = dp.spectral_concentration(dM)
        self.assertEqual(len(out["sigma_top10"]), 10)  # C=6 < 10 → zero-padded
        # Descending.
        s = out["sigma_top10"]
        for a, b in zip(s, s[1:]):
            self.assertGreaterEqual(a + 1e-9, b)


class TestGramVsSvd(unittest.TestCase):
    """KEY correctness test: Gram eigenvalues == squared singular values, and
    the production path never calls a full-matrix SVD."""

    def test_gram_eigvals_equal_squared_singular_values(self):
        torch.manual_seed(7)
        # Small wide matrix (C < d+1), like the real KM but tiny.
        dM = torch.randn(8, 20, dtype=torch.float64)

        # Direct SVD (reference ONLY — never used in production code path).
        sv = torch.linalg.svdvals(dM)            # length min(8,20)=8, descending
        sv2_sorted = torch.sort(sv ** 2, descending=True).values

        # Gram route (what the production code uses).
        eig = torch.linalg.eigvalsh(dM.double() @ dM.double().T).clamp(min=0.0)
        eig_sorted = torch.sort(eig, descending=True).values

        # The 8 nonzero-eligible eigenvalues must match σ² exactly.
        self.assertTrue(
            torch.allclose(eig_sorted, sv2_sorted, atol=1e-9, rtol=1e-7),
            f"Gram eigenvalues != σ²\n eig={eig_sorted}\n σ²={sv2_sorted}",
        )

        # And spectral_concentration's reported sigma_top10 must equal the SVD's
        # leading singular values (sqrt of the matching eigenvalues).
        out = dp.spectral_concentration(dM, n_top=8)
        got_sigma = torch.tensor(out["sigma_top10"], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(got_sigma, sv, atol=1e-8, rtol=1e-6),
            f"sigma_top10 != svdvals\n got={got_sigma}\n want={sv}",
        )

    def test_code_path_uses_gram_not_full_svd(self):
        """Static guard: spectral_concentration must NOT call torch.linalg.svd
        / svdvals / torch.svd on the full matrix. It must build the Gram and
        use eigvalsh. (A full SVD on the 1000×150529 ImageNet ΔM is infeasible;
        this test is the line of defence against a regression introducing one.)

        We tokenise the source and inspect attribute-access chains so the word
        "svd" appearing in a docstring/comment does NOT trip the guard — only a
        real *call* like torch.linalg.svd / torch.svd / x.svdvals would.
        """
        import ast

        src = inspect.getsource(dp.spectral_concentration)
        tree = ast.parse(src)

        svd_names = {"svd", "svdvals", "svd_lowrank"}
        offenders = []
        for node in ast.walk(tree):
            # Attribute access: torch.svd, torch.linalg.svdvals, x.svd, ...
            if isinstance(node, ast.Attribute) and node.attr in svd_names:
                offenders.append(node.attr)
            # Bare name: a `from torch import svd` style call.
            if isinstance(node, ast.Name) and node.id in svd_names:
                offenders.append(node.id)
        self.assertEqual(
            offenders, [],
            f"spectral_concentration must not call any SVD routine; found "
            f"{offenders} — use the C×C Gram eigvalsh route instead")

        # Must positively use the Gram eigvalsh route.
        self.assertIn("eigvalsh", src,
                      "spectral_concentration must use the Gram eigvalsh route")
        # The Gram product ΔM ΔMᵀ must be formed (the .T transpose).
        self.assertIn(".T", src, "spectral_concentration must form ΔM ΔMᵀ")


class TestVisibleInvisibleSplit(unittest.TestCase):
    """Pythagoras decomposition: d_M², d_f²/(d+1), invisible_mass, A."""

    def test_matches_handcomputed(self):
        torch.manual_seed(3)
        dM = torch.randn(5, 7, dtype=torch.float64)   # d+1 = 7
        df = dM.sum(dim=1)                              # Δf = ΔM·1
        out = dp.visible_invisible_split(dM, df)

        dM_f2 = float((dM ** 2).sum())
        d_f = float(torch.linalg.norm(df))
        d_M = math.sqrt(dM_f2)
        want_invisible = dM_f2 - d_f ** 2 / 7
        want_A = (d_f / d_M) ** 2

        self.assertAlmostEqual(out["d_M"], d_M, places=9)
        self.assertAlmostEqual(out["d_f"], d_f, places=9)
        self.assertAlmostEqual(out["invisible_mass"], want_invisible, places=8)
        self.assertAlmostEqual(out["A"], want_A, places=9)

    def test_invisible_mass_nonnegative_and_bounded(self):
        # invisible_mass ∈ [0, d_M²] always (Pythagoras).
        torch.manual_seed(8)
        dM = torch.randn(5, 7, dtype=torch.float64)
        df = dM.sum(dim=1)
        out = dp.visible_invisible_split(dM, df)
        self.assertGreaterEqual(out["invisible_mass"], 0.0)
        self.assertLessEqual(out["invisible_mass"], out["d_M"] ** 2 + 1e-9)

    def test_rank_one_along_ones_is_fully_visible(self):
        # If every row of ΔM is constant (proportional to the all-ones row),
        # ΔM lies entirely along the visible direction → invisible_mass ≈ 0.
        d1 = 7
        col = torch.tensor([[1.0], [2.0], [-3.0], [0.5], [4.0]],
                           dtype=torch.float64)        # 5x1 per-row scale
        dM = col @ torch.ones(1, d1, dtype=torch.float64)  # 5x7, each row const
        df = dM.sum(dim=1)                             # = col.squeeze() * 7
        out = dp.visible_invisible_split(dM, df)
        self.assertAlmostEqual(out["invisible_mass"], 0.0, places=8)
        # A = (d_f/d_M)² = d+1 here (one-pixel-law style ceiling on the visible
        # direction): ||Δf||² = (Σ row·1)², d_M² = Σ row² · d1.
        self.assertAlmostEqual(out["A"], float(d1), places=8)

    def test_zero_matrix_A_is_none(self):
        out = dp.visible_invisible_split(torch.zeros(5, 7), torch.zeros(5))
        self.assertIsNone(out["A"])           # d_M == 0 → guarded to None
        self.assertEqual(out["invisible_mass"], 0.0)

    def test_roundoff_negative_clamped(self):
        # A purely-visible ΔM can give a tiny-negative d_M² − d_f²/(d+1) from
        # float roundoff; the result must be clamped to exactly 0, never < 0.
        d1 = 13
        col = torch.randn(9, 1, dtype=torch.float64)
        dM = col @ torch.ones(1, d1, dtype=torch.float64)
        df = dM.sum(dim=1)
        out = dp.visible_invisible_split(dM, df)
        self.assertGreaterEqual(out["invisible_mass"], 0.0)


class TestParticipationRecord(unittest.TestCase):
    """Assembled per-pair schema from ΔM and Δf."""

    EXPECTED_KEYS = {
        "idx", "d_f", "d_M", "A", "PR_col", "top1_col_frac",
        "nonzero_columns", "PR_spec", "sigma_top10", "invisible_mass",
    }

    def test_record_schema_and_types(self):
        torch.manual_seed(5)
        dM = torch.randn(6, 11, dtype=torch.float64)
        df = dM.sum(dim=1)
        rec = dp.participation_record(dM, df, idx=42)
        self.assertEqual(set(rec.keys()), self.EXPECTED_KEYS)
        self.assertEqual(rec["idx"], 42)
        for k in ("d_f", "d_M", "A", "PR_col", "top1_col_frac",
                  "PR_spec", "invisible_mass"):
            self.assertTrue(math.isfinite(rec[k]), f"{k} not finite")
        self.assertIsInstance(rec["nonzero_columns"], int)
        self.assertIsInstance(rec["sigma_top10"], list)
        self.assertEqual(len(rec["sigma_top10"]), dp.N_SIGMA_TOP)

    def test_df_consistent_with_matrix_rowsum(self):
        # The driver builds Δf = ΔM·1; confirm d_f equals ||row-sum||.
        torch.manual_seed(6)
        dM = torch.randn(6, 11, dtype=torch.float64)
        df = dM.sum(dim=1)
        rec = dp.participation_record(dM, df, idx=0)
        self.assertAlmostEqual(
            rec["d_f"], float(torch.linalg.norm(dM.sum(dim=1))), places=9)


if __name__ == "__main__":
    unittest.main()
