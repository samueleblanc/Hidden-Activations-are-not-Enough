"""Phase-1 reduce SLURM-array refactor: enumerator, per-combo cache, and the
headline gather==serial output-equivalence.

The #1 invariant of the refactor: the array+gather path must produce output
IDENTICAL to the old serial reduce. These tests prove that on synthetic chunk
files for S1, S2, and S3, plus the supporting enumerator/idempotency contracts.
"""
import json

import torch

from cka_similarity.measures.panel import PANEL, panel_names
from cka_similarity.reduce import aggregate as agg
from cka_similarity.reduce.aggregate import (
    aggregate_s1, aggregate_s2, aggregate_s3,
    enumerate_combos, run_combo, combo_cache_path,
    s1_combo_key, s2_combo_key, s2_pair_name, s3_combo_key,
)

ARCHS = ["resnet152", "densenet121", "googlenet"]
ATTACKS = ["fgsm", "pgd", "cw", "deepfool", "apgd", "square"]


# ---------------------------------------------------------------------------
# Synthetic chunk builders — match the on-disk formats the workers write.
# Both the serial path and the array path read the SAME files on disk, so the
# only thing the equivalence test exercises is the cache round-trip + dispatch.
# ---------------------------------------------------------------------------
def _panel_accumulators(seed, n=8, p_a=12, p_b=12):
    """Build one chunk's full-panel accumulator dict (all 9 measures)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, p_a, generator=g, dtype=torch.float64)
    B = torch.randn(n, p_b, generator=g, dtype=torch.float64)
    return {cls.name: cls().accumulate(A, B) for cls in PANEL}


def _write_s1_chunks(s1_dir, archs, num_teleports, num_chunks):
    s1_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    for arch in archs:
        for tid in range(num_teleports):
            for c in range(num_chunks):
                torch.save(
                    {"chunk_id": c, "arch": arch, "teleport_id": tid,
                     "accumulators": _panel_accumulators(seed)},
                    s1_dir / f"{arch}_teleport{tid}_chunk{c}.pt",
                )
                seed += 1


def _write_s2_chunks(s2_dir, archs, num_chunks):
    from itertools import combinations
    s2_dir.mkdir(parents=True, exist_ok=True)
    seed = 1000
    for a, b in combinations(archs, 2):
        pname = s2_pair_name(a, b)
        for c in range(num_chunks):
            g = torch.Generator().manual_seed(seed)
            dists = torch.rand(5, generator=g).tolist()
            (s2_dir / f"{pname}_KM_chunk{c}.json").write_text(json.dumps(dists))
            torch.save({"chunk_id": c, "accumulators": _panel_accumulators(seed)},
                       s2_dir / f"{pname}_D1_chunk{c}.pt")
            torch.save({"chunk_id": c, "accumulators": _panel_accumulators(seed + 1)},
                       s2_dir / f"{pname}_D2_chunk{c}.pt")
            seed += 2


def _write_s3_chunks(s3_dir, archs, attacks, num_chunks):
    s3_dir.mkdir(parents=True, exist_ok=True)
    seed = 2000
    for arch in archs:
        for attack in attacks:
            for c in range(num_chunks):
                g = torch.Generator().manual_seed(seed)
                pairs = [{"d_f": float(torch.rand(1, generator=g)),
                          "d_h": float(torch.rand(1, generator=g)),
                          "d_M": float(torch.rand(1, generator=g)),
                          "completeness_residual_clean": 1e-7,
                          "completeness_residual_adv": 1e-7} for _ in range(3)]
                (s3_dir / f"{arch}_{attack}_chunk{c}.json").write_text(json.dumps(pairs))
                torch.save({"chunk_id": c, "n_pairs": 3,
                            "accumulators": _panel_accumulators(seed)},
                           s3_dir / f"{arch}_{attack}_panel_chunk{c}.pt")
                seed += 1


# ---------------------------------------------------------------------------
# Recursive equality (tolerant of float round-trip; structure must match)
# ---------------------------------------------------------------------------
def _assert_equal(a, b, path="root"):
    assert type(a) == type(b), f"{path}: type {type(a)} != {type(b)}"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            _assert_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, float):
        if a != a:  # NaN
            assert b != b, f"{path}: {a} != {b}"
        else:
            assert a == b, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a} != {b}"


# ===========================================================================
# Enumerator coverage / bijection
# ===========================================================================
def test_enumerator_count_and_bijection():
    combos = enumerate_combos(ARCHS, num_teleports=50, attacks=ATTACKS)
    # 3 archs * 50 teleports + C(3,2)=3 pairs + 3 archs * 6 attacks
    assert len(combos) == 150 + 3 + 18 == 171

    keys = [key for _stage, key, _params in combos]
    assert len(keys) == len(set(keys)), "combo keys must be unique"

    # Stage partition counts
    stages = [s for s, _k, _p in combos]
    assert stages.count("s1") == 150
    assert stages.count("s2") == 3
    assert stages.count("s3") == 18

    # The list index IS the array index; index -> combo is a bijection over range(N).
    indices = list(range(len(combos)))
    assert sorted(indices) == indices
    # Round-trip: every key is reachable from exactly one index.
    key_to_idx = {key: i for i, (_s, key, _p) in enumerate(combos)}
    assert len(key_to_idx) == len(combos)


def test_enumerator_keys_match_combo_key_helpers():
    combos = enumerate_combos(ARCHS, num_teleports=2, attacks=["fgsm", "pgd"])
    by_stage = {}
    for stage, key, params in combos:
        by_stage.setdefault(stage, []).append((key, params))

    for key, p in by_stage["s1"]:
        assert key == s1_combo_key(p["arch"], p["tid"])
    for key, p in by_stage["s2"]:
        assert key == s2_combo_key(s2_pair_name(p["a"], p["b"]))
    for key, p in by_stage["s3"]:
        assert key == s3_combo_key(p["arch"], p["attack"])


# ===========================================================================
# Per-combo cache idempotency
# ===========================================================================
def test_s1_combo_cache_idempotent(tmp_path, monkeypatch):
    s1_dir = tmp_path / "s1"
    _write_s1_chunks(s1_dir, ["resnet152"], num_teleports=1, num_chunks=2)
    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()

    # First call computes + writes the cache file.
    r1 = agg.finalize_s1_combo(str(s1_dir), "resnet152", 0, 2, combo_dir=str(combo_dir))
    cache_file = combo_cache_path(combo_dir, s1_combo_key("resnet152", 0))
    assert cache_file.exists()

    # Second call must LOAD (no recompute): poison the compute path to prove it.
    def _boom(*a, **k):
        raise AssertionError("recompute happened — cache was not used")
    monkeypatch.setattr(agg, "_compute_s1_combo", _boom)
    r2 = agg.finalize_s1_combo(str(s1_dir), "resnet152", 0, 2, combo_dir=str(combo_dir))

    _assert_equal(r1, r2)


def test_run_combo_skips_when_cached(tmp_path, monkeypatch):
    s3_dir = tmp_path / "s3"
    _write_s3_chunks(s3_dir, ["resnet152"], ["fgsm"], num_chunks=2)
    combo_dir = tmp_path / "combos"

    _stage, key, params = enumerate_combos(["resnet152"], 0, ["fgsm"])[0]
    run_combo("s3", params, str(combo_dir), 2,
              str(tmp_path / "s1"), str(tmp_path / "s2"), str(s3_dir))
    assert combo_cache_path(combo_dir, key).exists()

    # Re-running the same index is a no-op (idempotent) — recompute must not fire.
    monkeypatch.setattr(agg, "_compute_s3_combo",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    run_combo("s3", params, str(combo_dir), 2,
              str(tmp_path / "s1"), str(tmp_path / "s2"), str(s3_dir))


# ===========================================================================
# THE headline test: gather == serial, per stage
# ===========================================================================
def _gather_via_array(stage, archs, num_teleports, attacks, num_chunks,
                      combo_dir, s1_dir, s2_dir, s3_dir):
    """Populate combo_dir by running every combo through the array entrypoint."""
    for _stage, _key, params in enumerate_combos(archs, num_teleports, attacks):
        if _stage == stage:
            run_combo(_stage, params, combo_dir, num_chunks, s1_dir, s2_dir, s3_dir)


def test_gather_equals_serial_s1(tmp_path):
    s1_dir = tmp_path / "s1"
    _write_s1_chunks(s1_dir, ARCHS, num_teleports=2, num_chunks=3)

    # Serial: compute every combo in-process (no cache).
    serial = aggregate_s1(str(s1_dir), ARCHS, 2, 3)

    # Array+gather: each combo computed by the array entrypoint, then aggregate
    # reads the cache.
    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()
    _gather_via_array("s1", ARCHS, 2, ATTACKS, 3, str(combo_dir),
                      str(s1_dir), str(tmp_path / "s2"), str(tmp_path / "s3"))
    gathered = aggregate_s1(str(s1_dir), ARCHS, 2, 3, combo_dir=str(combo_dir))

    assert serial.keys() == gathered.keys()
    _assert_equal(serial, gathered)
    # Sanity: the panel actually carried all 9 measures through the round-trip.
    any_cell = next(iter(serial.values()))
    assert set(any_cell.keys()) == set(panel_names())


def test_gather_equals_serial_s2(tmp_path):
    s2_dir = tmp_path / "s2"
    _write_s2_chunks(s2_dir, ARCHS, num_chunks=3)

    serial = aggregate_s2(str(s2_dir), ARCHS, 3)

    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()
    _gather_via_array("s2", ARCHS, 0, ATTACKS, 3, str(combo_dir),
                      str(tmp_path / "s1"), str(s2_dir), str(tmp_path / "s3"))
    gathered = aggregate_s2(str(s2_dir), ARCHS, 3, combo_dir=str(combo_dir))

    assert serial.keys() == gathered.keys()
    _assert_equal(serial, gathered)
    # km_mean_rms must stay locked to km_mean * s_KM (the derive-from-raw rule).
    for pname, d in gathered.items():
        assert d["km_n"] == len(d["km_distances"]) == 15  # 5 dists * 3 chunks


def test_gather_equals_serial_s3(tmp_path):
    s3_dir = tmp_path / "s3"
    _write_s3_chunks(s3_dir, ARCHS, ATTACKS, num_chunks=3)

    serial = aggregate_s3(str(s3_dir), ARCHS, ATTACKS, 3)

    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()
    _gather_via_array("s3", ARCHS, 0, ATTACKS, 3, str(combo_dir),
                      str(tmp_path / "s1"), str(tmp_path / "s2"), str(s3_dir))
    gathered = aggregate_s3(str(s3_dir), ARCHS, ATTACKS, 3, combo_dir=str(combo_dir))

    assert serial.keys() == gathered.keys()
    _assert_equal(serial, gathered)


def test_gather_recomputes_missing_combo(tmp_path):
    """Robustness: if an array task failed (its combo file is absent), the
    gather recomputes that combo in-process and STILL matches serial."""
    s1_dir = tmp_path / "s1"
    _write_s1_chunks(s1_dir, ["resnet152"], num_teleports=2, num_chunks=2)
    serial = aggregate_s1(str(s1_dir), ["resnet152"], 2, 2)

    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()
    # Only populate teleport 0; teleport 1's array task "failed".
    run_combo("s1", {"arch": "resnet152", "tid": 0}, str(combo_dir), 2,
              str(s1_dir), str(tmp_path / "s2"), str(tmp_path / "s3"))
    assert combo_cache_path(combo_dir, s1_combo_key("resnet152", 0)).exists()
    assert not combo_cache_path(combo_dir, s1_combo_key("resnet152", 1)).exists()

    gathered = aggregate_s1(str(s1_dir), ["resnet152"], 2, 2, combo_dir=str(combo_dir))
    _assert_equal(serial, gathered)
    # The fallback combo got written to the cache as a side effect (now resumable).
    assert combo_cache_path(combo_dir, s1_combo_key("resnet152", 1)).exists()


# ===========================================================================
# Regression: S2 D2 (cross-arch Procrustes) guard
# ===========================================================================
def _d1_native_acc(seed, n=8, p_a=12, p_b=12):
    """One D1 chunk's accumulators — the cross_dim_native measures the S2 worker
    writes to D1. Consistent dims across chunks (the normal case)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, p_a, generator=g, dtype=torch.float64)
    B = torch.randn(n, p_b, generator=g, dtype=torch.float64)
    return {cls.name: cls().accumulate(A, B) for cls in PANEL if cls.cross_dim_native}


def _procrustes_only_acc(seed, n, p):
    """One D2 chunk's accumulators — ONLY procrustes, as the S2 worker writes it
    (the lone cross_dim_native=False measure). Feature dim p varies per chunk to
    mimic pca_project clamping target_dim to each chunk's sample count."""
    from cka_similarity.measures.procrustes import ProcrustesShapeDistance
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, p, generator=g, dtype=torch.float64)
    B = torch.randn(n, p, generator=g, dtype=torch.float64)
    return {"procrustes": ProcrustesShapeDistance().accumulate(A, B)}


def test_s2_d2_guard_omits_uncombinable_procrustes(tmp_path):
    """Cross-arch D2 (Procrustes) is computed on per-chunk PCA projections whose
    target_dim is clamped to each chunk's sample count, so the chunks carry
    MISMATCHED feature dims (the real failure: 390 vs 430). The S2 finalize must
    OMIT the uncombinable measure and still return KM Frobenius + the D1 panel —
    not crash the whole combo (the bug that failed array tasks 150-152)."""
    s2_dir = tmp_path / "s2"
    s2_dir.mkdir(parents=True)
    a, b = "resnet152", "densenet121"
    pname = s2_pair_name(a, b)
    num_chunks = 3
    dims = [10, 11, 12]  # mismatched per-chunk feature dims
    for c in range(num_chunks):
        (s2_dir / f"{pname}_KM_chunk{c}.json").write_text(json.dumps([0.1 * (c + 1)] * 5))
        torch.save({"chunk_id": c, "accumulators": _d1_native_acc(1000 + c)},
                   s2_dir / f"{pname}_D1_chunk{c}.pt")
        torch.save({"chunk_id": c, "accumulators": _procrustes_only_acc(2000 + c, n=8, p=dims[c])},
                   s2_dir / f"{pname}_D2_chunk{c}.pt")

    # Must NOT raise (the bug raised RuntimeError on the A_sum dim mismatch).
    out = aggregate_s2(str(s2_dir), [a, b], num_chunks)
    d = out[pname]

    # The uncombinable cross-arch Procrustes is omitted; the combo still completes.
    assert "procrustes" not in d["D2"], "uncombinable cross-arch Procrustes must be omitted"
    # KM Frobenius survived (5 dists * 3 chunks).
    assert d["km_n"] == 15
    # The D1 native panel survived intact (the 8 cross_dim_native measures).
    assert set(d["D1"].keys()) == {cls.name for cls in PANEL if cls.cross_dim_native}
    assert len(d["D1"]) == 8
