"""Phase-1 Step C — aggregate chunk artifacts → final per-cell measure values.

This module is shared by two execution paths that MUST produce identical
output:

  * the **serial** driver (``cka_similarity.reduce.__main__``), which loops
    every combo in-process, and
  * the **SLURM-array** path (``cka_similarity.reduce.combo``) that computes one
    combo per array task, followed by the same driver acting as the **gather**.

Both call the SAME per-combo finalize functions (``finalize_s1_combo`` /
``finalize_s2_combo`` / ``finalize_s3_combo``). Each of those is a thin
cache wrapper around the original, unchanged finalize body: given a
``combo_dir`` it loads ``combo_dir/{key}.pt`` if present, otherwise computes
via the verbatim logic and writes the result. Because the computation is the
identical function in both paths, the array+gather output is provably the
same as the serial output; the serial path also becomes resumable for free.

Cache files are ``torch.save``'d ``.pt`` (NOT json): a combo's value carries a
``MeasureResult.extras`` dict that can hold tensors/arrays (e.g. soft-matching
couplings), which json cannot round-trip. ``torch.save``/``torch.load``
reconstructs the dict byte-for-byte.
"""
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from cka_similarity.measures.panel import PANEL

# Stable short codes used in the S2 pair name (matches the s2 worker).
S2_SHORT_NAMES = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}


def _measure_finalize(measure_name, chunk_accs: List):
    """Find the right MeasureBase subclass by name and call finalize."""
    for cls in PANEL:
        m = cls()
        if m.name == measure_name:
            return m.finalize(chunk_accs)
    raise ValueError(f"Unknown measure: {measure_name}")


def _finalize_panel(panel_files: List) -> Dict:
    """Load each chunk file once, then finalize every measure in its panel.

    The previous per-measure ``torch.load`` re-read every file once per
    measure (×9 I/O — ~4.3 TB of redundant deserialization across the 150
    S1 combos), risking the reduce walltime. Holding one combo's 64 chunk
    files (~3 GB of feature blocks) is well within the job's memory.
    """
    loaded = [torch.load(f)["accumulators"] for f in panel_files]
    results = {}
    for mname in loaded[0].keys():
        chunk_accs = [d[mname] for d in loaded]
        r = _measure_finalize(mname, chunk_accs)
        results[mname] = {"value": r.value, "extras": r.extras}
    return results


# ---------------------------------------------------------------------------
# Per-combo cache helpers
# ---------------------------------------------------------------------------
def combo_cache_path(combo_dir, key: str) -> Path:
    """Deterministic cache path for one combo's finalized result."""
    return Path(combo_dir) / f"{key}.pt"


def _load_combo(combo_dir, key: str):
    """Return the cached combo result if present, else None."""
    if combo_dir is None:
        return None
    p = combo_cache_path(combo_dir, key)
    if p.exists():
        return torch.load(p)
    return None


def _store_combo(combo_dir, key: str, result) -> None:
    """Atomically persist one combo's finalized result to the cache."""
    if combo_dir is None:
        return
    from utils.atomic_io import atomic_torch_save
    atomic_torch_save(str(combo_cache_path(combo_dir, key)), result)


# ---------------------------------------------------------------------------
# S1 — within-arch invariance panel (archs × teleports)
# ---------------------------------------------------------------------------
def s1_combo_key(arch: str, tid: int) -> str:
    return f"s1_{arch}_t{tid}"


def _compute_s1_combo(s1_dir: str, arch: str, tid: int, num_chunks: int):
    """Compute one (arch, teleport) S1 finalize. Returns the panel dict, or
    None when the combo has no chunk files at all (skipped, as in the serial
    path). VERBATIM body of the old ``aggregate_s1`` inner loop."""
    chunk_files = sorted(Path(s1_dir).glob(f"{arch}_teleport{tid}_chunk*.pt"))
    if len(chunk_files) != num_chunks:
        print(f"WARN: {arch} teleport {tid} has {len(chunk_files)}/{num_chunks} chunks")
        if not chunk_files:
            return None
    return _finalize_panel(chunk_files)


def finalize_s1_combo(s1_dir: str, arch: str, tid: int, num_chunks: int,
                      combo_dir=None):
    """Cache-or-compute one S1 combo. Identical result on both paths."""
    key = s1_combo_key(arch, tid)
    cached = _load_combo(combo_dir, key)
    if cached is not None:
        return cached
    result = _compute_s1_combo(s1_dir, arch, tid, num_chunks)
    if result is not None:
        _store_combo(combo_dir, key, result)
    return result


def aggregate_s1(s1_dir: str, archs: List[str], num_teleports: int,
                 num_chunks: int, combo_dir=None) -> Dict:
    """For each (arch, teleport_id), glob chunk files and finalize each measure.

    Returns: {(arch, teleport_id): {measure_name: {"value": ..., "extras": ...}}}

    When ``combo_dir`` is given, each combo's finalize is read from / written
    to ``combo_dir/{key}.pt``; otherwise it is computed in-process. The
    returned dict is identical either way.
    """
    out = {}
    for arch in archs:
        for tid in range(num_teleports):
            result = finalize_s1_combo(s1_dir, arch, tid, num_chunks, combo_dir)
            if result is not None:
                out[(arch, tid)] = result
    return out


# ---------------------------------------------------------------------------
# S2 — cross-arch (KM Frobenius + D1/D2 penultimate panels) per arch pair
# ---------------------------------------------------------------------------
def s2_pair_name(a: str, b: str) -> str:
    return f"{S2_SHORT_NAMES[a]}_{S2_SHORT_NAMES[b]}"


def s2_combo_key(pname: str) -> str:
    return f"s2_{pname}"


def _compute_s2_combo(s2_dir: str, a: str, b: str, num_chunks: int) -> Dict:
    """Compute one arch-pair S2 finalize. VERBATIM body of the old
    ``aggregate_s2`` inner loop (KM raw list + derived RMS + D1/D2 panels)."""
    from utils.scaling import (
        IMAGENET_INPUT_NUMEL, IMAGENET_NUM_CLASSES, km_numel,
    )
    s_KM = 1.0 / (km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL) ** 0.5)
    pname = s2_pair_name(a, b)

    # KM Frobenius — concatenate per-chunk lists (raw)
    km_files = sorted(Path(s2_dir).glob(f"{pname}_KM_chunk*.json"))
    all_distances = []
    for f in km_files:
        all_distances.extend(json.loads(Path(f).read_text()))

    # KM Frobenius — RMS variant. Always derived from the raw list: rms is
    # the deterministic rescale ``raw * s_KM`` (the s2 worker writes exactly
    # ``[v * s_KM for v in km_distances]``). We deliberately do NOT read the
    # on-disk ``*_KM_rms`` files: a partially-regenerated rms set (observed
    # 2026-06-06: RN_DN had 39/64 rms files after a cross-version rerun while
    # raw had 64/64) would otherwise be aggregated over fewer chunks than the
    # raw list, silently desyncing km_mean_rms from km_mean. Deriving from
    # raw keeps the two in lock-step by construction.
    all_distances_rms = [d * s_KM for d in all_distances]

    # D1 panel
    d1_files = sorted(Path(s2_dir).glob(f"{pname}_D1_chunk*.pt"))
    d1_results = _finalize_panel(d1_files) if d1_files else {}

    # D2 panel
    d2_files = sorted(Path(s2_dir).glob(f"{pname}_D2_chunk*.pt"))
    d2_results = _finalize_panel(d2_files) if d2_files else {}

    return {
        "km_distances": all_distances,
        "km_distances_rms": all_distances_rms,
        "km_mean": sum(all_distances) / max(1, len(all_distances)),
        "km_mean_rms": sum(all_distances_rms) / max(1, len(all_distances_rms)),
        "km_n": len(all_distances),
        "D1": d1_results,
        "D2": d2_results,
    }


def finalize_s2_combo(s2_dir: str, a: str, b: str, num_chunks: int,
                      combo_dir=None) -> Dict:
    """Cache-or-compute one S2 arch-pair combo. Identical result on both paths."""
    key = s2_combo_key(s2_pair_name(a, b))
    cached = _load_combo(combo_dir, key)
    if cached is not None:
        return cached
    result = _compute_s2_combo(s2_dir, a, b, num_chunks)
    _store_combo(combo_dir, key, result)
    return result


def aggregate_s2(s2_dir: str, archs: List[str], num_chunks: int,
                 combo_dir=None) -> Dict:
    """For each unordered arch pair, aggregate KM Frobenius distances + D1/D2 panels.

    Per-pair KM Frobenius distances are stored on disk in both raw units
    ({pname}_KM_chunk*.json) and per-coordinate RMS units
    ({pname}_KM_rms_chunk*.json — written by
    cka_similarity/workers/s2_cross_architecture.py since 2026-05-10).
    If RMS files are missing (pre-2026-05-10 chunks), fall back to deriving
    RMS values from raw using utils.scaling.km_numel — this preserves the
    canonical metric across legacy and fresh data.
    """
    out = {}
    for a, b in combinations(archs, 2):
        out[s2_pair_name(a, b)] = finalize_s2_combo(s2_dir, a, b, num_chunks, combo_dir)
    return out


# ---------------------------------------------------------------------------
# S3 — distance amplification per (arch, attack)
# ---------------------------------------------------------------------------
def s3_combo_key(arch: str, attack: str) -> str:
    return f"s3_{arch}_{attack}"


def _compute_s3_combo(s3_dir: str, arch: str, attack: str, num_chunks: int) -> Dict:
    """Compute one (arch, attack) S3 finalize. VERBATIM body of the old
    ``aggregate_s3`` inner loop (per-pair list + empty-marker-filtered panel)."""
    # Per-pair distance lists
    dist_files = sorted(Path(s3_dir).glob(f"{arch}_{attack}_chunk*.json"))
    all_pairs = []
    for f in dist_files:
        all_pairs.extend(json.loads(Path(f).read_text()))

    # Panel accumulators. S3 writes n_pairs=0 markers for chunks past
    # the pair n_done (deepfool's partial pairs.pth — see
    # workers/s3_distance_amplification.py). Filter those out before
    # finalizing measures so an empty-marker chunk0 doesn't blank the
    # measure_names list.
    panel_files = sorted(Path(s3_dir).glob(f"{arch}_{attack}_panel_chunk*.pt"))
    loaded = [torch.load(f) for f in panel_files]
    non_empty = [d for d in loaded if d.get("n_pairs", 0) > 0]
    panel_results = {}
    if non_empty:
        measure_names = list(non_empty[0]["accumulators"].keys())
        for mname in measure_names:
            chunk_accs = [d["accumulators"][mname] for d in non_empty]
            r = _measure_finalize(mname, chunk_accs)
            panel_results[mname] = {"value": r.value, "extras": r.extras}

    return {
        "per_pair": all_pairs,
        "n_pairs": len(all_pairs),
        "panel": panel_results,
    }


def finalize_s3_combo(s3_dir: str, arch: str, attack: str, num_chunks: int,
                      combo_dir=None) -> Dict:
    """Cache-or-compute one S3 combo. Identical result on both paths."""
    key = s3_combo_key(arch, attack)
    cached = _load_combo(combo_dir, key)
    if cached is not None:
        return cached
    result = _compute_s3_combo(s3_dir, arch, attack, num_chunks)
    _store_combo(combo_dir, key, result)
    return result


def aggregate_s3(s3_dir: str, archs: List[str], attacks: List[str],
                 num_chunks: int, combo_dir=None) -> Dict:
    """For each (arch, attack), aggregate per-pair distances + panel accumulators."""
    out = {}
    for arch in archs:
        for attack in attacks:
            out[(arch, attack)] = finalize_s3_combo(
                s3_dir, arch, attack, num_chunks, combo_dir)
    return out


# ---------------------------------------------------------------------------
# Combo enumerator (the source of truth for the array bound + index→combo map)
# ---------------------------------------------------------------------------
def enumerate_combos(archs: List[str], num_teleports: int,
                     attacks: List[str]) -> List[Tuple[str, str, dict]]:
    """Ordered, gap-free, dup-free list of every reduce combo.

    Returns a list of ``(stage, key, params)`` where ``stage`` is one of
    ``"s1" | "s2" | "s3"``, ``key`` is the deterministic cache key, and
    ``params`` carries the identifiers that ``run_combo`` needs to finalize
    exactly that combo. Ordering is stable: all S1 (archs × teleports), then
    S2 (arch pairs, ``itertools.combinations`` order — matches ``aggregate_s2``),
    then S3 (archs × attacks). The list index IS the SLURM array index, so the
    map index→combo is a bijection over ``range(len(...))``.
    """
    combos: List[Tuple[str, str, dict]] = []
    for arch in archs:
        for tid in range(num_teleports):
            combos.append(("s1", s1_combo_key(arch, tid),
                           {"arch": arch, "tid": tid}))
    for a, b in combinations(archs, 2):
        combos.append(("s2", s2_combo_key(s2_pair_name(a, b)),
                       {"a": a, "b": b}))
    for arch in archs:
        for attack in attacks:
            combos.append(("s3", s3_combo_key(arch, attack),
                           {"arch": arch, "attack": attack}))
    return combos


def run_combo(stage: str, params: dict, combo_dir, num_chunks: int,
              s1_dir: str, s2_dir: str, s3_dir: str):
    """Finalize a single combo (used by the array entrypoint).

    Dispatches to the same per-combo finalize functions the gather uses, so a
    combo populated here is byte-identical to one the serial path would
    produce. Idempotent: the finalize functions skip recompute when the combo
    is already cached in ``combo_dir``.
    """
    if stage == "s1":
        return finalize_s1_combo(s1_dir, params["arch"], params["tid"],
                                 num_chunks, combo_dir)
    if stage == "s2":
        return finalize_s2_combo(s2_dir, params["a"], params["b"],
                                 num_chunks, combo_dir)
    if stage == "s3":
        return finalize_s3_combo(s3_dir, params["arch"], params["attack"],
                                 num_chunks, combo_dir)
    raise ValueError(f"Unknown stage: {stage}")
