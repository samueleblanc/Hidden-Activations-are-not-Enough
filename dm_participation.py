"""Plan B / D5 — ΔM participation (sparse-vs-dense attack characterisation).

For each stored adversarial pair this script measures HOW the knowledge-matrix
change ΔM = M(x') − M(x) is *spread* across the matrix, along two axes:

  Entrywise / column concentration (sparse-vs-dense)
    Per-column mass  c_j = ||ΔM[:, j]||₂²  over all d+1 columns. From these:
      PR_col          = (Σ_j c_j)² / Σ_j c_j²   participation ratio — the
                        effective number of active columns (small for a
                        sparse / few-pixel attack, large for a dense one).
      top1_col_frac   = max_j c_j / Σ_j c_j      mass in the single hottest col.
      nonzero_columns = # columns with c_j > 0.

  Spectral concentration
    Eigenvalues λ_i of the C×C Gram  G = ΔM ΔMᵀ  (= squared singular values
    of ΔM). From these:
      PR_spec    = (Σ_i λ_i)² / Σ_i λ_i²         effective rank.
      sigma_top10= top-10 sqrt(λ_i)              leading singular values.

  Visible / invisible split (the Pythagoras decomposition along 1/√(d+1))
      d_M²          = ||ΔM||_F²
      d_f           = ||Δf||₂  with Δf = ΔM·1 = f(x')−f(x) (row-sum identity)
      invisible_mass= d_M² − d_f²/(d+1)          logit-invisible Frobenius
                      energy (clamped at 0 against roundoff).
      A             = (d_f / d_M)²               coherence (metric-invariant).

Sparse attacks (e.g. one-pixel / few-pixel families) concentrate ΔM into a
handful of columns ⇒ small PR_col, large top1_col_frac; dense attacks (FGSM /
PGD touch every pixel) spread it ⇒ large PR_col. This complements D4's
crossing-mechanism panel (mask_hamming_trio.py) on the SAME stored endpoints.

CRITICAL — Gram, never full SVD. ΔM is 1000 × 150529 for ImageNet; a full
torch.linalg.svd on it is infeasible. The C×C Gram G = ΔM ΔMᵀ is only
1000 × 1000, and its eigenvalues are exactly the squared singular values of
ΔM (the nonzero spectrum is identical). We therefore eigendecompose G and
NEVER call svd/eig on the full ΔM. (Verified against a direct SVD on a small
matrix in unit_test/test_dm_participation.py::test_gram_matches_svd.)

This script does NOT regenerate attacks. It reads the stored A2 endpoints
(experiments/{arch}_imagenet/.../pairs.pth) and runs TWO KM forwards per pair.

Cost: 2 KM extractions/pair (resnet152 @ ~104 s/KM on an H100 is the worst
case) plus one 1000×1000 eigendecomposition — negligible next to the forwards.
MAX_PAIRS (default 100) bounds each (arch, attack) cell. Per-sample atomic
checkpointing resumes a wall-hit from the last completed sample.

Usage:
    python dm_participation.py --arch resnet152 --attack PGD \
        --pairs experiments/resnet152_imagenet/adversarial_pairs_N5000/pgd/pairs.pth
    python dm_participation.py --arch googlenet --attack FGSM \
        --pairs <pairs.pth> --temp_dir $SLURM_TMPDIR
"""
from __future__ import annotations

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.km_models import build_model
from utils.atomic_io import atomic_json_dump, atomic_json_load


SUPPORTED_ARCHS = ("resnet152", "densenet121", "googlenet")
ATTACKS = ("FGSM", "PGD", "CW", "DeepFool", "APGD", "Square")

# Number of leading singular values to report per pair.
N_SIGMA_TOP = 10


# ---------------------------------------------------------------------------
# Model / KM helpers (mirrors mask_hamming_trio.make_kmc)
# ---------------------------------------------------------------------------

def make_kmc(model: nn.Module, device: str, batch_size: int = 4096):
    """Construct a KnowledgeMatrixComputer for `model`.

    batch_size here is the column-chunk size of the KM construction (number
    of input pixels processed per inner batch), NOT a sample batch — the KM
    is built one sample at a time. 4096 is comfortable on an H100 for the
    1000 x 150529 ImageNet KM; lower it if a co-located GPU OOMs.
    """
    return KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)


# ---------------------------------------------------------------------------
# Core participation math (UNIT-TESTED on small synthetic ΔM — no heavy models)
# ---------------------------------------------------------------------------

def _participation_ratio(weights: torch.Tensor) -> float:
    """Participation ratio PR = (Σ w_i)² / Σ w_i² for a vector of non-negative
    weights. PR is the *effective count* of active entries: PR=1 when all the
    mass is on one entry, PR=k when k entries share the mass equally. Returns
    0.0 when the total mass is 0 (a ΔM that is exactly zero)."""
    w = weights.double()
    s1 = float(w.sum().item())
    s2 = float((w * w).sum().item())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def column_concentration(dM: torch.Tensor) -> dict:
    """Entrywise / column concentration of ΔM (shape C × (d+1)).

    Per-column mass c_j = ||ΔM[:, j]||₂² (= sum of squared entries down the
    column). Returns PR_col (effective # active columns), top1_col_frac
    (largest column mass / total), and nonzero_columns. Computed column-wise
    so it never materialises anything larger than ΔM itself.
    """
    # c_j = Σ_i ΔM[i, j]^2  — sum of squares down each column → length-(d+1).
    col_mass = (dM.double() ** 2).sum(dim=0)  # (d+1,)
    total = float(col_mass.sum().item())
    pr_col = _participation_ratio(col_mass)
    if total > 0.0:
        top1 = float((col_mass.max() / col_mass.sum()).item())
    else:
        top1 = 0.0
    nonzero = int((col_mass > 0.0).sum().item())
    return {
        "PR_col": pr_col,
        "top1_col_frac": top1,
        "nonzero_columns": nonzero,
    }


def spectral_concentration(dM: torch.Tensor, n_top: int = N_SIGMA_TOP) -> dict:
    """Spectral concentration of ΔM via the C×C Gram (NEVER a full SVD).

    G = ΔM ΔMᵀ is C×C (1000×1000 for ImageNet). Its eigenvalues λ_i equal the
    squared singular values of ΔM, so the nonzero spectrum is identical to a
    full SVD's — but we never form / decompose the C×(d+1) matrix, which would
    be infeasible. We use torch.linalg.eigvalsh (G is symmetric PSD), clamp
    tiny-negative eigenvalues from roundoff to 0, and report:
        PR_spec     = (Σ λ_i)² / Σ λ_i²   effective rank
        sigma_top10 = sqrt of the top-n eigenvalues (leading singular values)
    """
    G = dM.double() @ dM.double().T               # C × C Gram — the ONLY decomp input
    # G is symmetric PSD; eigvalsh is the correct, cheap route. NO svd on dM.
    eig = torch.linalg.eigvalsh(G)                # ascending eigenvalues (= σ²)
    eig = torch.clamp(eig, min=0.0)               # kill roundoff-negative λ
    pr_spec = _participation_ratio(eig)
    # Leading singular values = sqrt of the largest eigenvalues, descending.
    # Zero-pad to exactly n_top so the schema is fixed-length even when the
    # matrix has fewer than n_top rows (C < n_top). For ImageNet C=1000 so the
    # pad is never exercised; it only matters for the small unit-test matrices.
    top = torch.sort(eig, descending=True).values[:n_top]
    sigma_top = torch.sqrt(top).tolist()
    sigma_top += [0.0] * (n_top - len(sigma_top))
    return {
        "PR_spec": pr_spec,
        "sigma_top10": [float(s) for s in sigma_top],
    }


def visible_invisible_split(dM: torch.Tensor, df: torch.Tensor) -> dict:
    """Pythagoras split of ΔM's Frobenius energy along the all-ones direction.

    The row-sum identity gives Δf = ΔM·1 (the bias/ones column included), so
    the component of each row along the unit vector 1/√(d+1) carries energy
    ||Δf||₂² / (d+1). The orthogonal complement is the logit-INVISIBLE energy:
        invisible_mass = d_M² − d_f²/(d+1)
    which is ≥ 0 in exact arithmetic; we clamp tiny-negative roundoff to 0.
    Also returns A = (d_f/d_M)² (coherence; None when d_M == 0).
    """
    d1 = int(dM.shape[1])                          # = d + 1, the column count
    dM_f2 = float((dM.double() ** 2).sum().item())  # ||ΔM||_F²  = d_M²
    d_f = float(torch.linalg.norm(df.double()).item())
    d_M = float(dM_f2 ** 0.5)

    invisible = dM_f2 - (d_f * d_f) / d1
    if invisible < 0.0:                            # roundoff guard (Pythagoras ≥ 0)
        invisible = 0.0

    A_val: Optional[float] = float((d_f / d_M) ** 2) if d_M > 0.0 else None

    return {
        "d_f": d_f,
        "d_M": d_M,
        "A": A_val,
        "invisible_mass": invisible,
    }


def participation_record(dM: torch.Tensor, df: torch.Tensor, idx: int) -> dict:
    """Assemble the full per-pair participation record from ΔM and Δf."""
    rec = {"idx": int(idx)}
    rec.update(visible_invisible_split(dM, df))
    rec.update(column_concentration(dM))
    rec.update(spectral_concentration(dM))
    return rec


# ---------------------------------------------------------------------------
# Per-sample extraction (two KM forwards; 3D-input rule; eval mode via build_model)
# ---------------------------------------------------------------------------

def per_sample_record(kmc, x_clean: torch.Tensor, x_adv: torch.Tensor,
                      device: str, idx: int) -> dict:
    """Compute the participation record for one (clean, adv) pair.

    Inputs are 3D (C, H, W) — NO .unsqueeze(0) (knowledgematrix 3D-input rule,
    CLAUDE.md). Two KM forwards: M(x) and M(x'). ΔM = M(x') − M(x);
    Δf = ΔM·1 (equals f(x')−f(x) by the row-sum identity). We build Δf from
    the matrix row-sum so the visible/invisible split is internally consistent
    with the ΔM whose energy we are decomposing.
    """
    sample_c = x_clean.to(device).float()
    sample_a = x_adv.to(device).float()

    mat_c = kmc.forward(sample_c).detach()
    mat_a = kmc.forward(sample_a).detach()

    dM = (mat_a.double() - mat_c.double())   # C × (d+1)
    df = dM.sum(dim=1)                        # Δf = ΔM·1 = f(x')−f(x)  (length C)

    rec = participation_record(dM, df, idx)

    del mat_c, mat_a, dM, df, sample_c, sample_a
    return rec


# ---------------------------------------------------------------------------
# pairs.pth loading (identical convention to mask_hamming_trio.load_pairs)
# ---------------------------------------------------------------------------

def load_pairs(pairs_path: str):
    """Load stored (clean, adv) endpoints from a pairs.pth.

    Two producers save pairs in this repo with different key names:
      * generate_adversarial_pairs_scaleup.py (Step A2, the N5000 sets):
          keys x_clean / x_adv  (+ y_clean / y_adv / n_done)
      * validate_theorem45.py (the 200-pair Study-2 .pairs.pt):
          keys clean / adv      (+ attack / n_done)
    Accept either. Returns (clean, adv) CPU tensors of shape (N, C, H, W).
    """
    blob = torch.load(pairs_path, map_location="cpu")
    if not isinstance(blob, dict):
        raise ValueError(
            f"{pairs_path}: expected a dict of tensors, got {type(blob)}")
    if "x_clean" in blob and "x_adv" in blob:
        clean, adv = blob["x_clean"], blob["x_adv"]
    elif "clean" in blob and "adv" in blob:
        clean, adv = blob["clean"], blob["adv"]
    else:
        raise KeyError(
            f"{pairs_path}: no recognized endpoint keys; expected "
            f"('x_clean','x_adv') or ('clean','adv'), got {sorted(blob)}")
    clean = torch.as_tensor(clean)
    adv = torch.as_tensor(adv)
    if clean.shape[0] != adv.shape[0]:
        raise ValueError(
            f"{pairs_path}: clean/adv length mismatch "
            f"{clean.shape[0]} vs {adv.shape[0]}")
    return clean, adv


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _median(values) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.median(vals))


def build_summary(per_sample: list) -> dict:
    """Median participation statistics over all pairs.

    Medians are robust to the heavy-tailed coherence/concentration
    distributions. PR_col / top1_col_frac separate sparse (PR_col small,
    top1 large) from dense (PR_col large, top1 small) attacks; PR_spec is the
    effective rank of ΔM; invisible_mass is the logit-invisible Frobenius
    energy.
    """
    return {
        "n_total": len(per_sample),
        "median_d_f": _median(s["d_f"] for s in per_sample),
        "median_d_M": _median(s["d_M"] for s in per_sample),
        "median_A": _median(s["A"] for s in per_sample),
        "median_PR_col": _median(s["PR_col"] for s in per_sample),
        "median_top1_col_frac": _median(s["top1_col_frac"] for s in per_sample),
        "median_nonzero_columns": _median(
            s["nonzero_columns"] for s in per_sample),
        "median_PR_spec": _median(s["PR_spec"] for s in per_sample),
        "median_invisible_mass": _median(
            s["invisible_mass"] for s in per_sample),
        "note": (
            "PR_col = effective # active columns (sparse attack ⇒ small, dense "
            "⇒ large); top1_col_frac = mass in the hottest column; PR_spec = "
            "effective rank of ΔM (from the C×C Gram, not a full SVD); "
            "invisible_mass = d_M² − d_f²/(d+1) = logit-invisible Frobenius "
            "energy; A = (d_f/d_M)² coherence."
        ),
    }


# ---------------------------------------------------------------------------
# Main driver (mirrors mask_hamming_trio.run: checkpoint/resume + atomic save)
# ---------------------------------------------------------------------------

def run(arch: str, attack: str, pairs_path: str, out_path: str,
        device: Optional[str] = None, matrix_batch_size: int = 4096,
        max_pairs: Optional[int] = None) -> dict:
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported arch {arch!r}; supported: {SUPPORTED_ARCHS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Building {arch} KM wrapper on {device} (offline-safe, "
          f"torchvision DEFAULT weights)...", flush=True)
    model = build_model(arch, device)  # eval() + residual-device fix inside

    print(f"Loading pairs from {pairs_path} ...", flush=True)
    clean, adv = load_pairs(pairs_path)
    n = clean.shape[0]
    if max_pairs is not None:
        n = min(n, max_pairs)
    print(f"  {n} pairs (shape {tuple(clean.shape[1:])})", flush=True)

    kmc = make_kmc(model, device=device, batch_size=matrix_batch_size)

    out_file = Path(out_path)
    ckpt_file = out_file.with_suffix(".ckpt.json")

    # Resume from per-sample checkpoint.
    per_sample: list = []
    start_idx = 0
    existing = atomic_json_load(str(ckpt_file), default=None)
    if isinstance(existing, dict) and existing.get("arch") == arch \
            and existing.get("attack") == attack \
            and existing.get("num_pairs") == n:
        per_sample = existing.get("per_sample", [])
        start_idx = len(per_sample)
        if start_idx:
            print(f"  Resuming from checkpoint: {start_idx}/{n} done",
                  flush=True)

    for i in range(start_idx, n):
        rec = per_sample_record(kmc, clean[i], adv[i], device=device, idx=i)
        per_sample.append(rec)

        if torch.cuda.is_available() and (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i == n - 1 or i == start_idx:
            A_disp = "n/a" if rec["A"] is None else f"{rec['A']:.3e}"
            print(f"  [{i + 1}/{n}] d_f={rec['d_f']:.3e} d_M={rec['d_M']:.3e} "
                  f"A={A_disp} PR_col={rec['PR_col']:.2f} "
                  f"top1={rec['top1_col_frac']:.3f} "
                  f"nnz_col={rec['nonzero_columns']} "
                  f"PR_spec={rec['PR_spec']:.2f} "
                  f"inv_mass={rec['invisible_mass']:.3e}", flush=True)

        # Atomic per-sample checkpoint.
        atomic_json_dump(str(ckpt_file), {
            "arch": arch,
            "attack": attack,
            "num_pairs": n,
            "per_sample": per_sample,
        })

    summary = build_summary(per_sample)
    print(f"\n{'#' * 60}", flush=True)
    print(f"  SUMMARY  ({arch} / {attack})", flush=True)
    print(f"{'#' * 60}", flush=True)
    print(f"  pairs                  : {len(per_sample)}", flush=True)
    print(f"  median d_f             : {summary['median_d_f']}", flush=True)
    print(f"  median d_M             : {summary['median_d_M']}", flush=True)
    print(f"  median A               : {summary['median_A']}", flush=True)
    print(f"  median PR_col          : {summary['median_PR_col']}", flush=True)
    print(f"  median top1_col_frac   : {summary['median_top1_col_frac']}", flush=True)
    print(f"  median PR_spec         : {summary['median_PR_spec']}", flush=True)
    print(f"  median invisible_mass  : {summary['median_invisible_mass']}", flush=True)

    save_data = {
        "arch": arch,
        "attack": attack,
        "num_pairs": len(per_sample),
        "per_sample": per_sample,
        "summary": summary,
    }
    atomic_json_dump(str(out_file), save_data)
    print(f"\n  Saved: {out_file}", flush=True)

    # Drop the checkpoint now that the final file is the canonical deliverable.
    if ckpt_file.exists():
        try:
            ckpt_file.unlink()
        except OSError:
            pass

    return save_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(
        description="D5: per-sample ΔM participation (entrywise + spectral "
                    "concentration) for the trio — sparse-vs-dense attacks.")
    p.add_argument("--arch", required=True, choices=list(SUPPORTED_ARCHS))
    p.add_argument("--attack", required=True, choices=list(ATTACKS))
    p.add_argument("--pairs", required=True,
                   help="Path to the stored adversarial pairs.pth (A2 N5000 "
                        "set or a validate_theorem45 .pairs.pt).")
    p.add_argument("--n_pairs", type=int, default=100,
                   help="Pairs processed per cell (default 100).")
    p.add_argument("--temp_dir", default=None,
                   help="SLURM_TMPDIR (reserved for parity with sibling "
                        "scripts; pairs are read directly from --pairs).")
    p.add_argument("--out", default=None,
                   help="Output JSON. Default: experiments/{arch}_imagenet/"
                        "dm_participation/{attack}.json")
    p.add_argument("--matrix_batch_size", type=int, default=4096,
                   help="KM column-chunk size (lower if a co-located GPU OOMs).")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or (
        f"experiments/{args.arch}_imagenet/dm_participation/{args.attack}.json")
    print("D5 ΔM participation (trio)", flush=True)
    print(f"  arch    : {args.arch}", flush=True)
    print(f"  attack  : {args.attack}", flush=True)
    print(f"  pairs   : {args.pairs}", flush=True)
    print(f"  n_pairs : {args.n_pairs}", flush=True)
    print(f"  out     : {out_path}", flush=True)
    t0 = time.perf_counter()
    run(arch=args.arch, attack=args.attack, pairs_path=args.pairs,
        out_path=out_path, device=args.device,
        matrix_batch_size=args.matrix_batch_size, max_pairs=args.n_pairs)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
