"""Plan B / D4 — mask-Hamming for the trio (crossing-mechanism, Prop 13.5).

Generalizes the VGG diagnostic (legacy/debug_vgg_gamma_zero.py) to the
Phase-3 trio (ResNet-152 / DenseNet-121 / GoogLeNet). For each stored
adversarial pair it produces the per-sample data that tests the
*conditional crossing mechanism* behind the attack-family ordering /
coherence statistic (Study 2, Prop 13.5):

    d_f                    : logit L2 distance       ||f(x) - f(x')||
    d_M                    : KM Frobenius distance   ||M(x) - M(x')||
    A = (d_f / d_M)^2      : coherence (metric-invariant; guarded at d_M=0)
    relu_hamming           : # ReLU sign-pattern flips between clean & adv
    relu_pattern_len       : total ReLU units (the Hamming denominator)
    maxpool_mismatches     : # MaxPool argmax positions that changed
    maxpool_total_positions: total MaxPool argmax positions

The size-controlled crossing-mechanism statistic is the partial Spearman
correlation rho(A, relu_hamming | d_f) computed over the attack-success
subset (d_f >= 1, d_M > 0). The VGG mechanism pilot found this NEGATIVE
(-0.30..-0.34) after controlling for perturbation size; D4 inherits that
analysis plan and reports the same statistic per (arch, attack).

This script does NOT regenerate attacks. It reads the stored A2 endpoints
(experiments/{arch}_imagenet/.../pairs.pth) and runs ONE KM forward per
endpoint.

Branch-ReLU coverage (Critic 4's flag)
--------------------------------------
get_relu_sign_pattern was validated on VGG (no branches). DenseNet (concat
skips) and GoogLeNet (inception branches) have branch ReLUs. In the
knowledgematrix wrappers every branch is *linearized* into the flat
model.layers ModuleList (concat_skip / branch_input wiring in
neural_net.py), so every nn.ReLU is visited by the save=True forward and
gets a populated pre_acts[i]. check_relu_coverage() verifies this empirically
and is the gate for which archs are supported. Measured locally (env/ Py3.11,
torch 2.6.0): resnet152 151/151, densenet121 121/121, googlenet 57/57 ReLUs
covered — all three FULLY covered, all three supported.

NOTE on DenseNet maxpool: the pretrained densenet121 wrapper DOES have one
MaxPool2d (the stem pool0), so maxpool_total_positions is nonzero for it.
The per-sample extraction handles 0 or >0 maxpools generically — no arch is
special-cased.

Usage:
    python mask_hamming_trio.py --arch resnet152 --attack PGD \
        --pairs experiments/resnet152_imagenet/adversarial_pairs_N5000/pgd/pairs.pth
    python mask_hamming_trio.py --arch googlenet --attack FGSM \
        --pairs <pairs.pth> --temp_dir $SLURM_TMPDIR
"""
from __future__ import annotations

import math
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.km_models import build_model
from utils.atomic_io import atomic_json_dump, atomic_json_load


SUPPORTED_ARCHS = ("resnet152", "densenet121", "googlenet")
ATTACKS = ("FGSM", "PGD", "CW", "DeepFool", "APGD", "Square")

# Attack-success filter for the summary statistic (matches the VGG mechanism
# pilot / CLAUDE.md "Facts established 2026-06-11": d_f >= 1 success filter,
# d_M > 0 to avoid a zero coherence denominator).
DF_SUCCESS_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Model / KM helpers
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
# Activation-pattern snapshotting (lifted from debug_vgg_gamma_zero.py and
# generalized — the iteration is identical because branches are flattened
# into model.layers in the knowledgematrix wrappers).
# ---------------------------------------------------------------------------

def get_relu_sign_pattern(model: nn.Module) -> torch.Tensor:
    """Flatten the sign pattern of ALL ReLU pre-activations in model.layers.

    Must be called IMMEDIATELY after kmc.forward(x): the pre_acts list is
    re-allocated on every save=True forward (neural_net.py:556), so a later
    forward overwrites it.

    Counts ReLU only (not Sigmoid/Tanh/etc.) to match the piecewise-linear
    region argument: the relevant region boundary for the trio is the ReLU
    sign pattern plus the MaxPool argmax. Returns a 1-D bool tensor on CPU.
    """
    parts = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.ReLU):
            if i < len(model.pre_acts) and model.pre_acts[i] is not None:
                parts.append((model.pre_acts[i] > 0).flatten().cpu())
    if not parts:
        return torch.zeros(0, dtype=torch.bool)
    return torch.cat(parts)


def get_maxpool_argmax_snapshot(model: nn.Module) -> torch.Tensor:
    """Flatten the argmax indices of every populated MaxPool layer.

    maxpool_indices[i] is populated in the save=True forward
    (neural_net.py:593-594). Returns a 1-D long tensor on CPU (empty if the
    model has no maxpools).
    """
    parts = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            if i < len(model.maxpool_indices) and model.maxpool_indices[i] is not None:
                parts.append(model.maxpool_indices[i].flatten().cpu())
    if not parts:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Branch-ReLU coverage gate (Critic 4's flag)
# ---------------------------------------------------------------------------

def check_relu_coverage(model: nn.Module,
                        input_shape: Sequence[int] = (3, 224, 224)) -> dict:
    """Verify every nn.ReLU in model.layers gets a populated pre_acts entry.

    Runs ONE save=True forward on a small valid 3D ImageNet-shaped input and
    counts: (# nn.ReLU modules) vs (# ReLU indices with non-None pre_acts),
    plus the same for maxpools. `full_coverage` is True iff there is at least
    one ReLU and all ReLUs are covered.

    This is the supported-arch gate: an arch with uncovered branch ReLUs MUST
    be excluded (a silent undercount of the Hamming denominator would corrupt
    the crossing-mechanism statistic). Returns a report dict.
    """
    C, H, W = input_shape
    x = torch.randn(C, H, W, device=getattr(model, "device", "cpu"))
    was_saving = getattr(model, "save", False)
    model.save = True
    try:
        _ = model.forward(x)
    finally:
        model.save = was_saving

    relu_idx = [i for i, l in enumerate(model.layers) if isinstance(l, nn.ReLU)]
    relu_cov = [i for i in relu_idx
                if i < len(model.pre_acts) and model.pre_acts[i] is not None]
    mp_idx = [i for i, l in enumerate(model.layers)
              if isinstance(l, (nn.MaxPool2d, nn.AdaptiveMaxPool2d))]
    mp_cov = [i for i in mp_idx
              if i < len(model.maxpool_indices)
              and model.maxpool_indices[i] is not None]

    relu_units = int(get_relu_sign_pattern(model).numel())
    mp_positions = int(get_maxpool_argmax_snapshot(model).numel())

    return {
        "n_relu": len(relu_idx),
        "n_relu_covered": len(relu_cov),
        "uncovered_relu_indices": [i for i in relu_idx if i not in set(relu_cov)],
        "relu_units": relu_units,
        "n_maxpool": len(mp_idx),
        "n_maxpool_covered": len(mp_cov),
        "maxpool_positions": mp_positions,
        "full_coverage": len(relu_idx) > 0 and len(relu_idx) == len(relu_cov),
    }


# ---------------------------------------------------------------------------
# Per-sample extraction
# ---------------------------------------------------------------------------

def per_sample_record(kmc, model: nn.Module, x_clean: torch.Tensor,
                      x_adv: torch.Tensor, device: str, idx: int) -> dict:
    """Compute the per-sample crossing-mechanism record for one (clean, adv).

    Inputs are 3D (C, H, W) — NO .unsqueeze(0) (knowledgematrix 3D-input
    rule, CLAUDE.md). One KM forward per endpoint; the sign-pattern / argmax
    snapshots are read immediately after each forward (the pre_acts list is
    overwritten by the next forward).

    A = (d_f / d_M)^2 is guarded: when d_M == 0 (clean & adv in the same
    linear region) A is set to None (JSON null) rather than inf/NaN.
    """
    sample_c = x_clean.to(device).float()
    sample_a = x_adv.to(device).float()

    # Clean endpoint
    mat_c = kmc.forward(sample_c)
    out_c = kmc.current_output.detach().clone()
    pat_c = get_relu_sign_pattern(model)
    mp_c = get_maxpool_argmax_snapshot(model)

    # Adversarial endpoint
    mat_a = kmc.forward(sample_a)
    out_a = kmc.current_output.detach().clone()
    pat_a = get_relu_sign_pattern(model)
    mp_a = get_maxpool_argmax_snapshot(model)

    d_f = float(torch.linalg.norm(out_c.double() - out_a.double()).item())
    d_M = float(torch.linalg.norm(mat_c.double() - mat_a.double()).item())

    # Coherence A = (d_f / d_M)^2, guarded against d_M == 0.
    if d_M > 0.0:
        A_val: Optional[float] = float((d_f / d_M) ** 2)
    else:
        A_val = None

    if pat_c.numel() != pat_a.numel():
        relu_hamming, pat_len = -1, -1
    else:
        relu_hamming = int((pat_c != pat_a).sum().item())
        pat_len = int(pat_c.numel())

    if mp_c.numel() != mp_a.numel():
        mp_mismatch, mp_len = -1, -1
    else:
        mp_mismatch = int((mp_c != mp_a).sum().item())
        mp_len = int(mp_c.numel())

    rec = {
        "idx": int(idx),
        "d_f": d_f,
        "d_M": d_M,
        "A": A_val,
        "relu_hamming": relu_hamming,
        "relu_pattern_len": pat_len,
        "maxpool_mismatches": mp_mismatch,
        "maxpool_total_positions": mp_len,
    }

    del mat_c, mat_a, out_c, out_a, pat_c, pat_a, mp_c, mp_a, sample_c, sample_a
    return rec


# ---------------------------------------------------------------------------
# Statistics — partial Spearman rho(A, H | z)
# ---------------------------------------------------------------------------

def partial_spearman(A, H, z) -> Optional[float]:
    """Partial Spearman correlation rho(A, H | z).

    Spearman = Pearson on ranks. The partial version rank-transforms all
    three variables, OLS-regresses rank(A) and rank(H) each on [1, rank(z)],
    and Pearson-correlates the residuals. This is the size-controlled
    crossing-mechanism statistic: A = coherence, H = relu_hamming, z = d_f
    (the perturbation-size confound). Returns None on degenerate input
    (< 3 finite points, or any variable constant after ranking).
    """
    from scipy.stats import rankdata

    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    z = np.asarray(z, dtype=float)
    if not (len(A) == len(H) == len(z)):
        raise ValueError("partial_spearman: A, H, z must be equal length")

    mask = np.isfinite(A) & np.isfinite(H) & np.isfinite(z)
    A, H, z = A[mask], H[mask], z[mask]
    if len(A) < 3:
        return None

    ra = rankdata(A)
    rh = rankdata(H)
    rz = rankdata(z)
    # Constant after ranking (all ties) -> undefined correlation.
    if np.ptp(ra) == 0 or np.ptp(rh) == 0 or np.ptp(rz) == 0:
        return None

    X = np.column_stack([np.ones_like(rz), rz])

    def _resid(y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    ea = _resid(ra)
    eh = _resid(rh)
    if np.ptp(ea) == 0 or np.ptp(eh) == 0:
        return None
    denom = math.sqrt(float(np.dot(ea, ea)) * float(np.dot(eh, eh)))
    if denom == 0.0:
        return None
    return float(np.dot(ea, eh) / denom)


# ---------------------------------------------------------------------------
# pairs.pth loading (handles both stored key conventions)
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

def build_summary(per_sample: list, df_threshold: float = DF_SUCCESS_THRESHOLD) -> dict:
    """Apply the attack-success filter and compute the crossing-mechanism stat.

    Filter: d_f >= df_threshold AND d_M > 0 AND A is not None AND
    relu_hamming >= 0. On the surviving subset report
    partial_spearman(A, relu_hamming | d_f) — expected NEGATIVE per the VGG
    mechanism pilot. Also reports the size-confounded raw Spearman(A, H) for
    contrast (the pilot showed the raw correlation is positive/size-driven
    and only the partial flips negative).
    """
    A_vals, H_vals, z_vals = [], [], []
    for s in per_sample:
        if s["A"] is None:
            continue
        if s["d_M"] <= 0.0:
            continue
        if s["relu_hamming"] < 0:
            continue
        if s["d_f"] < df_threshold:
            continue
        A_vals.append(s["A"])
        H_vals.append(s["relu_hamming"])
        z_vals.append(s["d_f"])

    n_success = len(A_vals)
    raw_spearman = None
    partial = None
    if n_success >= 3:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(A_vals, H_vals)
        raw_spearman = float(rho) if np.isfinite(rho) else None
        partial = partial_spearman(A_vals, H_vals, z_vals)

    # d_M == 0 diagnostics (the VGG gamma=0 mechanism: same linear region).
    same_region = [s["idx"] for s in per_sample
                   if s["d_M"] == 0.0 and s["relu_hamming"] == 0
                   and s["maxpool_mismatches"] == 0]
    d_M_zero = [s["idx"] for s in per_sample if s["d_M"] == 0.0]

    return {
        "n_total": len(per_sample),
        "df_success_threshold": float(df_threshold),
        "n_success_filtered": n_success,
        "partial_spearman_A_hamming_given_df": partial,
        "raw_spearman_A_hamming": raw_spearman,
        "partial_spearman_note": (
            "Size-controlled crossing-mechanism statistic "
            "rho(A, relu_hamming | d_f) on the d_f>=1, d_M>0 subset. "
            "Expected NEGATIVE (VGG pilot: -0.30..-0.34). A=(d_f/d_M)^2."
        ),
        "d_M_exact_zero_indices": d_M_zero,
        "n_d_M_exact_zero": len(d_M_zero),
        "same_linear_region_indices": same_region,
    }


# ---------------------------------------------------------------------------
# Main driver
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

    # Coverage gate — fail loudly rather than silently undercount masks.
    cov = check_relu_coverage(model, input_shape=(3, 224, 224))
    print(f"  ReLU coverage: {cov['n_relu_covered']}/{cov['n_relu']} "
          f"(units={cov['relu_units']}) | maxpool "
          f"{cov['n_maxpool_covered']}/{cov['n_maxpool']} "
          f"(positions={cov['maxpool_positions']})", flush=True)
    if not cov["full_coverage"]:
        raise AssertionError(
            f"{arch}: branch-ReLU coverage FAILED "
            f"({cov['n_relu'] - cov['n_relu_covered']} uncovered: "
            f"{cov['uncovered_relu_indices']}). Refusing to run — a silent "
            f"undercount of the Hamming denominator would corrupt the "
            f"crossing-mechanism statistic. Exclude this arch.")

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
    relu_pattern_len = None
    maxpool_total_positions = None
    start_idx = 0
    existing = atomic_json_load(str(ckpt_file), default=None)
    if isinstance(existing, dict) and existing.get("arch") == arch \
            and existing.get("attack") == attack \
            and existing.get("num_pairs") == n:
        per_sample = existing.get("per_sample", [])
        relu_pattern_len = existing.get("relu_pattern_len")
        maxpool_total_positions = existing.get("maxpool_total_positions")
        start_idx = len(per_sample)
        if start_idx:
            print(f"  Resuming from checkpoint: {start_idx}/{n} done",
                  flush=True)

    for i in range(start_idx, n):
        rec = per_sample_record(kmc, model, clean[i], adv[i],
                                device=device, idx=i)
        if relu_pattern_len is None and rec["relu_pattern_len"] >= 0:
            relu_pattern_len = rec["relu_pattern_len"]
        if maxpool_total_positions is None and rec["maxpool_total_positions"] >= 0:
            maxpool_total_positions = rec["maxpool_total_positions"]
        per_sample.append(rec)

        if torch.cuda.is_available() and (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i == n - 1 or i == start_idx:
            A_disp = "n/a" if rec["A"] is None else f"{rec['A']:.3e}"
            print(f"  [{i + 1}/{n}] d_f={rec['d_f']:.3e} d_M={rec['d_M']:.3e} "
                  f"A={A_disp} relu_hamming={rec['relu_hamming']}/"
                  f"{rec['relu_pattern_len']} maxpool={rec['maxpool_mismatches']}"
                  f"/{rec['maxpool_total_positions']}", flush=True)

        # Atomic per-sample checkpoint.
        atomic_json_dump(str(ckpt_file), {
            "arch": arch,
            "attack": attack,
            "num_pairs": n,
            "relu_pattern_len": relu_pattern_len,
            "maxpool_total_positions": maxpool_total_positions,
            "per_sample": per_sample,
        })

    summary = build_summary(per_sample)
    print(f"\n{'#' * 60}", flush=True)
    print(f"  SUMMARY  ({arch} / {attack})", flush=True)
    print(f"{'#' * 60}", flush=True)
    print(f"  pairs                          : {len(per_sample)}", flush=True)
    print(f"  relu_pattern_len               : {relu_pattern_len}", flush=True)
    print(f"  maxpool_total_positions        : {maxpool_total_positions}", flush=True)
    print(f"  n success-filtered (d_f>=1)    : {summary['n_success_filtered']}", flush=True)
    print(f"  partial Spearman(A,H | d_f)    : "
          f"{summary['partial_spearman_A_hamming_given_df']}", flush=True)
    print(f"  raw Spearman(A,H)              : "
          f"{summary['raw_spearman_A_hamming']}", flush=True)
    print(f"  n with d_M == 0 exactly        : {summary['n_d_M_exact_zero']}", flush=True)

    save_data = {
        "arch": arch,
        "attack": attack,
        "num_pairs": len(per_sample),
        "relu_pattern_len": relu_pattern_len,
        "maxpool_total_positions": maxpool_total_positions,
        "coverage": cov,
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
        description="D4: per-sample mask-Hamming / coherence for the trio "
                    "(crossing-mechanism, Prop 13.5).")
    p.add_argument("--arch", required=True, choices=list(SUPPORTED_ARCHS))
    p.add_argument("--attack", required=True, choices=list(ATTACKS))
    p.add_argument("--pairs", required=True,
                   help="Path to the stored adversarial pairs.pth (A2 N5000 "
                        "set or a validate_theorem45 .pairs.pt).")
    p.add_argument("--temp_dir", default=None,
                   help="SLURM_TMPDIR (reserved for parity with sibling "
                        "scripts; pairs are read directly from --pairs).")
    p.add_argument("--out", default=None,
                   help="Output JSON. Default: "
                        "experiments/{arch}_imagenet/mask_hamming/{attack}.json")
    p.add_argument("--matrix_batch_size", type=int, default=4096,
                   help="KM column-chunk size (lower if a co-located GPU OOMs).")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap pairs processed (smoke testing).")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or (
        f"experiments/{args.arch}_imagenet/mask_hamming/{args.attack}.json")
    print("D4 mask-Hamming (trio)", flush=True)
    print(f"  arch   : {args.arch}", flush=True)
    print(f"  attack : {args.attack}", flush=True)
    print(f"  pairs  : {args.pairs}", flush=True)
    print(f"  out    : {out_path}", flush=True)
    t0 = time.perf_counter()
    run(arch=args.arch, attack=args.attack, pairs_path=args.pairs,
        out_path=out_path, device=args.device,
        matrix_batch_size=args.matrix_batch_size, max_pairs=args.max_pairs)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
