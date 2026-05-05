"""Emit paper-ready LaTeX tables from aggregated S1/S2/S3 results.

Each table cell holds: mean ± half-CI; per-cell 95% bootstrap CI is taken
over the 50 teleports (S1) or 5000 pairs (S3) or N=25K samples (S2).
"""
import json
from pathlib import Path
from typing import Dict


def _fmt(v: float, ci: tuple = None, decimals: int = 3) -> str:
    """Format 'mean (lo, hi)' as LaTeX."""
    if ci is None:
        return f"${v:.{decimals}f}$"
    return f"${v:.{decimals}f}\\;[{ci[0]:.{decimals}f},\\,{ci[1]:.{decimals}f}]$"


def emit_s1_table(s1_results: Dict, archs, num_teleports, out_path: str):
    """S1 table: rows = measures (with invariance class), columns = archs.

    Each cell: mean across teleports ± 95% bootstrap CI of the mean.
    """
    from cka_similarity.reduce.statistics import bootstrap_ci

    measure_invariances = {
        "debiased_cka":   "orth + iso-scale",
        "angular_cka":    "orth + iso-scale",
        "procrustes":     "orth",
        "bures":          "orth",
        "soft_matching":  "permutation",
        "rsa":            "rotation + monotone",
        "output_jsd":     "(none, functional)",
        "gw":             "isometry",
        "dcor":           "translation + orth",
    }

    lines = []
    lines.append(r"\begin{tabular}{l l " + "r " * len(archs) + "}")
    lines.append(r"\toprule")
    lines.append("Measure & Invariance class & " + " & ".join(archs) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"KM Frobenius (theorem) & quiver-iso & " + " & ".join([r"$\mathbf{0}$"] * len(archs)) + r" \\")

    for mname, invariance in measure_invariances.items():
        cells = []
        for arch in archs:
            vals = [s1_results[(arch, tid)][mname]["value"]
                    for tid in range(num_teleports)
                    if (arch, tid) in s1_results and mname in s1_results[(arch, tid)]]
            if not vals:
                cells.append("--")
                continue
            mean = sum(vals) / len(vals)
            lo, hi = bootstrap_ci(vals)
            cells.append(_fmt(mean, ci=(lo, hi)))
        lines.append(f"{mname.replace('_', ' ')} & {invariance} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def emit_s2_table(s2_results: Dict, out_path: str):
    """S2 table: rows = arch pairs, columns = measures + KM Frobenius mean."""
    measure_order = ["debiased_cka", "angular_cka", "procrustes", "bures",
                     "soft_matching", "rsa", "output_jsd", "gw", "dcor"]
    lines = []
    lines.append(r"\begin{tabular}{l " + "r " * (len(measure_order) + 1) + "}")
    lines.append(r"\toprule")
    lines.append("Pair & KM Frob & " + " & ".join(measure_order) + r" \\")
    lines.append(r"\midrule")
    for pname, d in s2_results.items():
        cells = [_fmt(d.get("km_mean", float("nan")))]
        for mname in measure_order:
            v_d1 = d.get("D1", {}).get(mname, {}).get("value")
            v_d2 = d.get("D2", {}).get(mname, {}).get("value")
            v = v_d1 if v_d1 is not None else v_d2
            cells.append(_fmt(v) if v is not None else "--")
        lines.append(f"{pname} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def emit_s3_table(s3_results: Dict, out_path: str):
    """S3 table: amplification factors (d_M / d_h, d_M / d_f) per (arch, attack)."""
    lines = []
    lines.append(r"\begin{tabular}{l l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Arch & Attack & $d_M/d_f$ mean & $d_M/d_h$ mean & N pairs \\")
    lines.append(r"\midrule")
    for (arch, attack), d in s3_results.items():
        per_pair = d["per_pair"]
        if not per_pair:
            continue
        amp_M_f = [p["d_M"] / p["d_f"] for p in per_pair if p["d_f"] > 1e-6]
        amp_M_h = [p["d_M"] / p["d_h"] for p in per_pair if p["d_h"] > 1e-6]
        mean_amp_M_f = sum(amp_M_f) / max(1, len(amp_M_f))
        mean_amp_M_h = sum(amp_M_h) / max(1, len(amp_M_h))
        lines.append(f"{arch} & {attack} & {_fmt(mean_amp_M_f)} & {_fmt(mean_amp_M_h)} & {len(per_pair)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
