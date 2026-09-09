#!/usr/bin/env python
"""Regenerate the HAaNE I attack-family ordering tables from the on-disk reduces.

NOTATION: the stored JSON keys are "d_f" / "d_f_rms" (unchanged, they name data on disk),
but the paper writes this quantity as d_\Psi, since the network function is \Psi(W,f).
LaTeX-emitting strings below therefore say d_\Psi while the data access says p["d_f"].

Two reduces of the per-pair ratio d_M/d_f exist on disk and the paper draws on both:

  * Phase-1 S3 reduce  -- cluster-data/results/phase1/aggregated/s3_results.json
        trio only (resnet152 / densenet121 / googlenet), per-pair d_f / d_h / d_M stored
        raw and RMS-rescaled with the pair index, M.1 = f residual recorded per pair.
        1000 stored pairs per (arch, attack) cell EXCEPT resnet152/deepfool, which stopped
        at 512.  Feeds the appendix panel  tables/s3_table.tex (\\label{tab:s3_ordering_panel}).
  * theorem45 reduce   -- experiments/<exp>/theorem45/theorem45_results.json
        all six raters (trio + ResNet-18 / AlexNet / VGG), N = 200 pairs per cell,
        AGGREGATES ONLY (amplification_M_median = median d_M/d_f over the pairs passing the
        guard d_f > max(1e-12, 1e-6 * max_i d_f_i), validate_theorem45.py:643-645).  Nothing in
        the current code strips per-pair data: the worker builds a per_pair_raw block and
        _save_per_attack_file / aggregate_theorem45 pass the result dict through verbatim.  The
        SHIPPED files simply predate that worker -- they also lack amplification_{M,h}_{std,iqr},
        five keys short of what the current code writes -- so what is on disk is per-cell
        aggregates only, and no per-pair ratio survives to resample.  Feeds tables/table_coherence.tex,
        tables/table_coherence_max.tex and the six-rater tables/table_ordering.tex
        (\\label{tab:ordering}) via resubmission-artifacts-2026-06-11/scripts/make_study2_tables.py.
        The ONLY theorem45 cells that retain per-pair records are the three VGG diagnostic
        dumps experiments/vgg_imagenet/theorem45/debug_gamma_zero_{APGD,DeepFool,Square}.json
        (legacy/debug_vgg_gamma_zero.py); this script uses them for the filter-sensitivity
        report, never to recompute a table.

POPULATION MATCHING (--population, default `common`).  Because resnet152/deepfool stopped at
512 pairs, comparing per-cell medians across the six ResNet-152 cells at face value compares
medians over DIFFERENT image populations.  Restricted to the common pair_idx population the
ResNet-152 CW median moves +12.9 % and Square +8.5 % while the untruncated FGSM and PGD cells
move < 1 %: the apparent CW < FGSM inversion in the unmatched panel is a sampling artifact.
`--population common` (the default, and what the paper prints) intersects the six cells'
stored pair indices per architecture before filtering; `--population all` reproduces the
unmatched computation for comparison only.

Statistic: per (architecture, attack) the per-pair MEDIAN of d_M/d_f over the pairs kept by
--convention, attacks ranked DESCENDING (most knowledge-matrix amplification = lowest
coherence A = (d_f/d_M)^2).  Ranks are invariant under the raw <-> RMS rescaling because the
RMS factors are per-architecture constants.  All three conventions are computed and reported.

UNCERTAINTY.  Per cell: the median with a seeded BCa 95 % bootstrap CI over pairs (B settable,
default 20000).  Per adjacent rank pair: the CI of the median difference, computed two ways --
independent resampling within each cell, and paired resampling of the common pair indices (the
matched-population design).  An adjacent gap counts as RESOLVED only when BOTH CIs exclude
zero, i.e. the verdict is the conservative one.  These CIs cover the ORDERING panel only; the
Table~\\ref{tab:coherence} coherence magnitudes come from the theorem45 reduce, whose per-pair
ratios are not on disk, so they cannot be given CIs without re-running validate_theorem45.py.

Concordance: Kendall's W with midranks and the tie correction,
    W = 12 S / ( m^2 (n^3 - n) - m * sum_j T_j ),   T_j = sum_groups (t^3 - t),
which reduces to the standard 12 S / (m^2 (n^3 - n)) when no rater has ties (the case here),
with the EXACT permutation p-value for m = 3 (W is invariant under a common relabelling of the
items, so fixing one rater and enumerating the other two covers the whole (6!)^2 null) and a
seeded Monte-Carlo p-value for m = 6.  The m = 3 null is supported on 77 lattice points spaced
0.0127 apart, so trio W values are quoted to two decimals.
Leave-one-architecture-out (LOO) Spearman: Spearman rank correlation (midranks) between one
rater's ranks and the mean consensus rank of the remaining raters -- exactly the recipe of
make_study2_tables.py, whose stored six-rater values (W = 0.9206, LOO 0.7714 / 0.9429 / 0.9276 /
0.9856 / 0.9429 / 0.9429) are reproduced here as an ENFORCED self-check.

Every recorded self-check is enforced: any failure exits non-zero and no table is left behind
claiming a number the data does not support.  In particular the script refuses to print an
ordering whose adjacent medians differ by less than --min-gap-rel (default 1 %), and it refuses
to run at all if the theorem45 JSONs are missing (an earlier version fell back to the committed
printed rows, which laundered the table back in as its own rater input).

Outputs (all in --out-dir; nothing is written into the paper tree):
    regen_ordering_summary.json      every number, with provenance
    s3_table.tex                     regenerated appendix panel (trio, Phase-1 S3 reduce)
    table_ordering_variantA.tex      six-rater headline, all rows from the theorem45 reduce
                                     (= the committed table's rows; W = 0.921) -- the shipped one
    table_ordering_variantB.tex      six-rater headline with the TRIO rows replaced by the
                                     Phase-1 S3 rows.  NOT the paper's table (Ruling C-R8 keeps
                                     the headline on one reduce); emitted only so the rejected
                                     option stays inspectable.

Usage (from the repo root, with the repo's env; --date is required for byte-reproducibility):
    ./env/bin/python scripts/regen_ordering_tables.py --out-dir /path/to/out --date 2026-09-05
"""
import argparse
import datetime as _dt
import itertools
import json
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S3_DEFAULT = os.path.join(REPO, "cluster-data", "results", "phase1", "aggregated", "s3_results.json")
T45_ROOT_DEFAULT = os.path.join(REPO, "experiments")

# Display order of the attacks (same as make_study2_tables.py) and the two key spellings.
ATTACKS = ["FGSM", "PGD", "CW", "DeepFool", "APGD", "Square"]
S3_ATTACK_KEY = {"FGSM": "fgsm", "PGD": "pgd", "CW": "cw", "DeepFool": "deepfool",
                 "APGD": "apgd", "Square": "square"}
TRIO = OrderedDict([("ResNet-152", "resnet152"), ("DenseNet-121", "densenet121"),
                    ("GoogLeNet", "googlenet")])
T45_EXP = OrderedDict([("ResNet-152", "resnet152_imagenet"), ("DenseNet-121", "densenet121_imagenet"),
                       ("GoogLeNet", "googlenet_imagenet"), ("ResNet-18", "resnet_imagenet"),
                       ("AlexNet", "alexnet_imagenet"), ("VGG", "vgg_imagenet")])
ORDERING_ONLY = ["ResNet-18", "AlexNet", "VGG"]
FAMILY_TOP = ["DeepFool", "CW", "Square"]
FAMILY_MID = "FGSM"
FAMILY_BOTTOM = ["PGD", "APGD"]

CONVENTIONS = OrderedDict([
    ("unfiltered", "d_f > 0, no attack-success filter"),
    ("df1", "d_f >= 1"),
    ("df1_dM0", "d_f >= 1 and d_M > 0, the stricter attack-success filter "
                "(used by the mechanism pilot and this panel, NOT by tab:coherence)"),
])
CONVENTION_TEX = {
    "unfiltered": r"the unfiltered pairs ($d_\Psi>0$)",
    "df1": r"the pairs with $d_\Psi\ge1$",
    "df1_dM0": r"the attack-success-filtered pairs ($d_\Psi\ge1$, $d_M>0$)",
}
POPULATIONS = OrderedDict([
    ("common", "per architecture, the intersection of the six cells' stored pair indices"),
    ("all", "every stored pair of each cell (UNMATCHED across cells; comparison only)"),
])

# The rows printed in the committed six-rater table (2026-06-24 file); the theorem45 reduce
# must reproduce them exactly.  There is deliberately NO committed-S3-panel constant: the
# committed appendix rows were stale, the panel is regenerated from the data every run, and a
# constant recording a superseded row set is a self-check that can only mislead.
COMMITTED_TAB_ORDERING = OrderedDict([
    ("ResNet-152", ["Square", "DeepFool", "CW", "FGSM", "PGD", "APGD"]),
    ("DenseNet-121", ["DeepFool", "CW", "Square", "FGSM", "APGD", "PGD"]),
    ("GoogLeNet", ["DeepFool", "Square", "CW", "FGSM", "PGD", "APGD"]),
    ("ResNet-18", ["DeepFool", "Square", "CW", "FGSM", "APGD", "PGD"]),
    ("AlexNet", ["DeepFool", "CW", "Square", "FGSM", "APGD", "PGD"]),
    ("VGG", ["DeepFool", "CW", "Square", "FGSM", "APGD", "PGD"]),
])
STORED_SIX_RATER = {  # resubmission-artifacts-2026-06-11/study2_tables/study2_tables_data.json
    "kendall_W": 0.9206349206349206,
    "loo": {"ResNet-152": 0.7714285714285715, "DenseNet-121": 0.9428571428571428,
            "GoogLeNet": 0.9276336570439175, "ResNet-18": 0.9856107606091623,
            "AlexNet": 0.9428571428571428, "VGG": 0.9428571428571428},
}
# VGG is the only rater with per-pair records on disk, and only for these three attacks.
VGG_DEBUG_ATTACKS = ["APGD", "DeepFool", "Square"]

# The Phase-1 generator's attack budgets are read out of its source rather than retyped: an
# earlier caption said "PGD 7, CW 100, DeepFool 100 steps, otherwise the torchattacks defaults"
# and silently omitted the APGD and Square overrides.  Deriving the clause makes that class of
# error impossible.  cls_map lives in generate_adversarial_pairs_scaleup.get_attack.
S3_GENERATOR = "generate_adversarial_pairs_scaleup.py"
TA_PARAM_TEX = {"steps": r"\texttt{steps}", "n_queries": r"\texttt{n\_queries}",
                "loss": r"loss", "eps": r"$\varepsilon$", "alpha": r"$\alpha$",
                "overshoot": r"\texttt{overshoot}", "c": r"$c$", "kappa": r"$\kappa$",
                "lr": r"\texttt{lr}"}


class Checks:
    """Every recorded self-check is enforced: failures make the run exit non-zero."""

    def __init__(self):
        self.items = OrderedDict()
        self.failed = []

    def require(self, name, ok, detail=None):
        self.items[name] = OrderedDict([("passed", bool(ok)), ("detail", detail)])
        if not ok:
            self.failed.append(name)
        return bool(ok)

    def note(self, name, value):
        self.items[name] = OrderedDict([("passed", True), ("detail", value)])

    def enforce(self):
        if self.failed:
            sys.stderr.write("SELF-CHECK FAILURES (%d):\n" % len(self.failed))
            for n in self.failed:
                sys.stderr.write("  - %s: %s\n" % (n, self.items[n]["detail"]))
            sys.stderr.write("No table is trustworthy under a failed self-check; exiting 3.\n")
            sys.exit(3)


def die(msg, code=2):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(code)


# ----------------------------------------------------------------------------- statistics

def keep_pair(p, convention):
    """Attack-success filter applied to one stored pair (raw units: d_f = logit l2, d_M = KM Frobenius)."""
    df, dM = float(p["d_f"]), float(p["d_M"])
    if convention == "unfiltered":
        return df > 0
    if convention == "df1":
        return df >= 1
    if convention == "df1_dM0":
        return (df >= 1) and (dM > 0)
    raise ValueError(convention)


def cell_stats(per_pair, convention, keep_idx=None):
    """Per-cell summary.  `keep_idx`: restrict to these pair indices before filtering."""
    pp = per_pair if keep_idx is None else [p for p in per_pair if p["pair_idx"] in keep_idx]
    kept = [p for p in pp if keep_pair(p, convention)]
    r = np.asarray([p["d_M"] / p["d_f"] for p in kept], dtype=float)
    r_rms = np.asarray([p["d_M_rms"] / p["d_f_rms"] for p in kept], dtype=float)
    df_all = np.asarray([p["d_f"] for p in pp], dtype=float)
    dM_all = np.asarray([p["d_M"] for p in pp], dtype=float)
    med = float(np.median(r)) if r.size else float("nan")
    return OrderedDict([
        ("n_stored", int(len(per_pair))),
        ("n_in_population", int(len(pp))),
        ("n_kept", int(r.size)),
        ("n_df_le0", int((df_all <= 0).sum())),
        ("n_df_lt1", int((df_all < 1).sum())),
        ("n_dM_le0", int((dM_all <= 0).sum())),
        ("median_ratio_raw", med),
        ("median_ratio_rms", float(np.median(r_rms)) if r_rms.size else float("nan")),
        ("A_from_median_ratio", (1.0 / med ** 2) if med and med > 0 else float("nan")),
        ("median_A_direct", float(np.median(1.0 / r ** 2)) if r.size else float("nan")),
        ("median_d_f_all", float(np.median(df_all)) if df_all.size else float("nan")),
        ("median_d_M_all", float(np.median(dM_all)) if dM_all.size else float("nan")),
        ("odd_count", bool(r.size % 2 == 1)),
    ])


def order_desc(values_by_attack):
    """Attacks sorted by descending value (ties broken by display order, reported separately)."""
    return sorted(ATTACKS, key=lambda a: (-values_by_attack[a], ATTACKS.index(a)))


def ranks_desc(values_by_attack):
    """Midranks with rank 1 = largest value (descending)."""
    v = np.asarray([values_by_attack[a] for a in ATTACKS], dtype=float)
    return rankdata(-v, method="average")


def kendall_w(rank_matrix):
    """Kendall's W with tie correction. rank_matrix: (m raters, n items) of midranks."""
    R = np.asarray(rank_matrix, dtype=float)
    m, n = R.shape
    colsum = R.sum(axis=0)
    S = float(((colsum - colsum.mean()) ** 2).sum())
    T = 0.0
    for j in range(m):
        _, counts = np.unique(R[j], return_counts=True)
        T += float(((counts ** 3) - counts).sum())
    denom = m * m * (n ** 3 - n) - m * T
    return 12.0 * S / denom, S, T


_M3_NULL = None


def kendall_w_null_m3():
    """Exact null distribution of W for m = 3 raters, n = 6 items (518400 points, cached)."""
    global _M3_NULL
    if _M3_NULL is None:
        base = np.arange(1.0, len(ATTACKS) + 1.0)
        perms = [np.asarray(p, dtype=float) for p in itertools.permutations(base)]
        _M3_NULL = np.asarray([kendall_w(np.vstack([base, p2, p3]))[0]
                               for p2 in perms for p3 in perms])
    return _M3_NULL


def kendall_w_p_exact_m3(W):
    null = kendall_w_null_m3()
    return float((null >= W - 1e-12).mean())


def kendall_w_p_mc(W, m, B, seed):
    """Monte-Carlo permutation p-value of W for m raters, n = 6 items."""
    n = len(ATTACKS)
    base = np.arange(1.0, n + 1.0)
    rng = np.random.default_rng(seed)
    hits, done, blk = 0, 0, 20000
    denom = m * m * (n ** 3 - n)
    while done < B:
        k = min(blk, B - done)
        R = np.empty((k, m, n))
        for j in range(m):
            R[:, j, :] = rng.permuted(np.tile(base, (k, 1)), axis=1)
        cs = R.sum(axis=1)
        S = ((cs - cs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        hits += int((12.0 * S / denom >= W - 1e-12).sum())
        done += k
    return float(hits) / float(B), int(hits), int(B)


def loo_spearman(rank_rows):
    """rank_rows: OrderedDict name -> midrank vector. Returns name -> Spearman vs mean consensus of the others."""
    names = list(rank_rows)
    out = OrderedDict()
    for a in names:
        others = [rank_rows[b] for b in names if b != a]
        cons = np.mean(np.vstack(others), axis=0)
        out[a] = float(spearmanr(cons, rank_rows[a]).statistic)
    return out


def family_check(values_by_attack):
    top_min = min(values_by_attack[a] for a in FAMILY_TOP)
    top_argmin = min(FAMILY_TOP, key=lambda a: values_by_attack[a])
    mid = values_by_attack[FAMILY_MID]
    bot_max = max(values_by_attack[a] for a in FAMILY_BOTTOM)
    bot_argmax = max(FAMILY_BOTTOM, key=lambda a: values_by_attack[a])
    return OrderedDict([
        ("holds", bool(top_min > mid > bot_max)),
        ("top_min", (top_argmin, float(top_min))),
        ("FGSM", float(mid)),
        ("bottom_max", (bot_argmax, float(bot_max))),
        ("ratio_topmin_over_FGSM", float(top_min / mid)),
        ("ratio_FGSM_over_bottommax", float(mid / bot_max)),
    ])


# ----------------------------------------------------------------------------- bootstrap

def _boot_medians(x, B, rng, chunk=2500):
    n = x.size
    out = np.empty(B)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        out[s:e] = np.median(x[rng.integers(0, n, (e - s, n))], axis=1)
    return out


def bca_median_ci(x, B, seed, alpha=0.05):
    """Seeded BCa 95 % bootstrap CI of the median (bias-corrected and accelerated)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    theta = float(np.median(x))
    boots = _boot_medians(x, B, np.random.default_rng(seed))
    # bias correction (mid-p for the atoms the median statistic produces)
    p0 = float((boots < theta).mean() + 0.5 * (boots == theta).mean())
    p0 = min(max(p0, 1.0 / B), 1.0 - 1.0 / B)
    z0 = float(norm.ppf(p0))
    xs = np.sort(x)
    jack = np.empty(n)
    for i in range(n):
        jack[i] = np.median(np.delete(xs, i))
    jm = jack.mean()
    s2 = float(((jm - jack) ** 2).sum())
    acc = float(((jm - jack) ** 3).sum() / (6.0 * s2 ** 1.5)) if s2 > 0 else 0.0
    lo_hi = []
    for a in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = norm.ppf(a)
        adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
        lo_hi.append(float(np.percentile(boots, 100.0 * norm.cdf(adj))))
    return OrderedDict([("median", theta), ("ci_lo", lo_hi[0]), ("ci_hi", lo_hi[1]),
                        ("method", "BCa"), ("B", int(B)), ("seed", int(seed)),
                        ("z0", z0), ("acceleration", acc), ("n", int(n))])


def diff_ci(col_a, col_b, B, seed, alpha=0.05, chunk=2500):
    """CIs for median(a) - median(b), two ways.

    col_a / col_b are aligned to the SAME population index (NaN where the cell dropped that
    pair).  `independent` resamples each cell's kept values on its own; `paired` resamples the
    population indices once per replicate and recomputes both medians on the draw (the matched
    design).  A gap is RESOLVED only if both CIs exclude zero -- the conservative verdict.
    """
    xa = col_a[~np.isnan(col_a)]
    xb = col_b[~np.isnan(col_b)]
    obs = float(np.median(xa) - np.median(xb))
    rng = np.random.default_rng(seed)
    di = _boot_medians(xa, B, rng, chunk) - _boot_medians(xb, B, rng, chunk)
    rng = np.random.default_rng(seed + 1)
    n = col_a.size
    dp = np.empty(B)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        S = rng.integers(0, n, (e - s, n))
        dp[s:e] = np.nanmedian(col_a[S], axis=1) - np.nanmedian(col_b[S], axis=1)
    lo_i, hi_i = (float(np.percentile(di, 100 * alpha / 2)),
                  float(np.percentile(di, 100 * (1 - alpha / 2))))
    lo_p, hi_p = (float(np.percentile(dp, 100 * alpha / 2)),
                  float(np.percentile(dp, 100 * (1 - alpha / 2))))
    resolved = bool((lo_i > 0 or hi_i < 0) and (lo_p > 0 or hi_p < 0))
    return OrderedDict([
        ("difference", obs), ("method", "percentile"), ("B", int(B)), ("seed", int(seed)),
        ("independent_ci", [lo_i, hi_i]), ("paired_ci", [lo_p, hi_p]),
        ("resolved_at_95", resolved),
    ])


# ----------------------------------------------------------------------------- loaders

def load_s3(path):
    if not os.path.exists(path):
        die("S3 reduce not found: %s" % path)
    d = json.load(open(path))
    cells = OrderedDict()
    for disp, key in TRIO.items():
        for a in ATTACKS:
            k = f"{key}|{S3_ATTACK_KEY[a]}"
            if k not in d:
                die(f"{k} missing from {path}")
            cells[(disp, a)] = d[k]["per_pair"]
    return cells


def common_population(s3, arch):
    sets = [set(p["pair_idx"] for p in s3[(arch, a)]) for a in ATTACKS]
    return set.intersection(*sets)


def load_t45(root):
    """theorem45 aggregates for all six raters.  Missing files are fatal (see module docstring)."""
    out = OrderedDict()
    missing = []
    for disp, exp in T45_EXP.items():
        p = os.path.join(root, exp, "theorem45", "theorem45_results.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        d = json.load(open(p))
        pa = d["per_attack"]
        out[disp] = OrderedDict([
            ("path", p),
            ("num_samples", d.get("num_samples")),
            ("median_ratio_raw", OrderedDict((a, float(pa[a]["amplification_M_median"])) for a in ATTACKS)),
            ("num_valid_pairs", OrderedDict((a, int(pa[a]["num_valid_pairs"])) for a in ATTACKS)),
            ("num_pairs", OrderedDict((a, int(pa[a]["num_pairs"])) for a in ATTACKS)),
            ("gamma_empirical", OrderedDict((a, float(pa[a]["gamma_empirical"])) for a in ATTACKS)),
            ("median_d_f", OrderedDict((a, float(pa[a]["d_f_stats"]["median"])) for a in ATTACKS)),
            ("median_d_M", OrderedDict((a, float(pa[a]["d_M_stats"]["median"])) for a in ATTACKS)),
        ])
    if missing:
        die("theorem45 reduce missing (%d file(s)); refusing to fall back to the committed "
            "printed rows, which would feed the table back in as its own rater input:\n  %s"
            % (len(missing), "\n  ".join(missing)))
    return out


def s3_attack_budgets(repo):
    """Phase-1 attack budgets, parsed from the generator source and diffed against the
    torchattacks defaults actually installed.  Returns {display attack: {param: (default, used)}}
    holding ONLY the parameters that depart from the default."""
    import ast
    import inspect

    import torchattacks

    # literal_eval cannot handle the generator's `8/255` etc., so evaluate the small arithmetic
    # grammar those literals actually use -- and nothing else (no names, no calls).
    _OPS = {ast.Div: lambda a, b: a / b, ast.Mult: lambda a, b: a * b,
            ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b}

    def _const(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_const(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_const(node.left), _const(node.right))
        if isinstance(node, ast.Tuple):
            return tuple(_const(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [_const(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {_const(k): _const(v) for k, v in zip(node.keys, node.values)}
        raise ValueError("unsupported node in cls_map: %r" % (node,))

    src = os.path.join(repo, S3_GENERATOR)
    if not os.path.exists(src):
        die("Phase-1 generator not found, cannot derive its attack budgets: %s" % src)
    tree = ast.parse(open(src).read())
    cls_map = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "cls_map" for t in node.targets):
            try:
                cls_map = _const(node.value)
            except ValueError as e:
                die("could not read `cls_map` from %s: %s" % (src, e))
            break
    if cls_map is None:
        die("could not find `cls_map` in %s; the budget clause must not be retyped by hand" % src)
    key_to_disp = {v: k for k, v in S3_ATTACK_KEY.items()}
    out = OrderedDict()
    for a in ATTACKS:
        key = S3_ATTACK_KEY[a]
        if key not in cls_map:
            die("%s: cls_map has no entry for %r" % (src, key))
        cls_name, kwargs = cls_map[key]
        sig = inspect.signature(getattr(torchattacks, cls_name).__init__).parameters
        diffs = OrderedDict()
        for k, v in kwargs.items():
            if k not in sig:
                die("%s: torchattacks %s has no parameter %r" % (src, cls_name, k))
            d = sig[k].default
            same = (abs(d - v) < 1e-12 if isinstance(v, (int, float))
                    and isinstance(d, (int, float)) and not isinstance(v, bool) else d == v)
            if not same:
                diffs[k] = (d, v)
        out[a] = diffs
    assert set(key_to_disp.values()) == set(ATTACKS)
    return out, torchattacks.__version__


def budgets_tex(budgets, ta_version):
    """LaTeX clause naming EVERY Phase-1 override, and the attacks left at the defaults."""
    parts = []
    for a, diffs in budgets.items():
        if not diffs:
            continue
        bits = []
        for k, (_d, v) in diffs.items():
            if k == "loss":
                bits.append(r"the %s loss" % str(v).upper())
            else:
                bits.append("%s$=%s$" % (TA_PARAM_TEX.get(k, r"\texttt{%s}" % k),
                                         f"{v:,}".replace(",", "{,}") if isinstance(v, int) else v))
        parts.append("%s %s" % (a, ", ".join(bits)))
    defaults = [a for a, d in budgets.items() if not d]
    tail = (r"; %s at the library defaults" % ", ".join(defaults)) if defaults else ""
    return (r"Phase-1 attack budgets, every departure from the \texttt{torchattacks}~"
            + ta_version + r" defaults: " + "; ".join(parts) + tail)


def vgg_filter_sensitivity(root):
    """The only theorem45 cells with per-pair records: three VGG diagnostic dumps.

    Reports, per cell, the median under the theorem45 guard (d_f > 1e-6 max d_f) that produced
    the committed tables and under the stricter attack-success filter (d_f >= 1, d_M > 0).
    Disclosure only -- no table is recomputed with the stricter filter.
    """
    out = OrderedDict()
    for atk in VGG_DEBUG_ATTACKS:
        p = os.path.join(root, "vgg_imagenet", "theorem45", f"debug_gamma_zero_{atk}.json")
        if not os.path.exists(p):
            die("VGG per-pair diagnostic missing: %s" % p)
        d = json.load(open(p))
        df = np.asarray([s["d_f"] for s in d["per_sample"]], dtype=float)
        dM = np.asarray([s["d_M"] for s in d["per_sample"]], dtype=float)
        thr = max(1e-12, 1e-6 * float(df.max()))
        g = df > thr
        s = (df >= 1) & (dM > 0)
        m_g = float(np.median(dM[g] / df[g]))
        m_s = float(np.median(dM[s] / df[s]))
        out[atk] = OrderedDict([
            ("path", p), ("n_samples", int(df.size)),
            ("guard_threshold", float(thr)),
            ("n_guard", int(g.sum())), ("median_guard", m_g),
            ("n_strict", int(s.sum())), ("median_strict", m_s),
            ("relative_shift", float((m_s - m_g) / m_g)),
            ("n_0_lt_df_lt_1", int(((df > thr) & (df < 1)).sum())),
            ("n_dM_le_0", int((dM <= 0).sum())),
            ("n_dropped_by_guard", int((~g).sum())),
        ])
    return out


# ----------------------------------------------------------------------------- LaTeX

def _order_tex(order):
    return " $>$ ".join(order)


def _resolution_tex(order, diffs):
    """`DeepFool $\\approx$ Square $>$ FGSM ...` from the adjacent-gap resolution verdicts."""
    out = [order[0]]
    for k in range(len(order) - 1):
        out.append(r" $>$ " if diffs[k]["resolved_at_95"] else r" $\approx$ ")
        out.append(order[k + 1])
    return "".join(out)


GG_THRESHOLD = 1.5


def _gg_or_gt(min_ratio, threshold=GG_THRESHOLD):
    r"""`\gg` is only written where the smallest measured gap actually justifies it."""
    return r"\gg" if min_ratio >= threshold else r">"


def _fmt_p(p, B=None):
    if p <= 0.0:
        return r"$p<%s$" % ("10^{-6}" if B is None else ("%g" % (1.0 / B)))
    return r"$p=%.1f\times10^{%d}$" % (p / 10 ** int(np.floor(np.log10(p))),
                                       int(np.floor(np.log10(p))))


def population_tex(ctx):
    r"""The population sentence, generated from the computed truncation and shift figures.

    Nothing here is a literal claim about the data: which architecture was truncated, by how
    much, which cells moved and the bound on the untruncated cells all come from `ctx`, and the
    `--population all` branch says the opposite of the `common` branch.  A doctored input or a
    mode switch therefore cannot leave a matching claim standing over unmatched numbers.
    """
    t = ctx["truncation"]
    arch, atk = t["arch"], t["attack"]
    n_lo, n_hi = t["n_truncated"], t["n_full"]
    moved = ", ".join(
        "%s by $%+.1f$\\%%" % (a, 100 * ctx["match_shift"][a]) for a in t["moved_cells"])
    bound = t["untruncated_max_abs_shift_pct_ceiling"]
    others = [a for a in ctx["n_common"] if a != arch]
    if ctx["population"] == "common":
        return (
            r"Cells are matched on a common image population --- the intersection of the six "
            r"cells' stored pair indices, $n=" + str(ctx["n_common"][arch]) + r"$ on " + arch +
            r" and $n=" + str(ctx["n_common"][others[0]]) + r"$ on the other two --- because the "
            + arch + " " + atk + r" run stopped at $" + str(n_lo) + r"$ of $" + str(n_hi) +
            r"$ pairs, so an unmatched panel would compare medians across different images "
            r"(matching moves " + arch + " " + moved + r", while the cells the filter leaves "
            r"intact --- " + " and ".join(t["filter_untouched_cells"]) + r" --- move by under $"
            + str(bound) + r"$\%).")
    return (
        r"\textbf{Cells are NOT matched on a common image population}: every stored pair of each "
        r"cell is used, although the " + arch + " " + atk + r" run stopped at $" + str(n_lo) +
        r"$ of $" + str(n_hi) + r"$ pairs, so the " + arch + r" medians below are compared across "
        r"different images. Matching would move " + arch + " " + moved + r" while leaving "
        + " and ".join(t["filter_untouched_cells"]) + r" under $" + str(bound) +
        r"$\%; this panel is a comparison run, not the one the paper prints.")


def tex_s3_panel(ctx):
    conv, pop = ctx["convention"], ctx["population"]
    rows, W, p_exact = ctx["rows"], ctx["W"], ctx["W_p_exact"]
    n_common = ctx["n_common"]
    lines = []
    lines.append("% Phase-1 S3 attack-family ordering panel (APPENDIX cross-check). Source:")
    lines.append("% cluster-data/results/phase1/aggregated/s3_results.json (full-scale Phase-1 reduce;")
    lines.append("% 1000 stored pairs per cell except ResNet-152/DeepFool, 512; da36545/400d083 on Rorqual, 2026-06-19).")
    lines.append(f"% REGENERATED {ctx['date']} by {ctx['script']} "
                 f"--convention {conv} --population {pop} --bootstrap-B {ctx['B']} --seed {ctx['seed']}.")
    lines.append(f"% Statistic: per (arch, attack) the per-pair MEDIAN d_M/d_f over {CONVENTIONS[conv]}, descending")
    lines.append("% (= most knowledge-matrix amplification = lowest coherence A).")
    lines.append(f"% POPULATION: {POPULATIONS[pop]}. n_common = "
                 + ", ".join(f"{a} {n_common[a]}" for a in TRIO) + ".")
    t = ctx["truncation"]
    lines.append(f"% The {t['arch']}/{t['attack']} cell stopped at {t['n_truncated']} of {t['n_full']} stored pairs,")
    lines.append("% so an unmatched panel would compare medians across different image populations. Matching moves "
                 + ", ".join(f"{t['arch']}/{a} {100*ctx['match_shift'][a]:+.1f} %" for a in ATTACKS) + ".")
    lines.append("% The ranking is identical under all three filtering conventions and under raw<->RMS rescaling")
    lines.append("% (the RMS factors are per-architecture constants), so this panel does NOT introduce a second")
    lines.append("% coherence-A magnitude -- the A magnitudes are the Study-2 headline (Table~\\ref{tab:coherence});")
    lines.append("% this trio only cross-checks the ORDERING.")
    lines.append("% Kendall's W: midranks, tie-corrected; p = exact permutation p-value over the (6!)^2 null.")
    lines.append("% CIs: seeded BCa 95% bootstrap over pairs per cell; adjacent-rank differences resolved only")
    lines.append("% when the independent AND the paired percentile CI both exclude zero.")
    lines.append("% N/cell = pairs kept by the filter within the population (min--max over the six attacks).")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")

    unresolved = [a for a in TRIO if not all(d["resolved_at_95"] for d in ctx["diffs"][a])]
    resolved_all = [a for a in TRIO if a not in unresolved]
    bstr = f"{ctx['B']:,}".replace(",", "{,}")
    if unresolved:
        clause = (r" \emph{Bootstrap resolution} (BCa and percentile $95\%$ CIs, $B=" + bstr
                  + r"$, seeded, over pairs): ")
        clause += "; ".join(f"{a} reads {_resolution_tex(ctx['orders'][a], ctx['diffs'][a])}"
                            for a in unresolved)
        if resolved_all:
            clause += (r", while on " + " and ".join(resolved_all)
                       + r" every adjacent gap is resolved")
        clause += r"."
    else:
        clause = (r" \emph{Bootstrap resolution}: every adjacent gap is resolved at $95\%$ "
                  r"(BCa and percentile CIs, $B=" + bstr + r"$, seeded, over pairs).")

    cap = (
        r"\caption{Attack-family ordering cross-check on the Phase-1 trio (appendix). Each "
        r"architecture's six attack families ranked by median $d_M/d_\Psi$ over "
        + CONVENTION_TEX[conv] +
        r", descending (equivalently ascending coherence $A=(d_\Psi/d_M)^2$), computed on the "
        r"full-scale Phase-1 pair set (" + ctx["budgets_tex"] + r"; "
        r"Table~\ref{tab:ordering} uses the $200$-pair \textsc{theorem45} pair set of "
        r"Section~\ref{sec:setup}, which is drawn from different images). "
        + ctx["population_tex"] +
        r" The ranking is unchanged under all three filtering conventions. "
        r"Kendall $W=" + f"{W:.2f}" + r"$ across the three architectures (" + _fmt_p(p_exact) +
        r", exact permutation test; the $m=3$ null is supported on $77$ lattice points spaced "
        r"$0.0127$ apart, so $W$ is quoted to two decimals) cross-checks the six-architecture "
        r"headline concordance ($W=0.921$, Table~\ref{tab:ordering}), and the point-estimate "
        r"ordering reproduces its family-level grouping $\{$DeepFool, CW, Square$\}"
        + _gg_or_gt(ctx["fam_top_min"]) + r"$ FGSM $" + _gg_or_gt(ctx["fam_bot_min"])
        + r"\{$PGD, APGD$\}$ on all three architectures (gaps "
        + f"${ctx['fam_top_min']:.2f}$--${ctx['fam_top_max']:.2f}" + r"\times$ and "
        + f"${ctx['fam_bot_min']:.2f}$--${ctx['fam_bot_max']:.2f}" + r"\times$ respectively)."
        + clause +
        r" Computed on the metric-invariant ranks, so no raw/RMS unit choice enters.}"
    )
    lines.append(cap)
    lines.append(r"\label{tab:s3_ordering_panel}")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Ordering by median $d_M/d_\Psi$ (desc.) & $N$/cell \\")
    lines.append(r"\midrule")
    for arch, (order, nmin, nmax) in rows.items():
        lines.append(f"{arch:<12s} & {_order_tex(order)} & {nmin}--{nmax} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{3}}{{l}}{{Kendall's $W = {W:.2f}$ "
                 f"(3 architectures $\\times$ 6 attacks)}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def tex_table_ordering(rows, W, loo, variant, ctx):
    lines = []
    lines.append("% Attack-family ordering of d_M/d_f (equivalently reverse ordering of coherence A).")
    if variant == "A":
        lines.append(f"% Variant A ({ctx['date']}, {ctx['script']}): all six rows from the theorem45 reduce")
        lines.append("% (experiments/*/theorem45/theorem45_results.json, N=200 pairs/cell, guard d_f > 1e-6*max d_f) --")
        lines.append("% the same reduce as tables/table_coherence.tex; identical rows to the committed table.")
    else:
        lines.append(f"% Variant B ({ctx['date']}, {ctx['script']}): TRIO rows from the Phase-1 S3 reduce")
        lines.append(f"% (s3_results.json, --convention {ctx['convention']} --population {ctx['population']});")
        lines.append("% ResNet-18 / AlexNet / VGG rows from the theorem45 reduce (no other source on disk).")
        lines.append("% REJECTED by Ruling C-R8: this desynchronises the trio rows from tables/table_coherence.tex")
        lines.append("% (theorem45 reduce). Emitted only so the rejected option stays inspectable. NOT the paper's table.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{llc}")
    lines.append(r"\toprule")
    lines.append(r"architecture & ordering by median $d_M/d_\Psi$ (desc.) & LOO Spearman \\")
    lines.append(r"\midrule")
    for arch, order in rows.items():
        lines.append(f"{arch} & {_order_tex(order)} & {loo[arch]:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    cap = (r"\caption{Attack-family ordering concordance across six architectures: Kendall "
           f"$W={W:.3f}$"
           r"" "\n"
           r"(6 raters $\times$ 6 attacks); leave-one-architecture-out Spearman of each architecture's ranking against"
           "\n"
           r"the mean consensus rank of the other five (midrank ties). The six raters are the paper's trio"
           "\n"
           r"(ResNet-152, DenseNet-121, GoogLeNet) plus three \emph{ordering-only} cross-check raters (ResNet-18,"
           "\n"
           r"AlexNet, VGG) that enter no coherence-magnitude statistic and no cross-architecture distance.")
    if variant == "A":
        cap += ("\n" r"The stable structure is family-level: $\{$DeepFool, CW, Square$\}"
                + _gg_or_gt(ctx["t45_fam_top_min"]) + r"$ FGSM $"
                + _gg_or_gt(ctx["t45_fam_bot_min"]) + r"\{$PGD, APGD$\}$ in "
                r"$d_M/d_\Psi$ --- i.e.\ the iterative-PGD family produces the most \emph{coherent} germ motion.}")
    else:
        cap += ("\n" r"Trio rows recomputed on the population-matched Phase-1 pair set; ordering-only raters "
                r"unchanged.}")
    lines.append(cap)
    lines.append(r"\label{tab:ordering}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main

def to_native(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, str) else k: to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def resolve_date(arg):
    """Byte-reproducible date stamp: --date, else SOURCE_DATE_EPOCH, else fatal."""
    if arg:
        try:
            return _dt.date.fromisoformat(arg).isoformat()
        except ValueError:
            die("--date must be ISO YYYY-MM-DD, got %r" % arg)
    sde = os.environ.get("SOURCE_DATE_EPOCH")
    if sde:
        try:
            return _dt.datetime.fromtimestamp(int(sde), _dt.timezone.utc).date().isoformat()
        except (ValueError, OverflowError):
            die("SOURCE_DATE_EPOCH is not an integer unix timestamp: %r" % sde)
    die("no date stamp: pass --date YYYY-MM-DD (or set SOURCE_DATE_EPOCH). The emitted tables "
        "carry the date in a comment, so taking it from the clock would make the output "
        "non-reproducible.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3", default=S3_DEFAULT)
    ap.add_argument("--t45-root", default=T45_ROOT_DEFAULT)
    ap.add_argument("--convention", default="df1_dM0", choices=list(CONVENTIONS))
    ap.add_argument("--population", default="common", choices=list(POPULATIONS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--date", default=None,
                    help="ISO date stamped into the emitted files (or set SOURCE_DATE_EPOCH)")
    ap.add_argument("--bootstrap-B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--mc-permutations", type=int, default=1000000,
                    help="Monte-Carlo draws for the six-rater permutation p-value")
    ap.add_argument("--min-gap-rel", type=float, default=0.01,
                    help="refuse to print an ordering whose adjacent medians differ by less than this")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    date = resolve_date(args.date)
    script_name = "scripts/regen_ordering_tables.py"
    ck = Checks()
    conv, pop = args.convention, args.population
    B, seed = args.bootstrap_B, args.seed

    # ---- S3 reduce -----------------------------------------------------------------
    s3 = load_s3(args.s3)
    common = OrderedDict((arch, common_population(s3, arch)) for arch in TRIO)
    n_common = OrderedDict((arch, len(common[arch])) for arch in TRIO)
    for arch in TRIO:
        stored = {a: len(s3[(arch, a)]) for a in ATTACKS}
        ck.note(f"population|{arch}", OrderedDict([("stored_per_cell", stored),
                                                   ("n_common", n_common[arch])]))
        ck.require(f"population_nonempty|{arch}", n_common[arch] > 0,
                   "empty common population")

    keep = {arch: (None if pop == "all" else common[arch]) for arch in TRIO}
    s3_cells = OrderedDict()
    for c in CONVENTIONS:
        for p in POPULATIONS:
            k = {arch: (None if p == "all" else common[arch]) for arch in TRIO}
            for (arch, atk), pp in s3.items():
                s3_cells[(p, c, arch, atk)] = cell_stats(pp, c, k[arch])

    s3_orderings, s3_rank_rows, s3_family, s3_W = (OrderedDict() for _ in range(4))
    for p in POPULATIONS:
        for c in CONVENTIONS:
            s3_orderings[(p, c)] = OrderedDict()
            s3_rank_rows[(p, c)] = OrderedDict()
            s3_family[(p, c)] = OrderedDict()
            for arch in TRIO:
                vals = {a: s3_cells[(p, c, arch, a)]["median_ratio_raw"] for a in ATTACKS}
                s3_orderings[(p, c)][arch] = order_desc(vals)
                s3_rank_rows[(p, c)][arch] = ranks_desc(vals)
                s3_family[(p, c)][arch] = family_check(vals)
            W, S, T = kendall_w(np.vstack(list(s3_rank_rows[(p, c)].values())))
            s3_W[(p, c)] = OrderedDict([("W", W), ("S", S), ("tie_T", T),
                                        ("p_exact_permutation", kendall_w_p_exact_m3(W))])

    ck.require("s3_rank_invariant_across_conventions",
               all(s3_orderings[(pop, c)] == s3_orderings[(pop, conv)] for c in CONVENTIONS),
               {c: s3_orderings[(pop, c)] for c in CONVENTIONS})
    ck.require("s3_rank_invariant_raw_vs_rms",
               all(order_desc({a: s3_cells[(pop, conv, arch, a)]["median_ratio_rms"] for a in ATTACKS})
                   == s3_orderings[(pop, conv)][arch] for arch in TRIO), None)

    # ---- the truncation, found in the data (not assumed) ---------------------------
    # Which architecture holds a short cell, which cell it is, and how far matching moves each
    # of that architecture's cells.  Every figure the caption quotes is computed here and
    # ENFORCED below, so the caption cannot outlive the data that justified it.
    counts = {(arch, a): len(s3[(arch, a)]) for arch in TRIO for a in ATTACKS}
    n_full = max(counts.values())
    short = sorted((n, arch, a) for (arch, a), n in counts.items() if n < n_full)
    ck.require("exactly_one_truncated_cell", len(short) == 1,
               {"short_cells": short, "n_full": n_full})
    if len(short) != 1:
        ck.enforce()
    n_lo, t_arch, t_atk = short[0]

    match_shift = OrderedDict()
    for a in ATTACKS:
        m_all = s3_cells[("all", conv, t_arch, a)]["median_ratio_raw"]
        m_com = s3_cells[("common", conv, t_arch, a)]["median_ratio_raw"]
        match_shift[a] = float((m_com - m_all) / m_all)
    # The reference cells are those the FILTER never touches (n_kept == n_stored over the full
    # population): for them the restriction is a plain subsample of one distribution, so any
    # movement is pure sampling noise and bounds the artifact.  Cells that do lose pairs to the
    # filter lose them unevenly over the index range, and those are the ones that move.
    untrunc = [a for a in ATTACKS
               if counts[(t_arch, a)] == n_full
               and s3_cells[("all", conv, t_arch, a)]["n_kept"] == counts[(t_arch, a)]]
    ck.require("some_cell_is_untouched_by_the_filter", len(untrunc) >= 1,
               {"per_cell": {a: (counts[(t_arch, a)],
                                 s3_cells[("all", conv, t_arch, a)]["n_kept"]) for a in ATTACKS}})
    if not untrunc:
        ck.enforce()
    untrunc_max = max(abs(match_shift[a]) for a in untrunc)
    moved_cells = sorted((a for a in ATTACKS
                          if a not in untrunc and a != t_atk and abs(match_shift[a]) > untrunc_max),
                         key=lambda a: -abs(match_shift[a]))[:2]
    bound_pct = int(np.ceil(100 * untrunc_max)) or 1
    truncation = OrderedDict([
        ("arch", t_arch), ("attack", t_atk), ("n_truncated", int(n_lo)), ("n_full", int(n_full)),
        ("filter_untouched_cells", untrunc), ("untruncated_cells", untrunc),
        ("moved_cells", moved_cells),
        ("untruncated_max_abs_shift", untrunc_max),
        ("untruncated_max_abs_shift_pct_ceiling", bound_pct),
        ("match_shift", match_shift),
    ])
    # ENFORCED, not noted: the artifact signature the caption asserts.
    ck.require("truncated_cell_is_the_unmatched_one", counts[(t_arch, t_atk)] == n_lo, truncation)
    ck.require("population_shift_is_asymmetric",
               bool(moved_cells) and min(abs(match_shift[a]) for a in moved_cells) > untrunc_max,
               truncation)
    ck.require("untruncated_cells_move_less_than_the_quoted_bound",
               100 * untrunc_max <= bound_pct, truncation)
    ck.require("quoted_bound_is_tight", bound_pct - 100 * untrunc_max < 1.0, truncation)

    # ---- bootstrap on the shipped (population, convention) -------------------------
    cols, cell_ci, diffs = OrderedDict(), OrderedDict(), OrderedDict()
    for arch in TRIO:
        idx = sorted(common[arch]) if pop == "common" else sorted(
            set().union(*[set(p["pair_idx"] for p in s3[(arch, a)]) for a in ATTACKS]))
        posn = {v: i for i, v in enumerate(idx)}
        for a in ATTACKS:
            col = np.full(len(idx), np.nan)
            for p in s3[(arch, a)]:
                i = posn.get(p["pair_idx"])
                if i is not None and keep_pair(p, conv):
                    col[i] = p["d_M"] / p["d_f"]
            cols[(arch, a)] = col
            x = col[~np.isnan(col)]
            ck.require(f"cell_nonempty|{arch}|{a}", x.size > 0, "no kept pairs")
            cell_ci[(arch, a)] = bca_median_ci(x, B, seed + 7 * ATTACKS.index(a)
                                               + 101 * list(TRIO).index(arch))
            ck.require(f"cell_median_matches_summary|{arch}|{a}",
                       abs(cell_ci[(arch, a)]["median"]
                           - s3_cells[(pop, conv, arch, a)]["median_ratio_raw"]) < 1e-12, None)
        order = s3_orderings[(pop, conv)][arch]
        med = {a: s3_cells[(pop, conv, arch, a)]["median_ratio_raw"] for a in ATTACKS}
        diffs[arch] = []
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            rel = (med[a] - med[b]) / med[b]
            ck.require(f"adjacent_gap_above_min|{arch}|{a}-{b}", rel >= args.min_gap_rel,
                       {"relative_gap": rel, "min_gap_rel": args.min_gap_rel,
                        "note": "a gap this small is a tie the rank order cannot represent"})
            dd = diff_ci(cols[(arch, a)], cols[(arch, b)], B,
                         seed + 1000 + 10 * list(TRIO).index(arch) + k)
            dd["pair"] = [a, b]
            dd["relative_gap"] = float(rel)
            diffs[arch].append(dd)

    # ---- theorem45 reduce: six raters ----------------------------------------------
    t45 = load_t45(args.t45_root)
    t45_orderings, t45_rank_rows, t45_family = OrderedDict(), OrderedDict(), OrderedDict()
    for arch in T45_EXP:
        vals = t45[arch]["median_ratio_raw"]
        t45_orderings[arch] = order_desc(vals)
        t45_rank_rows[arch] = ranks_desc(vals)
        t45_family[arch] = family_check(vals)
    W_A, S_A, T_A = kendall_w(np.vstack(list(t45_rank_rows.values())))
    loo_A = loo_spearman(t45_rank_rows)
    W_trio_t45, _, _ = kendall_w(np.vstack([t45_rank_rows[a] for a in TRIO]))
    p_A, hits_A, B_A = kendall_w_p_mc(W_A, 6, args.mc_permutations, seed)
    p_trio_t45 = kendall_w_p_exact_m3(W_trio_t45)

    ck.require("six_rater_W_matches_stored",
               abs(W_A - STORED_SIX_RATER["kendall_W"]) < 1e-9, (W_A, STORED_SIX_RATER["kendall_W"]))
    ck.require("six_rater_loo_matches_stored",
               all(abs(loo_A[a] - STORED_SIX_RATER["loo"][a]) < 1e-9 for a in loo_A), loo_A)
    ck.require("six_rater_rows_match_committed_tab_ordering",
               all(t45_orderings[a] == COMMITTED_TAB_ORDERING[a] for a in COMMITTED_TAB_ORDERING),
               t45_orderings)
    ck.require("theorem45_family_grouping_holds_on_all_six",
               all(t45_family[a]["holds"] for a in T45_EXP),
               {a: t45_family[a]["holds"] for a in T45_EXP})
    ck.require("s3_family_grouping_holds_on_all_three",
               all(s3_family[(pop, conv)][a]["holds"] for a in TRIO),
               {a: s3_family[(pop, conv)][a] for a in TRIO})

    fam_top = [t45_family[a]["ratio_topmin_over_FGSM"] for a in T45_EXP]
    fam_bot = [t45_family[a]["ratio_FGSM_over_bottommax"] for a in T45_EXP]
    ck.note("theorem45_family_ratio_ranges",
            OrderedDict([("topmin_over_FGSM", [min(fam_top), max(fam_top)]),
                         ("FGSM_over_bottommax", [min(fam_bot), max(fam_bot)])]))
    # The >> / > wording rule, at the SAME threshold _gg_or_gt applies (an earlier version
    # tested 2.0, which is looser than the rule it is named for and would have passed a gap
    # the caption then printed as ">").
    ck.require("top_family_over_FGSM_is_not_gg", min(fam_top) < GG_THRESHOLD,
               "the top-family/FGSM gap is %.3f-%.3fx against the \\gg threshold %.2f; "
               "the paper must write '>' not '\\gg'" % (min(fam_top), max(fam_top), GG_THRESHOLD))

    # ---- VGG filter sensitivity (disclosure only) ----------------------------------
    vgg = vgg_filter_sensitivity(args.t45_root)
    vgg_new = dict(t45["VGG"]["median_ratio_raw"])
    for a, v in vgg.items():
        ck.require(f"vgg_guard_median_reproduces_stored|{a}",
                   abs(v["median_guard"] - t45["VGG"]["median_ratio_raw"][a])
                   / t45["VGG"]["median_ratio_raw"][a] < 0.01,
                   {"debug_dump": v["median_guard"], "stored": t45["VGG"]["median_ratio_raw"][a]})
        vgg_new[a] = v["median_strict"]
    vgg_order_guard = order_desc(t45["VGG"]["median_ratio_raw"])
    vgg_order_strict = order_desc(vgg_new)
    vgg_family_strict = family_check(vgg_new)
    ck.note("vgg_filter_sensitivity", OrderedDict([
        ("cells_with_per_pair_data", VGG_DEBUG_ATTACKS),
        ("relative_shift_range", [min(v["relative_shift"] for v in vgg.values()),
                                  max(v["relative_shift"] for v in vgg.values())]),
        ("order_guard", vgg_order_guard), ("order_strict", vgg_order_strict),
        ("top_two_reorders", vgg_order_guard[:2] != vgg_order_strict[:2]),
        ("family_grouping_unchanged", vgg_family_strict["holds"]),
        ("A_apgd_guard", 1.0 / t45["VGG"]["median_ratio_raw"]["APGD"] ** 2),
        ("A_apgd_strict", 1.0 / vgg_new["APGD"] ** 2),
    ]))
    ck.require("vgg_family_grouping_survives_strict_filter", vgg_family_strict["holds"],
               vgg_family_strict)

    # ---- Variant B panel (rejected option, kept inspectable) -----------------------
    rank_rows_B, orderings_B = OrderedDict(), OrderedDict()
    for arch in T45_EXP:
        if arch in TRIO:
            rank_rows_B[arch] = s3_rank_rows[(pop, conv)][arch]
            orderings_B[arch] = s3_orderings[(pop, conv)][arch]
        else:
            rank_rows_B[arch] = t45_rank_rows[arch]
            orderings_B[arch] = t45_orderings[arch]
    W_B, S_B, T_B = kendall_w(np.vstack(list(rank_rows_B.values())))
    loo_B = loo_spearman(rank_rows_B)

    # ---- rows + LaTeX ---------------------------------------------------------------
    s3_rows = OrderedDict()
    for arch in TRIO:
        ns = [s3_cells[(pop, conv, arch, a)]["n_kept"] for a in ATTACKS]
        s3_rows[arch] = (s3_orderings[(pop, conv)][arch], min(ns), max(ns))

    ctx = OrderedDict([
        ("date", date), ("script", script_name), ("convention", conv), ("population", pop),
        ("B", B), ("seed", seed), ("rows", s3_rows), ("orders", s3_orderings[(pop, conv)]),
        ("diffs", diffs), ("W", s3_W[(pop, conv)]["W"]),
        ("W_p_exact", s3_W[(pop, conv)]["p_exact_permutation"]),
        ("n_common", n_common), ("match_shift", match_shift), ("truncation", truncation),
    ])
    budgets, ta_version = s3_attack_budgets(REPO)
    ctx["budgets"] = budgets
    ctx["budgets_tex"] = budgets_tex(budgets, ta_version)
    ctx["population_tex"] = population_tex(ctx)
    ck.require("every_s3_override_is_named_in_the_caption",
               all((a in ctx["budgets_tex"]) for a, d in budgets.items() if d),
               {"overrides": {a: dict(d) for a, d in budgets.items() if d},
                "clause": ctx["budgets_tex"]})
    # "Cells are NOT matched ..." must not satisfy the positive test, hence the full phrase.
    ck.require("caption_population_claim_matches_mode",
               ("Cells are matched on a common image population" in ctx["population_tex"])
               == (pop == "common")
               and ("NOT matched" in ctx["population_tex"]) == (pop != "common"),
               {"population": pop, "sentence": ctx["population_tex"]})
    s3_fam_top = [s3_family[(pop, conv)][a]["ratio_topmin_over_FGSM"] for a in TRIO]
    s3_fam_bot = [s3_family[(pop, conv)][a]["ratio_FGSM_over_bottommax"] for a in TRIO]
    ctx["fam_top_min"], ctx["fam_top_max"] = min(s3_fam_top), max(s3_fam_top)
    ctx["fam_bot_min"], ctx["fam_bot_max"] = min(s3_fam_bot), max(s3_fam_bot)
    ctx["t45_fam_top_min"], ctx["t45_fam_bot_min"] = min(fam_top), min(fam_bot)
    ck.note("s3_family_ratio_ranges",
            OrderedDict([("topmin_over_FGSM", [ctx["fam_top_min"], ctx["fam_top_max"]]),
                         ("FGSM_over_bottommax", [ctx["fam_bot_min"], ctx["fam_bot_max"]])]))
    ck.enforce()  # nothing is written unless every recorded check passed

    with open(os.path.join(args.out_dir, "s3_table.tex"), "w") as fh:
        fh.write(tex_s3_panel(ctx))
    with open(os.path.join(args.out_dir, "table_ordering_variantA.tex"), "w") as fh:
        fh.write(tex_table_ordering(t45_orderings, W_A, loo_A, "A", ctx))
    with open(os.path.join(args.out_dir, "table_ordering_variantB.tex"), "w") as fh:
        fh.write(tex_table_ordering(orderings_B, W_B, loo_B, "B", ctx))

    summary = OrderedDict([
        ("generated", date), ("script", script_name),
        ("convention_for_tables", conv), ("population_for_tables", pop),
        ("bootstrap", OrderedDict([("B", B), ("seed", seed), ("cell_ci_method", "BCa"),
                                   ("difference_ci_method", "percentile"),
                                   ("resolution_rule",
                                    "resolved iff the independent AND the paired 95% CI exclude 0")])),
        ("conventions", CONVENTIONS), ("populations", POPULATIONS),
        ("sources", OrderedDict([("s3", args.s3),
                                 ("theorem45", OrderedDict((a, t45[a]["path"]) for a in T45_EXP)),
                                 ("vgg_per_pair", OrderedDict((a, vgg[a]["path"]) for a in vgg))])),
        ("n_common", n_common),
        ("truncation", truncation),
        ("s3_attack_budgets_vs_torchattacks_defaults",
         OrderedDict((a, OrderedDict((k, {"default": d, "used": v}) for k, (d, v) in dd.items()))
                     for a, dd in budgets.items())),
        ("generated_caption_clauses",
         OrderedDict([("budgets", ctx["budgets_tex"]), ("population", ctx["population_tex"])])),
        ("population_matching_shift", match_shift),
        ("s3_cells", OrderedDict((f"{p}|{c}|{a}|{k}", v) for (p, c, a, k), v in s3_cells.items())),
        ("s3_orderings", OrderedDict((f"{p}|{c}", v) for (p, c), v in s3_orderings.items())),
        ("s3_trio_kendall_W", OrderedDict((f"{p}|{c}", v) for (p, c), v in s3_W.items())),
        ("s3_family_grouping", OrderedDict((f"{p}|{c}", v) for (p, c), v in s3_family.items())),
        ("s3_panel_rows", OrderedDict((a, {"order": o, "n_min": lo, "n_max": hi})
                                      for a, (o, lo, hi) in s3_rows.items())),
        ("s3_cell_bootstrap_ci", OrderedDict((f"{a}|{k}", v) for (a, k), v in cell_ci.items())),
        ("s3_adjacent_rank_differences", OrderedDict((a, diffs[a]) for a in TRIO)),
        ("theorem45", OrderedDict((a, t45[a]) for a in T45_EXP)),
        ("theorem45_orderings", t45_orderings),
        ("theorem45_family_grouping", t45_family),
        ("theorem45_family_ratio_ranges",
         OrderedDict([("topmin_over_FGSM", [min(fam_top), max(fam_top)]),
                      ("FGSM_over_bottommax", [min(fam_bot), max(fam_bot)])])),
        ("vgg_filter_sensitivity", vgg),
        ("vgg_orderings", OrderedDict([("guard", vgg_order_guard), ("strict", vgg_order_strict),
                                       ("family_grouping_strict", vgg_family_strict)])),
        ("six_rater_variantA_all_theorem45", OrderedDict([
            ("kendall_W", W_A), ("S", S_A), ("loo_spearman", loo_A), ("trio_only_W", W_trio_t45),
            ("trio_only_W_p_exact", p_trio_t45),
            ("kendall_W_p_montecarlo", OrderedDict([("p", p_A), ("hits", hits_A), ("draws", B_A)]))])),
        ("six_rater_variantB_s3_trio_plus_theorem45_others", OrderedDict([
            ("kendall_W", W_B), ("S", S_B), ("loo_spearman", loo_B), ("orderings", orderings_B),
            ("trio_only_W", s3_W[(pop, conv)]["W"]), ("status", "REJECTED by Ruling C-R8")])),
        ("delta_W_B_minus_A", W_B - W_A),
        ("kendall_w_m3_null", OrderedDict([
            ("support_points", int(np.unique(np.round(kendall_w_null_m3(), 9)).size)),
            ("lattice_step", float(np.diff(np.unique(np.round(kendall_w_null_m3(), 9)))[0]))])),
        ("self_checks", ck.items),
        ("theorem45_vs_s3_trio_cells", OrderedDict(
            (f"{a}|{k}", OrderedDict([
                ("t45_median_ratio", t45[a]["median_ratio_raw"][k]),
                ("s3_median_ratio", s3_cells[(pop, conv, a, k)]["median_ratio_raw"]),
                ("t45_num_valid", t45[a]["num_valid_pairs"][k]),
                ("s3_n_kept", s3_cells[(pop, conv, a, k)]["n_kept"]),
                ("A_t45", 1.0 / t45[a]["median_ratio_raw"][k] ** 2),
                ("A_s3", s3_cells[(pop, conv, a, k)]["A_from_median_ratio"]),
            ])) for a in TRIO for k in ATTACKS)),
    ])
    with open(os.path.join(args.out_dir, "regen_ordering_summary.json"), "w") as fh:
        json.dump(to_native(summary), fh, indent=1)
    if not args.quiet:
        print(json.dumps(to_native(summary), indent=1))
    else:
        print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
