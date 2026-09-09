"""
Generate LaTeX tables for Theorem 4.5 validation results.

Usage:
    python generate_theorem45_tables.py
    python generate_theorem45_tables.py --experiments alexnet_cifar10 resnet_cifar10
    python generate_theorem45_tables.py --output tables/
"""

import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from constants.constants import ATTACKS

ARCH_DISPLAY = {
    'resnet152':   'ResNet152',
    'densenet121': 'DenseNet121',
    'googlenet':   'GoogLeNet',
    # Legacy (may appear in old JSONs):
    'alexnet': 'AlexNet', 'resnet': 'ResNet', 'vgg': 'VGG', 'lenet': 'LeNet',
}
DATASET_DISPLAY = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
                   'imagenet': 'ImageNet'}


def format_arch(exp_name):
    return ARCH_DISPLAY.get(exp_name.split("_")[0], exp_name.split("_")[0].capitalize())


def format_dataset(exp_name):
    ds = exp_name.split("_", 1)[1]
    return DATASET_DISPLAY.get(ds, ds.upper())


def write_tex(path: Path, lines: list):
    """Write a .tex file with a standard preamble comment."""
    content = "% Requires: \\usepackage{booktabs, amssymb}\n" + "\n".join(lines)
    path.write_text(content)


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_").replace("%", r"\%")


def load_theorem45_json(experiment: str) -> dict:
    """Load theorem45_results.json for an experiment."""
    path = Path(f"experiments/{experiment}/theorem45/theorem45_results.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def select_metric_view(per_attack_result: dict, metric: str) -> dict:
    """Return the (gamma, amp_M_*, amp_h_*, d_*_stats) keyset for a chosen metric.

    metric='raw'  -> top-level keys (||·|| in native ambient norm)
    metric='rms'  -> per-coordinate RMS keys from result['rms']
    """
    if metric == "raw":
        return per_attack_result
    if metric == "rms":
        rms = per_attack_result.get("rms")
        if not rms:
            raise KeyError(
                "metric='rms' requested but result has no 'rms' block. "
                "Run scripts/renormalize_distances.py on the experiment first."
            )
        return rms
    raise ValueError(f"Unknown metric {metric!r}; expected 'raw' or 'rms'.")


def select_aggregate_view(agg: dict, metric: str) -> dict:
    """Aggregate analogue of select_metric_view."""
    if metric == "raw":
        return agg
    if metric == "rms":
        return agg.get("rms", {}) if agg else {}
    raise ValueError(f"Unknown metric {metric!r}")


def generate_theorem45_table(experiments: list, output_dir: Path, metric: str = "rms"):
    """Theorem 4.5 validation table: gamma, d_M/d_f, d_h/d_f, d_M/d_h per attack.

    metric: 'rms' (canonical, per-coordinate RMS) or 'raw' (native norms).
    """
    all_data = {}
    for exp in experiments:
        data = load_theorem45_json(exp)
        if data and data.get('per_attack'):
            all_data[exp] = data

    if not all_data:
        print("No Theorem 4.5 validation data found.")
        return

    metric_tag = "RMS-per-coord" if metric == "rms" else "raw"
    metric_suffix = "" if metric == "rms" else "_raw"  # default tables are RMS
    fmt_amp = "{:.4f}" if metric == "rms" else "{:.2f}"
    fmt_gamma = "{:.4f}" if metric == "rms" else "{:.3f}"
    fmt_ci = "[{:.4f}, {:.4f}]" if metric == "rms" else "[{:.3f}, {:.3f}]"

    # --- Per-experiment detailed tables ---
    for exp_name, data in all_data.items():
        arch = format_arch(exp_name)
        per_attack = data['per_attack']
        attack_order = [a for a in ATTACKS if a in per_attack]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{Theorem~4.5 validation for {arch} ({metric_tag} units): "
            r"knowledge matrix distances lower-bound logit distances "
            r"($\|M(x) - M(x')\| \geq \gamma \cdot \|f(x) - f(x')\|$). "
            + (
                r"All quantities are reported in per-coordinate RMS units "
                r"($\|\cdot\| / \sqrt{\mathrm{numel}}$) so cross-space "
                r"magnitudes are dimensionally comparable across logit, "
                r"penultimate, and KM spaces; see \texttt{utils/scaling.py}."
                if metric == "rms"
                else r"Quantities are raw L2/Frobenius norms in each space's "
                r"native ambient dimension — magnitudes carry a "
                r"$\sqrt{\mathrm{numel}}$ scaling that conflates dimension "
                r"with geometry. Compare to the RMS-units variant for a "
                r"dimensionally-fair view."
            )
            + "}",
            f"\\label{{tab:theorem45_{exp_name}{metric_suffix}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Attack & $\hat{\gamma}$ & $d_M/d_\Psi$ & $d_h/d_\Psi$ & $d_M/d_h$ & $\gamma$ 95\% CI \\",
            r"\midrule",
        ]

        for atk in attack_order:
            raw_r = per_attack[atk]
            try:
                r = select_metric_view(raw_r, metric)
            except KeyError as exc:
                print(f"  WARNING: {exp_name}/{atk}: {exc}")
                continue
            gamma = r['gamma_empirical']
            # Headline cells: mean ± SD (heavy-tailed; the JSON also carries
            # median + IQR for an alternative robust presentation).
            amp_M = r.get('amplification_M_mean', r['amplification_M_median'])
            amp_h = r.get('amplification_h_mean', r.get('amplification_h_median'))
            amp_M_std = r.get('amplification_M_std')
            amp_h_std = r.get('amplification_h_std')
            ratio = (amp_M / amp_h
                     if (amp_h is not None and amp_h > 1e-12) else float('inf'))
            gamma_ci = r.get('gamma_ci_95', [None, None])

            # Bold the d_M/d_h column if > 1 (KMs amplify more per coordinate).
            # Under RMS this is the dimensionally-fair amplification signal.
            ratio_s = fmt_amp.format(ratio) if ratio != float('inf') else r"$\infty$"
            if ratio != float('inf') and ratio > 1.0:
                ratio_s = f"\\textbf{{{ratio_s}}}"

            amp_M_s = (f"{fmt_amp.format(amp_M)} $\\pm$ {fmt_amp.format(amp_M_std)}"
                       if amp_M_std is not None else fmt_amp.format(amp_M))
            if amp_h is None:
                amp_h_s = "---"
            elif amp_h_std is not None:
                amp_h_s = f"{fmt_amp.format(amp_h)} $\\pm$ {fmt_amp.format(amp_h_std)}"
            else:
                amp_h_s = fmt_amp.format(amp_h)

            ci_s = (fmt_ci.format(gamma_ci[0], gamma_ci[1])
                    if gamma_ci[0] is not None and gamma_ci[1] is not None
                    else "---")
            lines.append(
                f"{escape_latex(atk)} & {fmt_gamma.format(gamma)} & {amp_M_s} & "
                f"{amp_h_s} & {ratio_s} & {ci_s} \\\\"
            )

        # Aggregate row — mean ± SD across attacks
        agg_raw = data.get('aggregate', {})
        agg = select_aggregate_view(agg_raw, metric)
        if agg:
            lines.append(r"\midrule")
            g = agg.get('gamma_global', 0)
            mM = agg.get('mean_amplification_M', 0)
            mh = agg.get('mean_amplification_h')
            r_mh = agg.get('ratio_M_over_h')
            r_s = fmt_amp.format(r_mh) if r_mh is not None else "---"
            if r_mh is not None and r_mh > 1.0:
                r_s = f"\\textbf{{{r_s}}}"
            # Cross-attack SD on the per-attack means, in the chosen metric.
            attack_views = []
            for a in attack_order:
                try:
                    attack_views.append(select_metric_view(per_attack[a], metric))
                except KeyError:
                    pass
            attack_means_M = [v.get('amplification_M_mean',
                                     v.get('amplification_M_median'))
                              for v in attack_views]
            attack_means_h = [v.get('amplification_h_mean',
                                     v.get('amplification_h_median'))
                              for v in attack_views
                              if v.get('amplification_h_mean') is not None
                              or v.get('amplification_h_median') is not None]
            mM_std = float(np.std(attack_means_M)) if attack_means_M else 0.0
            mh_std = float(np.std(attack_means_h)) if attack_means_h else 0.0
            mh_cell = (f"{fmt_amp.format(mh)} $\\pm$ {fmt_amp.format(mh_std)}"
                       if mh is not None else "---")
            lines.append(
                f"Overall & {fmt_gamma.format(g)} "
                f"& {fmt_amp.format(mM)} $\\pm$ {fmt_amp.format(mM_std)} "
                f"& {mh_cell} & {r_s} & --- \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]

        fname = f"theorem45_{exp_name}.tex"
        write_tex(output_dir / fname, lines)
        print(f"Generated {output_dir / fname}")

    # --- Cross-experiment summary table ---
    if len(all_data) > 1:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Theorem~4.5 summary across experiments. "
            r"$\hat{\gamma}$ is the global distance lower-bound constant; "
            r"$d_M/d_h$ shows the advantage of knowledge matrices over penultimate features.}",
            r"\label{tab:theorem45_summary}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Arch & Dataset & $\hat{\gamma}$ & $d_M/d_\Psi$ & $d_h/d_\Psi$ & $d_M/d_h$ \\",
            r"\midrule",
        ]

        for exp_name, data in all_data.items():
            arch = format_arch(exp_name)
            dataset = format_dataset(exp_name)
            agg = select_aggregate_view(data.get('aggregate', {}), metric)
            g = agg.get('gamma_global')
            mM = agg.get('mean_amplification_M')
            mh = agg.get('mean_amplification_h')
            r_mh = agg.get('ratio_M_over_h')

            g_s = fmt_gamma.format(g) if g is not None else "---"
            mM_s = fmt_amp.format(mM) if mM is not None else "---"
            mh_s = fmt_amp.format(mh) if mh is not None else "---"
            r_s = fmt_amp.format(r_mh) if r_mh is not None else "---"
            if r_mh is not None and r_mh > 1.0:
                r_s = f"\\textbf{{{r_s}}}"

            lines.append(f"{arch} & {dataset} & {g_s} & {mM_s} & {mh_s} & {r_s} \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        write_tex(output_dir / "theorem45_summary.tex", lines)
        print(f"Generated {output_dir / 'theorem45_summary.tex'}")


def main():
    parser = ArgumentParser(description="Generate Theorem 4.5 LaTeX tables")
    parser.add_argument("--experiments", nargs="+",
                        default=["resnet152_imagenet", "densenet121_imagenet",
                                 "googlenet_imagenet"],
                        help="Experiment names to include (default: 3 main archs)")
    parser.add_argument("--output", type=str, default="tables",
                        help="Output directory for LaTeX files")
    parser.add_argument("--metric", choices=("rms", "raw"), default="rms",
                        help="Distance unit convention. 'rms' (default) reports "
                             "per-coordinate RMS values for dimensionally-fair "
                             "cross-space comparison. 'raw' reports the original "
                             "L2/Frobenius norms used in Theorem 4.5's exact "
                             "inequality but conflates dimension with geometry. "
                             "Requires renormalize_distances.py to have been run "
                             "for 'rms' on pre-2026-05-10 JSON files.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = [exp for exp in args.experiments
                 if Path(f"experiments/{exp}/theorem45/theorem45_results.json").exists()]
    if available:
        print(f"Found Theorem 4.5 results for: {available}  (metric={args.metric})")
        generate_theorem45_table(available, output_dir, metric=args.metric)
    else:
        print("No Theorem 4.5 results found (run validate_theorem45.py first).")

    print(f"\nAll tables written to {output_dir}/")


if __name__ == "__main__":
    main()
