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


def generate_theorem45_table(experiments: list, output_dir: Path):
    """Theorem 4.5 validation table: gamma, d_M/d_f, d_h/d_f, d_M/d_h per attack."""
    all_data = {}
    for exp in experiments:
        data = load_theorem45_json(exp)
        if data and data.get('per_attack'):
            all_data[exp] = data

    if not all_data:
        print("No Theorem 4.5 validation data found.")
        return

    # --- Per-experiment detailed tables ---
    for exp_name, data in all_data.items():
        arch = format_arch(exp_name)
        per_attack = data['per_attack']
        attack_order = [a for a in ATTACKS if a in per_attack]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{Theorem~4.5 validation for {arch}: knowledge matrix distances "
            r"lower-bound logit distances ($\|M(x) - M(x')\| \geq \gamma \cdot \|f(x) - f(x')\|$). "
            r"$d_M/d_h > 1$ shows KMs amplify more than penultimate features.}",
            f"\\label{{tab:theorem45_{exp_name}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Attack & $\hat{\gamma}$ & $d_M/d_f$ & $d_h/d_f$ & $d_M/d_h$ & $\gamma$ 95\% CI \\",
            r"\midrule",
        ]

        for atk in attack_order:
            r = per_attack[atk]
            gamma = r['gamma_empirical']
            # Headline cells: mean ± SD (heavy-tailed; the JSON also carries
            # median + IQR for an alternative robust presentation).
            amp_M = r.get('amplification_M_mean', r['amplification_M_median'])
            amp_h = r.get('amplification_h_mean', r['amplification_h_median'])
            amp_M_std = r.get('amplification_M_std')
            amp_h_std = r.get('amplification_h_std')
            ratio = amp_M / amp_h if amp_h > 1e-12 else float('inf')
            gamma_ci = r.get('gamma_ci_95', [None, None])

            # Bold the d_M/d_h column if > 1 (KMs better)
            ratio_s = f"{ratio:.2f}"
            if ratio > 1.0:
                ratio_s = f"\\textbf{{{ratio_s}}}"

            amp_M_s = (f"{amp_M:.2f} $\\pm$ {amp_M_std:.2f}"
                       if amp_M_std is not None else f"{amp_M:.2f}")
            amp_h_s = (f"{amp_h:.2f} $\\pm$ {amp_h_std:.2f}"
                       if amp_h_std is not None else f"{amp_h:.2f}")

            ci_s = f"[{gamma_ci[0]:.3f}, {gamma_ci[1]:.3f}]" if gamma_ci[0] is not None and gamma_ci[1] is not None else "---"
            lines.append(
                f"{escape_latex(atk)} & {gamma:.3f} & {amp_M_s} & "
                f"{amp_h_s} & {ratio_s} & {ci_s} \\\\"
            )

        # Aggregate row — mean ± SD across attacks
        agg = data.get('aggregate', {})
        if agg:
            lines.append(r"\midrule")
            g = agg.get('gamma_global', 0)
            mM = agg.get('mean_amplification_M', 0)
            mh = agg.get('mean_amplification_h', 0)
            r_mh = agg.get('ratio_M_over_h')
            r_s = f"{r_mh:.2f}" if r_mh is not None else "---"
            if r_mh is not None and r_mh > 1.0:
                r_s = f"\\textbf{{{r_s}}}"
            # Cross-attack SD on the per-attack means
            attack_means_M = [per_attack[a].get('amplification_M_mean',
                                                 per_attack[a]['amplification_M_median'])
                              for a in attack_order]
            attack_means_h = [per_attack[a].get('amplification_h_mean',
                                                 per_attack[a]['amplification_h_median'])
                              for a in attack_order]
            mM_std = float(np.std(attack_means_M)) if attack_means_M else 0.0
            mh_std = float(np.std(attack_means_h)) if attack_means_h else 0.0
            lines.append(
                f"Overall & {g:.3f} & {mM:.2f} $\\pm$ {mM_std:.2f} "
                f"& {mh:.2f} $\\pm$ {mh_std:.2f} & {r_s} & --- \\\\"
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
            r"Arch & Dataset & $\hat{\gamma}$ & $d_M/d_f$ & $d_h/d_f$ & $d_M/d_h$ \\",
            r"\midrule",
        ]

        for exp_name, data in all_data.items():
            arch = format_arch(exp_name)
            dataset = format_dataset(exp_name)
            agg = data.get('aggregate', {})
            g = agg.get('gamma_global')
            mM = agg.get('mean_amplification_M')
            mh = agg.get('mean_amplification_h')
            r_mh = agg.get('ratio_M_over_h')

            g_s = f"{g:.3f}" if g is not None else "---"
            mM_s = f"{mM:.2f}" if mM is not None else "---"
            mh_s = f"{mh:.2f}" if mh is not None else "---"
            r_s = f"{r_mh:.2f}" if r_mh is not None else "---"
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
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = [exp for exp in args.experiments
                 if Path(f"experiments/{exp}/theorem45/theorem45_results.json").exists()]
    if available:
        print(f"Found Theorem 4.5 results for: {available}")
        generate_theorem45_table(available, output_dir)
    else:
        print("No Theorem 4.5 results found (run validate_theorem45.py first).")

    print(f"\nAll tables written to {output_dir}/")


if __name__ == "__main__":
    main()
