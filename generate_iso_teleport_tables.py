"""Generate LaTeX table for Pillar 1 (teleportation invariance).

Pillar 1A (random neuron permutation) was dropped from the new TMLR direction;
the legacy script lives at legacy/isomorphism_experiment.py. Pillar 1
evidence comes entirely from neural teleportation on the 3 main archs.

Produces:
  tables/teleportation_summary.tex -- penultimate L2 drift + linear CKA
                                      drift under neuralteleportation COB,
                                      with mean ± SD across teleportations.

Usage:
    python generate_iso_teleport_tables.py [--output tables/]
"""

import json
from argparse import ArgumentParser
from pathlib import Path


ARCH_DISPLAY = {
    'resnet152':   'ResNet152',
    'densenet121': 'DenseNet121',
    'googlenet':   'GoogLeNet',
}


TELEPORT_ARCHS = ['resnet152', 'densenet121', 'googlenet']


def load_teleport(arch):
    path = Path(f'results/teleportation/{arch}_imagenet_teleportation.json')
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pm(mean, std, decimals=3):
    """Format mean ± std for a table cell."""
    if mean is None:
        return '---'
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def write_tex(path: Path, lines):
    path.write_text('% Requires: \\usepackage{booktabs, amssymb}\n' + '\n'.join(lines))


def generate_teleport_table(output_dir):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pillar 1 (Neural teleportation). Change-of-basis (COB) '
        r'transformations via \texttt{neuralteleportation} produce functionally '
        r'equivalent networks with different weights. Penultimate features drift '
        r'substantially under L2 ($\|h - h^\prime\|_2 / \sqrt{d}$) and even under '
        r'linear CKA ($1 - \mathrm{CKA}(h, h^\prime)$, the field-standard '
        r'representation-similarity metric); knowledge matrices remain '
        r'theoretically invariant. Mean $\pm$ SD across teleportations.}',
        r'\label{tab:teleport_summary}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Architecture & '
        r'$\|h - h^\prime\|/\sqrt{d}$ (test) & '
        r'$\|h - h^\prime\|/\sqrt{d}$ (random) & '
        r'$1 - \mathrm{CKA}(h, h^\prime)$ (test) & '
        r'Pred.\ match \\',
        r'\midrule',
    ]
    for arch in TELEPORT_ARCHS:
        d = load_teleport(arch)
        if not d:
            lines.append(f"{ARCH_DISPLAY.get(arch, arch)} & --- & --- & --- & --- \\\\")
            continue
        agg = d['aggregate']
        per = d['per_teleportation']
        nt = len(per)
        fails = sum(
            1 for r in per
            if not r['output_equivalence']['all_predictions_match']
        )
        match_cell = 'yes' if fails == 0 else f"{nt - fails}/{nt}"
        arch_display = ARCH_DISPLAY.get(arch, arch)

        # L2 drift columns (mean_of_means ± std_of_means)
        test_l2 = fmt_pm(agg['test']['mean_of_means'],
                         agg['test'].get('std_of_means'))
        rand_l2 = fmt_pm(agg['random']['mean_of_means'],
                         agg['random'].get('std_of_means'))

        # CKA distance column (1 - CKA), if present.
        test_block = agg.get('test', {})
        cka_mean = test_block.get('cka_linear_1m_mean')
        cka_std = test_block.get('cka_linear_1m_std')
        cka_cell = fmt_pm(cka_mean, cka_std)

        lines.append(
            f"{arch_display} & {test_l2} & {rand_l2} & {cka_cell} & {match_cell} \\\\"
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    out = output_dir / 'teleportation_summary.tex'
    write_tex(out, lines)
    print(f"Generated {out}")


def main():
    p = ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--output', default='tables',
                   help='Output directory for LaTeX files.')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    generate_teleport_table(out)
    print(f"\nTable written to {out}/")


if __name__ == '__main__':
    main()
