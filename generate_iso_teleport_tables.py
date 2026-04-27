"""Generate LaTeX tables for Pillar 1 (isomorphism + teleportation).

Produces:
  tables/isomorphism_summary.tex   -- penultimate vs knowledge-matrix change
                                      under random neuron permutations
  tables/teleportation_summary.tex -- penultimate change under
                                      neuralteleportation COB for 3 archs

Usage:
    python generate_iso_teleport_tables.py [--output tables/]
"""

import json
from argparse import ArgumentParser
from pathlib import Path


ARCH_DISPLAY = {
    'alexnet':     'AlexNet',
    'resnet':      'ResNet18',   # iso (legacy) uses resnet18 via arch_idx=-2
    'resnet152':   'ResNet152',  # Pillar 3 (commit 5cf31fc): wide-face permutation
    'densenet121': 'DenseNet121',
    'googlenet':   'GoogLeNet',
    'vgg':         'VGG11',      # iso (legacy) uses vgg11 via arch_idx=-1
    'vgg11':       'VGG11',
    'resnet18':    'ResNet18',
    'resnet50':    'ResNet50',
}


# Pillar 3 alignment: Step A (iso) is now resnet152-only because densenet121
# and googlenet have concat-based topologies that don't admit simple post-pool
# neuron permutations (see isomorphism_experiment.py:_STEP_A_UNSUPPORTED_MODELS).
# Pillar-1 evidence on densenet121/googlenet comes from Step B teleportation
# (which DOES support all three architectures via neuralteleportation COB).
ISO_EXPERIMENTS = ['resnet152_imagenet']
TELEPORT_ARCHS  = ['resnet152', 'densenet121', 'googlenet']

# Relative errors measured via `--debug` iso reruns. Updated 2026-04-27:
# legacy {alexnet,resnet18,vgg11} numbers retained for reference but no
# longer surfaced in the table (since ISO_EXPERIMENTS no longer includes
# them). Pillar 3 entry will be populated once the resnet152 cluster job
# runs with --debug; until then the value is None and the table prints '---'.
KNOWN_REL_ERRORS = {
    # Legacy (kept for reference):
    'alexnet_imagenet': 1.146e-7,  # mean of seed=42 (1.145e-7) + seed=43 (1.146e-7)
    'resnet_imagenet':  3.48e-4,   # mean of seed=42 (5.04e-4) + seed=43 (1.92e-4)
    'vgg_imagenet':     1.231e-7,  # mean of seed=42 (1.232e-7) + seed=43 (1.231e-7)
    # Pillar 3 (TODO — populate after the cluster --debug run on resnet152):
    'resnet152_imagenet': None,
}


def format_iso_arch(exp_name):
    return ARCH_DISPLAY.get(exp_name.split('_')[0],
                            exp_name.split('_')[0].capitalize())


def write_tex(path: Path, lines):
    path.write_text('% Requires: \\usepackage{booktabs, amssymb}\n' + '\n'.join(lines))


def iso_relative_error(exp_name):
    """Mean relative error if captured via --debug, else the known fallback."""
    path = Path(f'experiments/{exp_name}/isomorphism/isomorphism_results.json')
    if not path.exists():
        return KNOWN_REL_ERRORS.get(exp_name)
    with open(path) as f:
        d = json.load(f)
    rels = []
    for p in d.get('per_permutation', []):
        mc = p.get('matrix_change', {})
        if 'mean_relative_error' in mc:
            rels.append(mc['mean_relative_error'])
    if rels:
        return sum(rels) / len(rels)
    return KNOWN_REL_ERRORS.get(exp_name)


def load_iso(exp_name):
    path = Path(f'experiments/{exp_name}/isomorphism/isomorphism_results.json')
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_teleport(arch):
    path = Path(f'results/teleportation/{arch}_imagenet_teleportation.json')
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_iso_table(output_dir):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pillar 1 (Isomorphism invariance). Random neuron permutations '
        r'within the penultimate channel/layer group produce an isomorphic network '
        r'that computes the same function (predictions match exactly). Penultimate '
        r'activations change substantially ($\|h(x)-h(x\prime)\|$, absolute L2), '
        r'while knowledge matrices remain essentially invariant '
        r'($\|M(x)-M(x\prime)\|$, absolute Frobenius; relative to mean KM norm).}',
        r'\label{tab:iso_summary}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Architecture & '
        r'$\|h - h^\prime\|$ & '
        r'$\|M - M^\prime\|$ & '
        r'rel.\ $\|M - M^\prime\|$ & '
        r'Pred.\ match \\',
        r'\midrule',
    ]
    for exp in ISO_EXPERIMENTS:
        d = load_iso(exp)
        if not d:
            continue
        s = d['summary']
        arch = format_iso_arch(exp)
        act = s['activation_change']
        mat = s['matrix_change']
        pm = 'yes' if s['all_predictions_match'] else 'no'
        rel = iso_relative_error(exp)
        rel_s = f"{rel:.1e}" if rel is not None else '---'
        lines.append(
            f"{arch} & {act:.2f} & {mat:.2e} & {rel_s} & {pm} \\\\"
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    out = output_dir / 'isomorphism_summary.tex'
    write_tex(out, lines)
    print(f"Generated {out}")


def generate_teleport_table(output_dir):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pillar 1 (Neural teleportation). Change-of-basis (COB) '
        r'transformations via \texttt{neuralteleportation} produce functionally '
        r'equivalent networks with different weights. Penultimate activations '
        r'shift substantially (normalized $\|h - h^\prime\|/\sqrt{d}$, mean over '
        r'100 teleportations of 500 samples/split).}',
        r'\label{tab:teleport_summary}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Architecture & train & test & random & Pred.\ match (all teleports) \\',
        r'\midrule',
    ]
    for arch in TELEPORT_ARCHS:
        d = load_teleport(arch)
        if not d:
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
        lines.append(
            f"{arch_display} & {agg['train']['mean_of_means']:.3f} "
            f"& {agg['test']['mean_of_means']:.3f} "
            f"& {agg['random']['mean_of_means']:.3f} "
            f"& {match_cell} \\\\"
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
    p = ArgumentParser()
    p.add_argument('--output', default='tables',
                   help='Output directory for LaTeX files.')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    generate_iso_table(out)
    generate_teleport_table(out)
    print(f"\nTables written to {out}/")


if __name__ == '__main__':
    main()
