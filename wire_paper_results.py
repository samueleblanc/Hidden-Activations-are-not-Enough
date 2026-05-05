"""Substitute \\TODO{...} placeholders in paper LaTeX with computed values.

Reads results/phase1/aggregated/*.json and applies a key->value mapping.
The key inside \\TODO{key} is matched against a substitution dict.
Unmatched \\TODO{} markers are left in place so the paper still typesets
with the [TODO: ...] highlight defined in main.tex's preamble.

Phase-7 Task 7.8 of the CKA-similarity-experiments plan.
"""
import json
import re
from argparse import ArgumentParser
from pathlib import Path


TODO_PATTERN = re.compile(r"\\TODO\{([^}]*)\}")


def substitute_todos(path: str, substitutions: dict):
    """Replace \\TODO{key} with substitutions[key] where key matches.

    Unmatched \\TODO{key} (key not in substitutions) is left in place so
    the LaTeX still compiles with the \\TODO macro from main.tex.
    """
    src = Path(path).read_text()

    def repl(m):
        key = m.group(1)
        if key in substitutions:
            return substitutions[key]
        return m.group(0)

    new = TODO_PATTERN.sub(repl, src)
    Path(path).write_text(new)


def build_substitution_map(results_dir: str) -> dict:
    """Read aggregated JSONs, build placeholder substitutions for the paper.

    Currently wires the headline KM-vs-logit amplification range used in the
    Study-2 (S3) similarity-measure paragraph: ``\\TODO{8--16}\\times``.
    Additional substitutions can be added as more cells of the s1/s2/s3
    tables get computed.
    """
    s3_path = Path(results_dir) / "s3_results.json"
    subs: dict = {}
    if not s3_path.exists():
        return subs

    s3 = json.loads(s3_path.read_text())

    # Compute the typical S3 amplification range (5th--95th percentile of d_M/d_h).
    amp_M_h_list = []
    for key, d in s3.items():
        for pair in d.get("per_pair", []):
            if pair["d_h"] > 1e-6:
                amp_M_h_list.append(pair["d_M"] / pair["d_h"])
    if amp_M_h_list:
        amp_M_h_list.sort()
        # 5th and 95th percentile (rough, integer-index).
        p5 = amp_M_h_list[len(amp_M_h_list) // 20]
        p95 = amp_M_h_list[-len(amp_M_h_list) // 20]
        subs["8--16"] = f"{p5:.1f}--{p95:.1f}"

    return subs


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", default="results/phase1/aggregated")
    parser.add_argument("--paper_dir", default="docs/Final-twist/paper/sections")
    args = parser.parse_args()

    subs = build_substitution_map(args.results_dir)
    print(f"Substitutions: {subs}")

    for tex_file in Path(args.paper_dir).glob("*.tex"):
        substitute_todos(str(tex_file), subs)
        print(f"  Wired: {tex_file}")


if __name__ == "__main__":
    main()
