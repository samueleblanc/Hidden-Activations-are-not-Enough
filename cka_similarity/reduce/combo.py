"""Phase-1 Step C — single-combo array entrypoint.

The serial reduce timed out because ``aggregate_s1`` runs the expensive
optimal-transport measures (entropic Gromov-Wasserstein + Sinkhorn
soft-matching) for every (arch × teleport) combo serially at finalize. This
module lets a SLURM array compute ONE combo per task and persist it to a
shared ``combo_dir``; the gather (``python -m cka_similarity.reduce
--combo_dir <dir>``) then assembles the cached combos with no recompute.

Modes
-----
``--count``  : print the number of combos N (the array bound: ``--array=0-(N-1)``).
``--index I`` : compute combo I (mapped via ``enumerate_combos``) into ``combo_dir``.
``--list``   : print ``index<TAB>stage<TAB>key`` for every combo (debugging).

The combo set is defined ONLY by ``enumerate_combos`` so the array bound and
the index→combo map can never drift from what the gather expects. Computing a
combo that is already cached is a no-op (idempotent ⇒ a resubmitted array
resumes).
"""
from argparse import ArgumentParser

from cka_similarity.reduce.aggregate import enumerate_combos, run_combo


def _add_combo_args(parser: ArgumentParser) -> None:
    parser.add_argument("--archs", nargs="+",
                        default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--attacks", nargs="+",
                        default=["fgsm", "pgd", "cw", "deepfool", "apgd", "square"])
    parser.add_argument("--num_teleports", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=64)


def main():
    parser = ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count", action="store_true",
                      help="Print the number of combos (the array upper bound).")
    mode.add_argument("--list", action="store_true",
                      help="Print 'index<TAB>stage<TAB>key' for every combo.")
    mode.add_argument("--index", type=int,
                      help="Compute the combo at this index into --combo_dir.")
    _add_combo_args(parser)
    parser.add_argument("--combo_dir",
                        help="Shared dir holding per-combo cache files (required with --index).")
    parser.add_argument("--s1_dir", default="results/phase1/s1")
    parser.add_argument("--s2_dir", default="results/phase1/s2")
    parser.add_argument("--s3_dir", default="results/phase1/s3")
    args = parser.parse_args()

    combos = enumerate_combos(args.archs, args.num_teleports, args.attacks)

    if args.count:
        print(len(combos))
        return

    if args.list:
        for i, (stage, key, _params) in enumerate(combos):
            print(f"{i}\t{stage}\t{key}")
        return

    # --index mode
    if args.combo_dir is None:
        parser.error("--index requires --combo_dir")
    if not (0 <= args.index < len(combos)):
        parser.error(f"--index {args.index} out of range [0, {len(combos)})")

    import os
    os.makedirs(args.combo_dir, exist_ok=True)

    stage, key, params = combos[args.index]
    print(f"=== combo {args.index}/{len(combos)}: stage={stage} key={key} "
          f"params={params} ===", flush=True)
    run_combo(stage, params, args.combo_dir, args.num_chunks,
              args.s1_dir, args.s2_dir, args.s3_dir)
    print(f"  cached -> {args.combo_dir}/{key}.pt", flush=True)


if __name__ == "__main__":
    main()
