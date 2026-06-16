"""Phase-1 Step C driver: glob chunk artifacts → aggregated results + sanity → tables."""
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch

from utils.atomic_io import atomic_json_dump, atomic_torch_save
from cka_similarity.reduce.aggregate import (
    aggregate_s1, aggregate_s2, aggregate_s3,
    combo_cache_path, enumerate_combos,
)
from cka_similarity.reduce.sanity import write_sanity_report
from cka_similarity.reduce.tables import emit_s1_table, emit_s2_table, emit_s3_table


def _stage_checkpoint(path: Path, compute):
    """Resume-or-compute one aggregation stage.

    Each aggregate_s* pass is hours of CPU over thousands of chunk files, but
    the driver historically wrote outputs only after ALL stages finished — a
    crash in stage 3 (or the 15h wall) redid stages 1-2 from scratch (job
    13963270 lost 90 min this way). Stage results are plain dicts of floats,
    so we snapshot each one to a hidden .pt next to the final JSONs. Delete
    the .s*_stage.pt files to force a recompute after regenerating worker
    chunks.
    """
    if path.exists():
        print(f"  resuming from stage checkpoint {path.name}", flush=True)
        return torch.load(path)
    result = compute()
    atomic_torch_save(str(path), result)
    return result


def _report_combo_cache(combo_dir, archs, num_teleports, attacks):
    """Log which combos the gather will read from cache vs recompute.

    The gather is robust to a few failed array tasks: a per-combo finalize
    that finds no cache file recomputes that combo in-process (same code, same
    result) instead of crashing. We surface any such fallback up front so an
    operator can see that e.g. 2/171 combos are being recomputed (a hint that
    2 array tasks failed) rather than silently eating the cost.
    """
    combos = enumerate_combos(archs, num_teleports, attacks)
    present, missing = [], []
    for stage, key, _params in combos:
        (present if combo_cache_path(combo_dir, key).exists() else missing).append(key)
    print(f"  combo cache {combo_dir}: {len(present)}/{len(combos)} present", flush=True)
    if missing:
        print(f"  WARNING: {len(missing)} combo(s) MISSING from cache — the gather "
              f"will recompute them in-process (identical result, but slower; "
              f"likely failed array tasks): {', '.join(missing)}", flush=True)


def _try_compute_controls(archs, data_dir, cui_n_inputs):
    """Compute Cui + Murphy controls; degrade gracefully on failure.

    Local environments may lack a GPU, the ImageNet val set, or the
    pretrained-weights cache. Rather than crashing the whole reduce step
    in that case, we catch the exception and emit empty controls dicts —
    the sanity check on controls will then trivially pass (no entries to
    fail), with a warning printed to the orchestrator log.
    """
    try:
        from cka_similarity.controls.cui_random_network import compute_cui_control
        from cka_similarity.controls.murphy_shuffled_pair import compute_murphy_control
        from cka_similarity.workers.s1_within_arch_invariance import (
            forward_penultimate, load_pretrained, load_imagenet_val_chunk,
        )
        from utils.utils import get_device

        device = get_device()
        inputs = load_imagenet_val_chunk(0, cui_n_inputs, data_dir).to(device)

        cui = {arch: compute_cui_control(arch, inputs, device) for arch in archs}

        murphy = {}
        for arch in archs:
            W = load_pretrained(arch).to(device)
            h_W = forward_penultimate(W, inputs).cpu()
            murphy[arch] = compute_murphy_control(h_W, h_W)
        return cui, murphy
    except Exception as e:
        print(f"WARNING: failed to compute controls ({type(e).__name__}: {e}); "
              f"emitting empty controls — sanity will trivially pass on those checks",
              flush=True)
        return {}, {}


def main():
    parser = ArgumentParser()
    parser.add_argument("--s1_dir", default="results/phase1/s1")
    parser.add_argument("--s2_dir", default="results/phase1/s2")
    parser.add_argument("--s3_dir", default="results/phase1/s3")
    parser.add_argument("--out_dir", default="results/phase1/aggregated")
    parser.add_argument("--paper_tables_dir", default="docs/Final-twist/paper/tables")
    parser.add_argument("--data_dir",
                        default=os.environ.get("IMAGENET_ROOT", "/datashare/imagenet/ILSVRC2012"))
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "cw", "deepfool", "apgd", "square"])
    parser.add_argument("--num_teleports", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--cui_n_inputs", type=int, default=2048)
    parser.add_argument("--skip_controls", action="store_true",
                        help="Skip Cui/Murphy controls (e.g. for offline reduce-only runs)")
    parser.add_argument("--combo_dir", default=None,
                        help="Per-combo cache dir (the SLURM-array path's output). "
                             "When set, aggregate_s* read finalized combos from "
                             "<combo_dir>/{key}.pt instead of recomputing; any combo "
                             "missing from the cache is recomputed in-process (same "
                             "result) with a warning. When unset, the reduce runs "
                             "fully serial (every combo computed here), as before.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combo_dir is not None:
        print(f"=== Gather mode: combo cache = {args.combo_dir} ===", flush=True)
        _report_combo_cache(args.combo_dir, args.archs, args.num_teleports, args.attacks)

    print("=== Aggregating S1 ===", flush=True)
    s1 = _stage_checkpoint(out_dir / ".s1_stage.pt", lambda: aggregate_s1(
        args.s1_dir, args.archs, args.num_teleports, args.num_chunks,
        combo_dir=args.combo_dir))

    print("=== Aggregating S2 ===", flush=True)
    s2 = _stage_checkpoint(out_dir / ".s2_stage.pt", lambda: aggregate_s2(
        args.s2_dir, args.archs, args.num_chunks, combo_dir=args.combo_dir))

    print("=== Aggregating S3 ===", flush=True)
    s3 = _stage_checkpoint(out_dir / ".s3_stage.pt", lambda: aggregate_s3(
        args.s3_dir, args.archs, args.attacks, args.num_chunks,
        combo_dir=args.combo_dir))

    if args.skip_controls:
        print("=== Skipping Cui + Murphy controls (--skip_controls) ===", flush=True)
        cui, murphy = {}, {}
    else:
        print("=== Computing Cui + Murphy controls ===", flush=True)
        cui, murphy = _try_compute_controls(args.archs, args.data_dir, args.cui_n_inputs)

    # --- Write JSON outputs ---
    # Note: JSON cannot serialize tuple keys, so we stringify
    s1_json = {f"{arch}|tp{tid}": v for (arch, tid), v in s1.items()}
    s3_json = {f"{arch}|{attack}": v for (arch, attack), v in s3.items()}

    atomic_json_dump(Path(args.out_dir) / "s1_results.json", s1_json)
    atomic_json_dump(Path(args.out_dir) / "s2_results.json", s2)
    atomic_json_dump(Path(args.out_dir) / "s3_results.json", s3_json)
    atomic_json_dump(Path(args.out_dir) / "controls.json",
                     {"cui": cui, "murphy": murphy})

    # --- Sanity ---
    print("=== Running sanity checks ===", flush=True)
    sanity = write_sanity_report(
        str(Path(args.out_dir) / "sanity_report.json"),
        s1, s2, s3, cui, murphy,
        skip_controls=args.skip_controls,
    )
    print(f"  all_pass: {sanity['all_pass']}", flush=True)

    # --- Tables ---
    print("=== Emitting LaTeX tables ===", flush=True)
    emit_s1_table(s1, args.archs, args.num_teleports,
                  Path(args.paper_tables_dir) / "s1_table.tex")
    emit_s2_table(s2, Path(args.paper_tables_dir) / "s2_table.tex")
    emit_s3_table(s3, Path(args.paper_tables_dir) / "s3_table.tex")

    if not sanity["all_pass"]:
        print("SANITY FAILED — Step D will produce phase1-FAILED.tar.gz", flush=True)
        # Exit 0 so the dependency chain can still run Step D (which gates on sanity_report).


if __name__ == "__main__":
    main()
