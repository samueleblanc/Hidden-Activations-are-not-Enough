"""S3 worker: within-arch distance amplification, 9-measure panel on adversarial pairs.

Per chunk: load adversarial pairs from
experiments/{arch}_imagenet/adversarial_pairs_N5000/{attack}/pairs.pth;
slice to this chunk; for each pair compute d_f, d_h, d_M (Theorem 4.5
quantities); also accumulate panel measures on stacked clean-vs-adv features.

Output:
- per-pair distance list:      results/phase1/s3/{arch}_{attack}_chunk{i}.json
- per-(arch,attack,chunk) panel: results/phase1/s3/{arch}_{attack}_panel_chunk{i}.pt
"""
import json
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import torch

from utils.utils import get_device
from utils.atomic_io import atomic_json_dump, atomic_json_load, atomic_torch_save
from utils.scaling import (
    IMAGENET_INPUT_NUMEL, IMAGENET_NUM_CLASSES, penultimate_dim, km_numel,
)
from cka_similarity.workers.common import chunk_slice, load_active_km_batch_size, calibration_path_for
from cka_similarity.measures.panel import PANEL
# Pull the wrapper-aware loaders from S2 (NOT S1): S3 feeds models into
# KnowledgeMatrixComputer which requires ``.layers`` / ``.input_shape``,
# both of which the knowledgematrix wrapper exposes natively. S1's
# ``load_pretrained`` returns a COB-aware torchvision model — fine for
# teleportation, broken for KMC.
from cka_similarity.workers.s2_cross_architecture import (
    load_pretrained, forward_penultimate, forward_logits, extract_km_per_sample,
)


def _write_complete_sentinel_if_done(out_dir, archs, attacks, num_chunks):
    """Touch ``{out_dir}/.complete`` once every per-chunk file has landed.

    For each (arch, attack) pair this worker emits two files per chunk:
    ``{arch}_{attack}_chunk{i}.json`` and ``{arch}_{attack}_panel_chunk{i}.pt``.
    We probe the full grid here so whichever chunk-task happens to land
    last detects the all-done state and marks the sentinel for
    run_pipeline.sh. ``Path.touch()`` is idempotent — a TOCTOU race
    between two simultaneously-finishing tasks is benign.
    """
    out_path = Path(out_dir)
    for arch, attack in product(archs, attacks):
        for ci in range(num_chunks):
            for kind in ("chunk{}.json", "panel_chunk{}.pt"):
                expected = out_path / f"{arch}_{attack}_{kind.format(ci)}"
                if not expected.exists():
                    return
    (out_path / ".complete").touch()


def run_chunk(chunk_id, num_chunks, total_pairs_per_attack, archs, attacks, out_dir,
              pairs_root, km_batch_divisor=1):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    start, end = chunk_slice(chunk_id, total_pairs_per_attack, num_chunks)
    n_pairs = end - start

    panel_cls = list(PANEL)

    for arch in archs:
        # Calibration is per-arch; if it failed for one arch (e.g. CUDA-OOM
        # tiered out), keep processing the other archs rather than aborting
        # the whole chunk task. The orchestrator's afterany dep means S3 may
        # legitimately fire on partial calibration.
        try:
            # Calibration sizes KM extraction to ~85% of a *dedicated* GPU
            # (resnet152 -> bs=1088 -> ~73 GiB). On the packed gpubase_bygpu
            # partition tasks can share a physical GPU, so we divide the
            # calibrated batch down to leave headroom for a co-resident peer.
            # km_batch_divisor only changes memory tiling, never M(x).
            bs = max(1, load_active_km_batch_size(calibration_path_for(arch)) // km_batch_divisor)
        except FileNotFoundError as e:
            print(
                f"WARNING: calibration missing for {arch} ({e}); skipping all "
                f"({arch}, *) attacks in this chunk",
                flush=True,
            )
            continue
        model = load_pretrained(arch).to(device)

        for attack in attacks:
            pairs_path = Path(pairs_root) / f"{arch}_imagenet" / "adversarial_pairs_N5000" / attack / "pairs.pth"
            if not pairs_path.exists():
                print(
                    f"WARNING: {pairs_path} missing; skipping ({arch}, {attack})",
                    flush=True,
                )
                continue
            try:
                pairs = torch.load(pairs_path, map_location="cpu")
            except Exception as e:
                print(
                    f"WARNING: failed to load {pairs_path}: {e}; skipping ({arch}, {attack})",
                    flush=True,
                )
                continue

            # If the pairs file is partially populated, clamp the chunk slice
            # to the prefix that is actually filled in. The N=5000 scale-up
            # writes ``n_done`` after each successful sample so a partially
            # complete file is still usable up to that index.
            attack_end = end
            if isinstance(pairs, dict) and "n_done" in pairs and pairs["n_done"] < end:
                print(
                    f"WARNING: {pairs_path} has n_done={pairs['n_done']} < end={end}; "
                    f"using partial chunk",
                    flush=True,
                )
                attack_end = min(end, int(pairs["n_done"]))
            attack_n_pairs = attack_end - start
            if attack_n_pairs <= 0:
                # DeepFool on ImageNet is too slow to reach target_n=5000;
                # we accept the partial pairs.pth on disk (e.g. n_done=224 for
                # resnet152, 928 for densenet121/googlenet). For chunks past
                # n_done we emit empty markers so the full (arch, attack, chunk)
                # grid still exists on disk — _write_complete_sentinel_if_done
                # then fires and the orchestrator stops re-queueing S3.
                # aggregate_s3 filters n_pairs==0 chunks out of panel finalize.
                print(
                    f"WARNING: ({arch}, {attack}) chunk {chunk_id} has zero usable pairs after "
                    f"clamping to n_done; writing empty markers",
                    flush=True,
                )
                dist_path = Path(out_dir) / f"{arch}_{attack}_chunk{chunk_id}.json"
                if not dist_path.exists():
                    atomic_json_dump(str(dist_path), [])
                panel_path = Path(out_dir) / f"{arch}_{attack}_panel_chunk{chunk_id}.pt"
                if not panel_path.exists():
                    atomic_torch_save(str(panel_path), {
                        "chunk_id": chunk_id, "arch": arch, "attack": attack,
                        "n_pairs": 0, "accumulators": {},
                    })
                continue

            x_clean = pairs["x_clean"][start:attack_end].to(device)
            x_adv   = pairs["x_adv"][start:attack_end].to(device)

            # Theorem 4.5 quantities: per-pair d_f, d_h, d_M
            with torch.no_grad():
                f_clean = model(x_clean)
                f_adv   = model(x_adv)
            h_clean = forward_penultimate(model, x_clean).cpu()
            h_adv   = forward_penultimate(model, x_adv).cpu()
            M_clean = extract_km_per_sample(model, x_clean, batch_size=bs)
            M_adv   = extract_km_per_sample(model, x_adv,   batch_size=bs)

            # Append per-pair distances. Both raw and RMS-per-coordinate
            # values are saved: raw norms preserve Theorem 4.5's exact
            # statement, RMS values enable fair cross-space comparison.
            # See utils/scaling.py and docs/Final-twist/km-notes.md (2026-05-10).
            D_penult = penultimate_dim(arch)
            s_f = 1.0 / (IMAGENET_NUM_CLASSES ** 0.5)
            s_h = 1.0 / (D_penult ** 0.5)
            s_M = 1.0 / (km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL) ** 0.5)
            dist_path = Path(out_dir) / f"{arch}_{attack}_chunk{chunk_id}.json"
            existing = atomic_json_load(str(dist_path), default=[])
            for i in range(len(existing), attack_n_pairs):
                completeness_clean = float((M_clean[i].sum(1) - f_clean[i].cpu()).abs().max())
                completeness_adv   = float((M_adv[i].sum(1)   - f_adv[i].cpu()).abs().max())
                d_f_raw = float((f_clean[i] - f_adv[i]).norm(p=2))
                d_h_raw = float((h_clean[i] - h_adv[i]).norm(p=2))
                d_M_raw = float((M_clean[i] - M_adv[i]).norm(p='fro'))
                existing.append({
                    "pair_idx": start + i,
                    "d_f": d_f_raw,
                    "d_h": d_h_raw,
                    "d_M": d_M_raw,
                    "d_f_rms": d_f_raw * s_f,
                    "d_h_rms": d_h_raw * s_h,
                    "d_M_rms": d_M_raw * s_M,
                    "completeness_residual_clean": completeness_clean,
                    "completeness_residual_adv":   completeness_adv,
                })
                atomic_json_dump(str(dist_path), existing)

            # Panel accumulators on stacked features (clean vs adv)
            panel_path = Path(out_dir) / f"{arch}_{attack}_panel_chunk{chunk_id}.pt"
            if not panel_path.exists():
                accumulators = {}
                for cls in panel_cls:
                    m = cls()
                    if cls.__name__ == "OutputJSD":
                        accumulators[m.name] = m.accumulate(f_clean.cpu(), f_adv.cpu())
                    else:
                        accumulators[m.name] = m.accumulate(h_clean, h_adv)
                atomic_torch_save(str(panel_path), {
                    "chunk_id": chunk_id, "arch": arch, "attack": attack,
                    "n_pairs": attack_n_pairs,
                    "accumulators": accumulators,
                })

            # Free the heavy per-attack tensors before moving to the next attack
            # — otherwise each iteration accumulates ~2 × KM-batch worth of CPU
            # memory plus the GPU input batches across the full attack list.
            del x_clean, x_adv, f_clean, f_adv, h_clean, h_adv, M_clean, M_adv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Once this chunk's outputs land, check whether the full grid is on disk
    # and, if so, mark the worker complete for run_pipeline.sh.
    _write_complete_sentinel_if_done(out_dir, archs, attacks, num_chunks)


def main():
    parser = ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--total_pairs", type=int, default=5000)
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "cw", "deepfool", "apgd", "square"])
    parser.add_argument("--out_dir", default="results/phase1/s3")
    parser.add_argument("--pairs_root", default="experiments")
    parser.add_argument(
        "--km_batch_divisor", type=int, default=1,
        help="Divide the calibrated KM batch_size by this factor to leave GPU "
             "headroom for co-located tasks on the packed gpubase_bygpu "
             "partition (memory-only; M(x) is unchanged). 1 = use calibration.",
    )
    args = parser.parse_args()

    run_chunk(args.chunk_id, args.num_chunks, args.total_pairs, args.archs,
              args.attacks, args.out_dir, args.pairs_root,
              km_batch_divisor=args.km_batch_divisor)


if __name__ == "__main__":
    main()
