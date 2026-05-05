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
from pathlib import Path

import torch

from utils.utils import get_device
from utils.atomic_io import atomic_json_dump, atomic_json_load, atomic_torch_save
from cka_similarity.workers.common import chunk_slice, load_active_km_batch_size, calibration_path_for
from cka_similarity.measures.panel import PANEL
from cka_similarity.workers.s1_within_arch_invariance import (
    load_pretrained, forward_penultimate, forward_logits,
)
from cka_similarity.workers.s2_cross_architecture import extract_km_per_sample


def run_chunk(chunk_id, num_chunks, total_pairs_per_attack, archs, attacks, out_dir, pairs_root):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    start, end = chunk_slice(chunk_id, total_pairs_per_attack, num_chunks)
    n_pairs = end - start

    panel_cls = list(PANEL)

    for arch in archs:
        model = load_pretrained(arch).to(device)
        bs = load_active_km_batch_size(calibration_path_for(arch))

        for attack in attacks:
            pairs_path = Path(pairs_root) / f"{arch}_imagenet" / "adversarial_pairs_N5000" / attack / "pairs.pth"
            pairs = torch.load(pairs_path, map_location="cpu")
            x_clean = pairs["x_clean"][start:end].to(device)
            x_adv   = pairs["x_adv"][start:end].to(device)

            # Theorem 4.5 quantities: per-pair d_f, d_h, d_M
            with torch.no_grad():
                f_clean = model(x_clean)
                f_adv   = model(x_adv)
            h_clean = forward_penultimate(model, x_clean).cpu()
            h_adv   = forward_penultimate(model, x_adv).cpu()
            M_clean = extract_km_per_sample(model, x_clean, batch_size=bs)
            M_adv   = extract_km_per_sample(model, x_adv,   batch_size=bs)

            # Append per-pair distances
            dist_path = Path(out_dir) / f"{arch}_{attack}_chunk{chunk_id}.json"
            existing = atomic_json_load(str(dist_path), default=[])
            for i in range(len(existing), n_pairs):
                completeness_clean = float((M_clean[i].sum(1) - f_clean[i].cpu()).abs().max())
                completeness_adv   = float((M_adv[i].sum(1)   - f_adv[i].cpu()).abs().max())
                existing.append({
                    "pair_idx": start + i,
                    "d_f": float((f_clean[i] - f_adv[i]).norm(p=2)),
                    "d_h": float((h_clean[i] - h_adv[i]).norm(p=2)),
                    "d_M": float((M_clean[i] - M_adv[i]).norm(p='fro')),
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
                    "n_pairs": n_pairs,
                    "accumulators": accumulators,
                })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--total_pairs", type=int, default=5000)
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "cw", "deepfool", "apgd", "square"])
    parser.add_argument("--out_dir", default="results/phase1/s3")
    parser.add_argument("--pairs_root", default="experiments")
    args = parser.parse_args()

    run_chunk(args.chunk_id, args.num_chunks, args.total_pairs, args.archs,
              args.attacks, args.out_dir, args.pairs_root)


if __name__ == "__main__":
    main()
