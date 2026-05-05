"""S2 worker: cross-architecture KM Frobenius + measure-panel comparison.

Per chunk: load chunk slice of ImageNet val; forward through all 3 archs;
for each pair (RN-DN, RN-GN, DN-GN) emit:
  - per-sample KM Frobenius distance list (append-and-overwrite checkpointing)
  - D1 measure-panel accumulators on natively-cross-dim measures (penultimate)
  - D2 measure-panel accumulators on PCA-projected (to 1024) penultimate
"""
import json
import os
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import torch

from utils.utils import get_device
from utils.atomic_io import atomic_torch_save, atomic_json_dump, atomic_json_load
from cka_similarity.workers.common import chunk_slice, load_active_km_batch_size, calibration_path_for
from cka_similarity.measures.panel import PANEL
from cka_similarity.workers.s1_within_arch_invariance import (
    load_pretrained, forward_penultimate, forward_logits, load_imagenet_val_chunk,
)


def extract_km_per_sample(model, x_batch, batch_size: int):
    """Extract knowledge matrices for the chunk's input batch.

    Output shape: (n, 1000, d+1) where d = 3*224*224.

    Per CLAUDE.md "Critical Patterns": KnowledgeMatrixComputer.forward expects
    3D input (C, H, W) — never 4D — so iterate over the batch and feed
    each x[i] without unsqueeze.
    """
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
    device = next(model.parameters()).device
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)

    out = []
    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].to(device)            # (3, 224, 224) — note: no unsqueeze
        M_i = mc.forward(x_i)                  # KM expects 3D input per CLAUDE.md
        out.append(M_i.detach().cpu().unsqueeze(0))
    return torch.cat(out, dim=0)


def pca_project(X: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project (n, p) to (n, target_dim) via PCA on X."""
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:target_dim].T


def run_chunk(chunk_id, num_chunks, num_samples_total, archs, out_dir, data_dir,
              calib_dir="experiments/calibration"):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    start, end = chunk_slice(chunk_id, num_samples_total, num_chunks)
    inputs = load_imagenet_val_chunk(start, end, data_dir).to(device)

    # Forward through all 3 arches; KM extraction uses arch-specific batch size
    h = {}
    logits = {}
    M = {}
    for arch in archs:
        model = load_pretrained(arch).to(device)
        h[arch] = forward_penultimate(model, inputs).cpu()
        logits[arch] = forward_logits(model, inputs).cpu()
        bs = load_active_km_batch_size(calibration_path_for(arch))
        M[arch] = extract_km_per_sample(model, inputs, batch_size=bs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pairwise comparisons
    panel_cls = list(PANEL)
    short_names = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}
    pair_names = {(a, b): f"{short_names[a]}_{short_names[b]}" for a, b in combinations(archs, 2)}

    for (a, b), pname in pair_names.items():
        # --- Per-sample KM Frobenius distance with append-and-overwrite ---
        km_dist_path = Path(out_dir) / f"{pname}_KM_chunk{chunk_id}.json"
        km_distances = atomic_json_load(str(km_dist_path), default=[])
        start_local = len(km_distances)
        for i in range(start_local, M[a].shape[0]):
            d = float((M[a][i] - M[b][i]).norm(p='fro'))
            km_distances.append(d)
            atomic_json_dump(str(km_dist_path), km_distances)

        # --- D1 (natively-cross-dim) measure panel on penultimate ---
        d1_path = Path(out_dir) / f"{pname}_D1_chunk{chunk_id}.pt"
        if not d1_path.exists():
            d1_acc = {}
            for cls in panel_cls:
                m = cls()
                if not m.cross_dim_native:
                    continue   # skip dim-restricted measures for D1
                if cls.__name__ == "OutputJSD":
                    d1_acc[m.name] = m.accumulate(logits[a], logits[b])
                else:
                    d1_acc[m.name] = m.accumulate(h[a], h[b])
            atomic_torch_save(str(d1_path), {
                "chunk_id": chunk_id, "pair": pname,
                "n_samples": h[a].shape[0],
                "accumulators": d1_acc,
                "convention": "D1",
            })

        # --- D2 (PCA-padded to 1024) measure panel on penultimate ---
        d2_path = Path(out_dir) / f"{pname}_D2_chunk{chunk_id}.pt"
        if not d2_path.exists():
            target_dim = min(h[a].shape[1], h[b].shape[1])  # 1024 for our archs
            h_a_proj = pca_project(h[a], target_dim)
            h_b_proj = pca_project(h[b], target_dim)
            d2_acc = {}
            for cls in panel_cls:
                m = cls()
                if m.cross_dim_native:
                    continue   # D2 covers the dim-restricted measures only
                if cls.__name__ == "OutputJSD":
                    continue   # already in D1
                d2_acc[m.name] = m.accumulate(h_a_proj, h_b_proj)
            atomic_torch_save(str(d2_path), {
                "chunk_id": chunk_id, "pair": pname,
                "n_samples": h_a_proj.shape[0],
                "accumulators": d2_acc,
                "convention": "D2", "target_dim": target_dim,
            })


def main():
    parser = ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=25000)
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--out_dir", default="results/phase1/s2")
    parser.add_argument("--data_dir", default="/datashare/imagenet/ILSVRC2012")
    args = parser.parse_args()

    run_chunk(args.chunk_id, args.num_chunks, args.num_samples, args.archs,
              args.out_dir, args.data_dir)


if __name__ == "__main__":
    main()
