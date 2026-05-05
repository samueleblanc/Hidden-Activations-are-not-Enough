"""S2 worker: cross-architecture KM Frobenius + measure-panel comparison.

Per chunk: load chunk slice of ImageNet val; forward through all 3 archs;
for each pair (RN-DN, RN-GN, DN-GN) emit:
  - per-sample KM Frobenius distance list (append-and-overwrite checkpointing)
  - D1 measure-panel accumulators on natively-cross-dim measures (penultimate)
  - D2 measure-panel accumulators on PCA-projected (to 1024) penultimate

Memory note (2026-05): the prior implementation eagerly materialized all 3
archs' KMs in CPU memory (≈700 GB peak per chunk for ImageNet) which OOMs the
128 GB SLURM allocation. The current implementation streams KMs per sample —
each sample's M(x) is built on-GPU, immediately consumed for the per-sample
Frobenius distance, then discarded. Penultimate features and logits are still
batch-extracted because they are tiny (≈2048 floats per sample, not 150K).
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

    NOTE: This helper materializes the full (n, 1000, d+1) tensor on CPU and
    is therefore memory-intensive (≈600 MB per arch per sample for ImageNet).
    The S2 ``run_chunk`` no longer calls it — it streams M(x) per sample via
    a directly-allocated KnowledgeMatrixComputer to keep peak CPU memory tiny.
    The helper is kept here for the S3 worker (which retains the batch-level
    M_clean / M_adv tensors at small ``n_pairs`` per chunk) and for tests.
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
    """Project (n, p) to (n, target_dim) via PCA on X.

    Each architecture's penultimate features are projected to its OWN top
    ``target_dim`` principal components, NOT to a joint basis. The D2
    (PCA-padded) cross-arch convention is: both arches end in (n, target_dim)
    space, but each in its own variance-aligned basis. This preserves each
    arch's intrinsic structure rather than forcing a shared basis (per the
    design spec §6.3 D2 paragraph).

    A ``target_dim`` larger than the SVD's rank (i.e. ``min(n, p)``) is
    clamped down with a warning so callers don't have to special-case
    short chunks.
    """
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    if target_dim > Vh.shape[0]:
        print(
            f"WARNING: pca_project target_dim={target_dim} > rank={Vh.shape[0]}; clamping",
            flush=True,
        )
        target_dim = Vh.shape[0]
    return Xc @ Vh[:target_dim].T


def run_chunk(chunk_id, num_chunks, num_samples_total, archs, out_dir, data_dir,
              calib_dir="experiments/calibration"):
    """Streaming variant of S2: KMs are computed per-sample on-GPU, never stacked.

    Phase A (cheap): forward each arch once on the full chunk to extract
    penultimate features and logits, which are tiny enough to keep on CPU.
    Phase B (streaming): for each pair (a, b), reload both arches' KM
    computers and stream M_a_i, M_b_i one sample at a time, computing the
    per-sample Frobenius distance and discarding the matrices immediately.

    The streaming variant trades extra model loads (each arch is loaded once
    in Phase A and once per pair it appears in — 1 + 2 = 3 loads per arch)
    for a >100x reduction in peak memory. With 3 archs at ~600 MB / KM, the
    eager variant peaked around 705 GB per chunk; the streaming variant peaks
    at 2 × 0.6 GB on GPU (which is freed every iteration).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    start, end = chunk_slice(chunk_id, num_samples_total, num_chunks)
    inputs = load_imagenet_val_chunk(start, end, data_dir).to(device)

    # === Phase A: penultimate + logits for all 3 archs (small, batched) ===
    h = {}      # arch -> (n_chunk, D_arch) penultimate features on CPU
    logits = {} # arch -> (n_chunk, 1000) logits on CPU
    for arch in archs:
        model = load_pretrained(arch).to(device)
        h[arch] = forward_penultimate(model, inputs).cpu()
        logits[arch] = forward_logits(model, inputs).cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Phase B: per-pair, stream KMs per-sample to compute Frobenius distance ===
    panel_cls = list(PANEL)
    short_names = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}
    pair_names = {(a, b): f"{short_names[a]}_{short_names[b]}" for a, b in combinations(archs, 2)}

    for (a, b), pname in pair_names.items():
        # --- Per-sample KM Frobenius distance (streaming, append-and-overwrite) ---
        km_dist_path = Path(out_dir) / f"{pname}_KM_chunk{chunk_id}.json"
        km_distances = atomic_json_load(str(km_dist_path), default=[])
        start_local = len(km_distances)

        if start_local < inputs.shape[0]:
            # Lazy import: keep the import scoped to where KMs are actually
            # computed so the smoke test can monkey-patch it before run_chunk.
            from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

            model_a = load_pretrained(a).to(device)
            bs_a = load_active_km_batch_size(calibration_path_for(a))
            mc_a = KnowledgeMatrixComputer(model_a, batch_size=bs_a, device=device)

            model_b = load_pretrained(b).to(device)
            bs_b = load_active_km_batch_size(calibration_path_for(b))
            mc_b = KnowledgeMatrixComputer(model_b, batch_size=bs_b, device=device)

            for i in range(start_local, inputs.shape[0]):
                x_i = inputs[i]                       # (3, 224, 224) — no unsqueeze
                with torch.no_grad():
                    M_a_i = mc_a.forward(x_i)         # (1000, d+1) on GPU
                    M_b_i = mc_b.forward(x_i)
                    d = float((M_a_i - M_b_i).norm(p='fro').item())
                km_distances.append(d)
                atomic_json_dump(str(km_dist_path), km_distances)
                del M_a_i, M_b_i

            del model_a, model_b, mc_a, mc_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # --- D2 (PCA-padded to min(D_a, D_b)) measure panel on penultimate ---
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
