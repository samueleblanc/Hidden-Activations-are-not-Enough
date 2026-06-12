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
from utils.km_models import build_model
from utils.atomic_io import atomic_torch_save, atomic_json_dump, atomic_json_load
from utils.scaling import IMAGENET_INPUT_NUMEL, IMAGENET_NUM_CLASSES, km_numel
from cka_similarity.workers.common import chunk_slice, load_active_km_batch_size, calibration_path_for
from cka_similarity.measures.panel import PANEL
from cka_similarity.workers.s1_within_arch_invariance import load_imagenet_val_chunk


def load_pretrained(arch: str):
    """Load the knowledgematrix-wrapped pretrained ImageNet network.

    S2/S3 both feed their model into ``KnowledgeMatrixComputer``, which reads
    ``model.layers`` and ``model.input_shape``; raw torchvision models lack
    both, so the wrapper from ``utils.km_models.build_model`` is required.

    Note this is a different ``load_pretrained`` from S1's: S1 needs a
    COB-aware model for neural teleportation, S2/S3 need the wrapper for KM
    extraction. They cannot share a single helper.
    """
    return build_model(arch, get_device())


def forward_penultimate(model, x):
    """Penultimate-layer features (input to the final classifier).

    The knowledgematrix wrapper's ``forward`` accepts a ``return_penultimate``
    kwarg (see knowledgematrix/neural_net.py:523-536) which short-circuits
    ``self.layers[:-1]`` — exactly the input to the final Linear. This is a
    cleaner contract than the forward-hook pattern used in S1 (where the
    classifier is exposed as ``model.fc`` / ``model.classifier`` on COB
    models) because the wrapper's classifier is anonymous (just
    ``model.layers[-1]``).
    """
    with torch.no_grad():
        return model(x, return_penultimate=True)


def forward_logits(model, x):
    with torch.no_grad():
        return model(x)


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

    def _forward_with_oom_backoff(x_i, bs):
        """Compute one KM, halving the KMC batch_size on CUDA OOM.

        On the packed ``gpubase_bygpu`` partition an S3 task can be co-located
        on a physical GPU already holding another large KM process (observed
        2026-06-06: a peer held 72.8 GiB, leaving 0.7 GiB free → instant OOM).
        ``batch_size`` only tiles the 150529 input positions through the
        forward pass, so shrinking it lowers the transient activation peak
        until it fits the residual free memory **without changing M(x) at
        all** (the completeness residual stays ≤1e-6). Floor at 1. Returns the
        matrix and the batch_size that succeeded so the caller can keep using
        the reduced size for the remaining samples.
        """
        while True:
            try:
                mc = KnowledgeMatrixComputer(model, batch_size=bs, device=device)
                return mc.forward(x_i), bs
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                print(f"WARNING: CUDA OOM in KM extract; retrying at batch_size={bs}",
                      flush=True)

    out = []
    cur_bs = batch_size
    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].to(device)            # (3, 224, 224) — note: no unsqueeze
        M_i, cur_bs = _forward_with_oom_backoff(x_i, cur_bs)   # keep reduced bs
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


def _write_complete_sentinel_if_done(out_dir, archs, num_chunks):
    """Touch ``{out_dir}/.complete`` once every per-chunk file has landed.

    For each cross-arch pair this worker emits three files per chunk:
    ``{pname}_KM_chunk{i}.json``, ``{pname}_D1_chunk{i}.pt``,
    ``{pname}_D2_chunk{i}.pt``. We probe the full grid here so whichever
    chunk-task happens to finish last detects the all-done state and marks
    the sentinel for run_pipeline.sh. ``Path.touch()`` is idempotent — a
    TOCTOU race between two simultaneously-finishing tasks is benign.
    """
    out_path = Path(out_dir)
    short_names = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}
    pair_names = [f"{short_names[a]}_{short_names[b]}"
                  for a, b in combinations(archs, 2)]
    for pname in pair_names:
        for ci in range(num_chunks):
            for kind in ("KM_chunk{}.json", "D1_chunk{}.pt", "D2_chunk{}.pt"):
                expected = out_path / f"{pname}_{kind.format(ci)}"
                if not expected.exists():
                    return
    (out_path / ".complete").touch()


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

    # KM-space scale factor used to convert raw Frobenius to RMS-per-coord.
    # See utils/scaling.py and docs/Final-twist/km-notes.md (2026-05-10).
    s_KM = 1.0 / (km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL) ** 0.5)

    for (a, b), pname in pair_names.items():
        # --- Per-sample KM Frobenius distance (streaming, append-and-overwrite) ---
        # Raw distances are saved to {pname}_KM_chunk{i}.json (preserves the
        # original on-disk schema); RMS-scaled values are saved alongside to
        # {pname}_KM_rms_chunk{i}.json. Reducers can choose which to consume.
        km_dist_path = Path(out_dir) / f"{pname}_KM_chunk{chunk_id}.json"
        km_dist_rms_path = Path(out_dir) / f"{pname}_KM_rms_chunk{chunk_id}.json"
        km_distances = atomic_json_load(str(km_dist_path), default=[])
        start_local = len(km_distances)

        if start_local < inputs.shape[0]:
            # Lazy import: keep the import scoped to where KMs are actually
            # computed so the smoke test can monkey-patch it before run_chunk.
            from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

            # Calibration is per-arch; if it's missing for either arch in this
            # pair, skip the pair rather than crashing the whole chunk task.
            # Mirrors the robustness pattern in s3_distance_amplification.
            try:
                bs_a_full = load_active_km_batch_size(calibration_path_for(a))
                bs_b_full = load_active_km_batch_size(calibration_path_for(b))
            except FileNotFoundError as e:
                print(
                    f"WARNING: calibration missing for pair ({a}, {b}) — {e}; "
                    f"skipping this pair in chunk {chunk_id}",
                    flush=True,
                )
                continue

            # Calibration's active tier (93%) is sized for ONE model on the GPU.
            # S2 holds TWO archs' KnowledgeMatrixComputers concurrently, so each
            # must yield half the GPU. Halving the per-arch bs caps each KMC's
            # working set at ~half its calibrated peak; the two together then
            # fit. min floor of 64 prevents pathological tiny bs from breaking
            # KMC's batched compute on archs with very small calibrated bs.
            bs_a = max(64, bs_a_full // 2)
            bs_b = max(64, bs_b_full // 2)
            print(
                f"S2 cross-arch concurrent-model bs cap: ({a}) {bs_a_full}->{bs_a}, "
                f"({b}) {bs_b_full}->{bs_b}",
                flush=True,
            )

            model_a = load_pretrained(a).to(device)
            mc_a = KnowledgeMatrixComputer(model_a, batch_size=bs_a, device=device)

            model_b = load_pretrained(b).to(device)
            mc_b = KnowledgeMatrixComputer(model_b, batch_size=bs_b, device=device)

            for i in range(start_local, inputs.shape[0]):
                x_i = inputs[i]                       # (3, 224, 224) — no unsqueeze
                with torch.no_grad():
                    M_a_i = mc_a.forward(x_i)         # (1000, d+1) on GPU
                    M_b_i = mc_b.forward(x_i)
                    # Cross-arch KM Frobenius distance is only meaningful when
                    # the two matrices share shape — i.e. both archs produce
                    # the canonical 1000 × 150529 KM. utils.km_models.build_model
                    # holds this for the three Phase-1 archs; assert here so a
                    # future arch that breaks the shape contract fails loudly
                    # rather than silently broadcasting through ``norm``.
                    assert M_a_i.shape == M_b_i.shape, (
                        f"Cross-arch KM shape mismatch on sample {i}: "
                        f"{a}={tuple(M_a_i.shape)} vs {b}={tuple(M_b_i.shape)}"
                    )
                    d = float((M_a_i - M_b_i).norm(p='fro').item())
                km_distances.append(d)
                atomic_json_dump(str(km_dist_path), km_distances)
                # Mirror the raw list onto disk in RMS-per-coordinate units
                # for downstream consumers; kept in lock-step with the raw
                # file (overwrite both on every appended pair).
                atomic_json_dump(
                    str(km_dist_rms_path),
                    [v * s_KM for v in km_distances],
                )
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

    # Once this chunk's outputs land, check whether the full grid is on disk
    # and, if so, mark the worker complete for run_pipeline.sh.
    _write_complete_sentinel_if_done(out_dir, archs, num_chunks)


def main():
    parser = ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=25000)
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--out_dir", default="results/phase1/s2")
    parser.add_argument("--data_dir",
                        default=os.environ.get("IMAGENET_ROOT", "/datashare/imagenet/ILSVRC2012"))
    args = parser.parse_args()

    run_chunk(args.chunk_id, args.num_chunks, args.num_samples, args.archs,
              args.out_dir, args.data_dir)


if __name__ == "__main__":
    main()
