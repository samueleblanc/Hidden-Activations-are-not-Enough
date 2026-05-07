"""Pillar 3 (May 2026): Cross-model representation comparison.

Tests the conjecture that knowledge matrices admit basis-free per-sample
comparison across DIFFERENTLY-TRAINED same-architecture checkpoints, while
penultimate features require learned alignment (CKA / Git Re-Basin) to be
even defined.

Scope: resnet152 (k=5) + densenet121 (k=2). GoogLeNet excluded (only k=1
public checkpoint, declined to self-train). All checkpoints are
DIFFERENTLY-TRAINED (different recipes, not same-recipe seed variants);
this is a deliberate framing choice — see docs/Final-twist/paper-plan.md
§1.1 and km-notes.md 2026-05-02.

Per-pair execution model: each invocation processes ONE (i, j) pair and
writes a per-pair JSON to results/cross_model/<arch>/per_pair/<i>__<j>.json.
SLURM array dispatch then issues k(k-1)/2 tasks per arch.

Usage:
    # Verify a checkpoint loads + KM completeness check passes
    python cross_model_experiment.py --arch resnet152 --verify tv_v1

    # Run one pair (e.g., array index)
    python cross_model_experiment.py --arch resnet152 \
        --ckpt-i tv_v1 --ckpt-j timm_a1 \
        --num-samples 1000 \
        --imagenet-root /datashare/imagenet/ILSVRC2012

    # Smoke test (one pair, 5 samples, laptop-fast)
    python cross_model_experiment.py --arch resnet152 \
        --ckpt-i tv_v1 --ckpt-j tv_v2 --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger("cross_model")


# =====================================================================
# Checkpoint registry — declarative source-of-truth per (arch, alias).
# Each entry knows how to build the underlying torchvision/timm Module
# and provides its state-dict for positional remap into the KM wrapper.
# =====================================================================

def _tv_resnet152(weights_str: str):
    import torchvision.models as tv
    return tv.resnet152(weights=weights_str)


def _tv_densenet121(weights_str: str):
    import torchvision.models as tv
    return tv.densenet121(weights=weights_str)


def _timm_create(tag: str):
    import timm
    return timm.create_model(tag, pretrained=True)


CHECKPOINT_REGISTRY: Dict[str, Dict[str, Tuple[Callable[[], nn.Module], str]]] = {
    "resnet152": {
        # tv_v1 = ResNet152_Weights.IMAGENET1K_V1 (He 2015 recipe)
        "tv_v1":  (lambda: _tv_resnet152("IMAGENET1K_V1"),
                   "He et al. 2015 (SGD + step LR + classic aug)"),
        # tv_v2 = ResNet152_Weights.IMAGENET1K_V2 (modern recipe)
        "tv_v2":  (lambda: _tv_resnet152("IMAGENET1K_V2"),
                   "torchvision V2 (FixRes + long schedule + label smoothing)"),
        # timm RSB recipes (Wightman 2021)
        "timm_a1": (lambda: _timm_create("resnet152.a1_in1k"),
                    "RSB A1 (LAMB + BCE + RandAug)"),
        "timm_a2": (lambda: _timm_create("resnet152.a2_in1k"),
                    "RSB A2 (different LR/epoch budget)"),
        "timm_a3": (lambda: _timm_create("resnet152.a3_in1k"),
                    "RSB A3 (160px lower-budget)"),
    },
    "densenet121": {
        "tv_v1":  (lambda: _tv_densenet121("IMAGENET1K_V1"),
                   "Huang et al. 2017 original"),
        "timm_ra": (lambda: _timm_create("densenet121.ra_in1k"),
                    "timm RSB-style augmentation recipe"),
    },
}


def list_checkpoints(arch: str) -> List[str]:
    if arch not in CHECKPOINT_REGISTRY:
        raise ValueError(f"Unknown arch {arch!r}; known: {sorted(CHECKPOINT_REGISTRY)}")
    return list(CHECKPOINT_REGISTRY[arch].keys())


# =====================================================================
# KM-wrapper loading (alternate weights) + completeness check
# =====================================================================

def _stratified_remap(arch: str, ckpt_alias: str,
                      km_sd: Dict[str, torch.Tensor],
                      src_sd: Dict[str, torch.Tensor]
                      ) -> Dict[str, torch.Tensor]:
    """Stratified positional remap: KM stores skip projections in a separate
    ``residual_modules`` ModuleList while torchvision/timm interleave them
    inline as ``layerN.M.downsample.*``. Splitting both state-dicts into
    main + residual strata and zipping each stratum positionally aligns the
    parameters correctly, where a flat zip mis-aligns at every stage
    transition.
    """
    km_main = {k: v for k, v in km_sd.items()
               if not k.startswith("residual_modules.")}
    km_resid = {k: v for k, v in km_sd.items()
                if k.startswith("residual_modules.")}
    src_main = {k: v for k, v in src_sd.items() if "downsample" not in k}
    src_resid = {k: v for k, v in src_sd.items() if "downsample" in k}

    if len(km_main) != len(src_main) or len(km_resid) != len(src_resid):
        raise RuntimeError(
            f"Stratum cardinality mismatch for {arch}/{ckpt_alias}: "
            f"main km={len(km_main)} src={len(src_main)}; "
            f"residual km={len(km_resid)} src={len(src_resid)}. "
            f"May indicate a sub-architecture mismatch (extra BNs, different "
            f"skip-projection convention) — investigate before forcing."
        )

    remapped: Dict[str, torch.Tensor] = {}
    for stratum, km_keys, src_keys in (
        ("main", km_main, src_main),
        ("residual", km_resid, src_resid),
    ):
        for (mk, mv), (sk, sv) in zip(km_keys.items(), src_keys.items()):
            if mv.shape != sv.shape:
                raise RuntimeError(
                    f"Shape mismatch in {stratum} stratum during remap of "
                    f"{arch}/{ckpt_alias}: KM[{mk}]={tuple(mv.shape)} "
                    f"vs src[{sk}]={tuple(sv.shape)}"
                )
            remapped[mk] = sv
    return remapped


def build_km_model_with_alt_weights(arch: str, ckpt_alias: str,
                                    device: str) -> nn.Module:
    """Build the KM-wrapper model and load ALTERNATE pretrained weights.

    Process:
      1. Build the KM-wrapper (loads its default-pretrained weights, usually
         torchvision V1).
      2. Build the source model (torchvision/timm) per the registry.
      3. Stratified remap (see ``_stratified_remap``): main params zip with
         main src params; residual-projection params zip with src downsample
         params. This correctly handles ResNets, where KM stores residual
         projections in a separate ``residual_modules`` ModuleList while
         torchvision/timm interleave them inline.
      4. load_state_dict(remapped, strict=True) — abort on mismatch.
      5. Caller must verify M(x).sum(1) == f(x) before trusting.

    Raises:
        RuntimeError if the remap can't align shapes (signals a real
        architectural mismatch worth investigating, not silent corruption).
    """
    # Lazy imports so module loads on machines without knowledgematrix.
    if arch == "resnet152":
        from utils.km_models import _build_resnet152 as _km_build
    elif arch == "densenet121":
        from utils.km_models import _build_densenet121 as _km_build
    else:
        raise ValueError(f"No KM wrapper factory wired for arch {arch!r}")

    if ckpt_alias not in CHECKPOINT_REGISTRY[arch]:
        raise ValueError(
            f"Unknown checkpoint {ckpt_alias!r} for {arch}; "
            f"known: {list_checkpoints(arch)}"
        )

    logger.info("Building KM wrapper for %s (default weights)…", arch)
    km_model = _km_build(device)
    km_model.eval()

    factory, recipe_desc = CHECKPOINT_REGISTRY[arch][ckpt_alias]
    logger.info("Loading alternate source: %s (%s)", ckpt_alias, recipe_desc)
    src_model = factory().eval()

    km_sd = km_model.state_dict()
    src_sd = src_model.state_dict()
    remapped = _stratified_remap(arch, ckpt_alias, km_sd, src_sd)

    missing, unexpected = km_model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Load_state_dict had missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )

    # KM library's residual storage is bifurcated: state_dict reads
    # ``residual_modules`` (auto-created projections), but forward uses
    # ``residuals`` (a plain dict referencing the actual downsample
    # submodules — swapped in at resnet152.py:148-152). load_state_dict
    # updates the former; copy those weights into the latter so forward
    # actually sees the loaded src weights.
    rm_iter = iter(getattr(km_model, "residual_modules", []))
    for entries in getattr(km_model, "residuals", {}).values():
        for _start, projection in entries:
            for sub in projection:
                if isinstance(sub, nn.Identity):
                    continue
                rm_module = next(rm_iter)
                sub.load_state_dict(rm_module.state_dict())
                sub.to(device)

    del src_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return km_model


def verify_km_completeness(km_model: nn.Module, device: str,
                           atol: float = 1e-2) -> Tuple[bool, float]:
    """Verify M(x).sum(1) ≈ f(x) on one random sample.

    Returns (passed, max_abs_diff). Mirrors validate_theorem45.py:_verify_*.
    Looser atol (1e-2) than the sacred 1e-6 because cross-model porting
    introduces float-cast noise; if the diff exceeds 1e-2 the remap is
    structurally wrong, not just precision-bound.
    """
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

    torch.manual_seed(0)
    x = torch.randn(3, 224, 224, device=device)
    with torch.no_grad():
        # Logits via the wrapper's standard forward (model expects 4D)
        logits = km_model(x.unsqueeze(0)).squeeze(0)  # (1000,)
        # KM via the computer (expects 3D)
        mc = KnowledgeMatrixComputer(km_model, batch_size=512, device=device)
        M = mc.forward(x)  # (1000, d+1)
        row_sum = M.sum(dim=1)

    diff = float((logits - row_sum).abs().max().item())
    return diff < atol, diff


def verify_remap_logit_match(arch: str, ckpt_alias: str, device: str,
                             atol: float = 1e-3) -> Tuple[bool, float]:
    """Verify the remap is correct by comparing KM-wrapper logits to the
    fresh source model's logits on the same input.

    Bit-exact match (diff == 0) means every parameter, including the
    forward-path residual projections, was correctly remapped. This is a
    *stronger* check than `M(x).sum(1) ≈ f(x)` for the purpose of
    confirming the remap, and uses far less RAM (no Jacobian materialized)
    — usable on memory-constrained login nodes where the full KM compute
    OOMs.
    """
    factory, _ = CHECKPOINT_REGISTRY[arch][ckpt_alias]
    km_model = build_km_model_with_alt_weights(arch, ckpt_alias, device)
    src_model = factory().eval().to(device)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        km_logits = km_model(x)
        src_logits = src_model(x)
    diff = float((km_logits - src_logits).abs().max().item())

    del src_model, km_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return diff < atol, diff


# =====================================================================
# Per-pair distance computation (streaming)
# =====================================================================

def compute_penultimate_extractor(arch: str, model: nn.Module
                                  ) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a closure that extracts penultimate features for a model.

    Uses a forward hook on the architecture-appropriate layer. Mirrors
    teleportation_experiment.py:PenultimateExtractor but architecture-
    dispatched here.
    """
    cache: Dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, out):
        cache["feat"] = out.detach()

    # KM wrapper exposes layers via dotted access — walk to the right node.
    # For resnet152: avgpool output (B, 2048). For densenet121: features.norm5
    # post-ReLU+pool (B, 1024). Library wrapper structure may differ; we
    # introspect rather than hard-code paths.
    if arch == "resnet152":
        # Look for an avgpool / final pooling layer
        target = None
        target_name = None
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                target = mod
                target_name = name
        if target is None:
            raise RuntimeError("No pooling layer found in resnet152 wrapper")
    elif arch == "densenet121":
        # Find the last BatchNorm before classifier (densenet has features.norm5)
        target = None
        target_name = None
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                target = mod
                target_name = name
        if target is None:
            raise RuntimeError("No BN/pooling layer found in densenet121 wrapper")
    else:
        raise ValueError(f"No penultimate extractor for arch {arch!r}")

    handle = target.register_forward_hook(hook)
    logger.info("  penultimate hook on %s (%s)", target_name, type(target).__name__)

    def extract(x_batch: torch.Tensor) -> torch.Tensor:
        cache.clear()
        with torch.no_grad():
            _ = model(x_batch)
        feat = cache["feat"]
        # Flatten to (B, D)
        return feat.flatten(start_dim=1)

    extract.handle = handle  # so caller can remove
    return extract


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment (Kornblith et al. 2019).

    Same as teleportation_experiment.py:linear_cka — replicated here to
    keep this module self-contained.
    """
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    cross = Xc.T @ Yc
    num = (cross * cross).sum()
    den = ((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()).clamp_min(1e-30)
    return float((num / den).item())


def run_pair(arch: str, ckpt_i: str, ckpt_j: str,
             dataloader: DataLoader, device: str,
             matrix_batch_size: int = 1024) -> Dict:
    """Compute per-sample d_KM, d_h, d_logit + population CKA for a pair.

    Streaming: only model_i and model_j are in memory at once. KMs for
    each sample are materialized briefly, distances streamed, then freed.
    """
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

    logger.info("=== Pair: %s vs %s ===", ckpt_i, ckpt_j)

    # Build BOTH KM models (each carries its alternate weights).
    model_i = build_km_model_with_alt_weights(arch, ckpt_i, device)
    ok_i, diff_i = verify_km_completeness(model_i, device)
    if not ok_i:
        raise RuntimeError(
            f"KM completeness failed for {arch}/{ckpt_i}: "
            f"max|M.sum(1) - f(x)| = {diff_i:.3e}"
        )
    logger.info("  %s completeness ok (max diff = %.3e)", ckpt_i, diff_i)

    model_j = build_km_model_with_alt_weights(arch, ckpt_j, device)
    ok_j, diff_j = verify_km_completeness(model_j, device)
    if not ok_j:
        raise RuntimeError(
            f"KM completeness failed for {arch}/{ckpt_j}: "
            f"max|M.sum(1) - f(x)| = {diff_j:.3e}"
        )
    logger.info("  %s completeness ok (max diff = %.3e)", ckpt_j, diff_j)

    extract_i = compute_penultimate_extractor(arch, model_i)
    extract_j = compute_penultimate_extractor(arch, model_j)
    mc_i = KnowledgeMatrixComputer(model_i, batch_size=matrix_batch_size, device=device)
    mc_j = KnowledgeMatrixComputer(model_j, batch_size=matrix_batch_size, device=device)

    d_KM, d_h, d_logit = [], [], []
    h_i_all, h_j_all = [], []  # population CKA needs the full feature matrices

    t0 = time.perf_counter()
    n_processed = 0
    for batch_idx, (xb, _yb) in enumerate(dataloader):
        xb = xb.to(device).float()

        with torch.no_grad():
            f_i = model_i(xb)  # (B, 1000)
            f_j = model_j(xb)
            h_i = extract_i(xb)
            h_j = extract_j(xb)

        # Per-sample distances (penultimate + logit)
        d_h_b = torch.linalg.norm(h_i - h_j, dim=1).cpu().numpy()
        d_f_b = torch.linalg.norm(f_i - f_j, dim=1).cpu().numpy()
        d_h.extend(d_h_b.tolist())
        d_logit.extend(d_f_b.tolist())

        # Per-sample KM distance (streamed, never holds all KMs at once)
        for k in range(xb.shape[0]):
            x_one = xb[k]  # (3, 224, 224)
            with torch.no_grad():
                M_i = mc_i.forward(x_one).to(torch.float64)
                M_j = mc_j.forward(x_one).to(torch.float64)
                d_KM_one = float(torch.linalg.norm(M_i - M_j).item())
            d_KM.append(d_KM_one)
            del M_i, M_j

        # Stash penultimate features for population CKA
        h_i_all.append(h_i.cpu())
        h_j_all.append(h_j.cpu())

        n_processed += xb.shape[0]
        if batch_idx % 5 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  [%d samples] elapsed=%.1fs", n_processed, elapsed)

    extract_i.handle.remove()
    extract_j.handle.remove()
    del model_i, model_j, mc_i, mc_j
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Population CKA on the stashed features
    H_i = torch.cat(h_i_all, dim=0).double()
    H_j = torch.cat(h_j_all, dim=0).double()
    cka_lin = linear_cka(H_i, H_j)

    # Aggregates
    d_KM_arr = np.array(d_KM)
    d_h_arr = np.array(d_h)
    d_logit_arr = np.array(d_logit)

    # Theorem 4.5 transferred cross-model: ||M_i(x) - M_j(x)|| >= gamma_ij * ||f_i(x) - f_j(x)||
    # gamma_ij = min over samples of (d_KM / d_logit), excluding zero d_logit.
    valid = d_logit_arr > 1e-9
    if valid.sum() > 0:
        ratio = d_KM_arr[valid] / d_logit_arr[valid]
        gamma_cross = float(ratio.min())
        # Bootstrap 95% CI on gamma
        rng = np.random.default_rng(42)
        boot = [float(ratio[rng.integers(0, len(ratio), len(ratio))].min())
                for _ in range(1000)]
        gamma_ci = [float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5))]
    else:
        gamma_cross = float("nan")
        gamma_ci = [float("nan"), float("nan")]

    elapsed_total = time.perf_counter() - t0
    logger.info("Pair done in %.1fs: gamma_cross = %.4f  CKA = %.4f",
                elapsed_total, gamma_cross, cka_lin)

    return {
        "arch": arch,
        "ckpt_i": ckpt_i,
        "ckpt_j": ckpt_j,
        "ckpt_i_recipe": CHECKPOINT_REGISTRY[arch][ckpt_i][1],
        "ckpt_j_recipe": CHECKPOINT_REGISTRY[arch][ckpt_j][1],
        "n_samples": n_processed,
        "matrix_batch_size": matrix_batch_size,
        "elapsed_seconds": elapsed_total,
        "completeness": {ckpt_i: diff_i, ckpt_j: diff_j},
        "per_sample": {
            "d_KM":    d_KM,
            "d_h":     d_h,
            "d_logit": d_logit,
        },
        "aggregate": {
            "d_KM_mean":    float(d_KM_arr.mean()),
            "d_KM_std":     float(d_KM_arr.std()),
            "d_h_mean":     float(d_h_arr.mean()),
            "d_h_std":      float(d_h_arr.std()),
            "d_logit_mean": float(d_logit_arr.mean()),
            "d_logit_std":  float(d_logit_arr.std()),
            "cka_linear":   cka_lin,
            "cka_distance": 1.0 - cka_lin,
            "gamma_cross":  gamma_cross,
            "gamma_cross_ci_95": gamma_ci,
        },
    }


# =====================================================================
# Dataset loading (ImageNet val)
# =====================================================================

def load_imagenet_val(imagenet_root: str, num_samples: int,
                      batch_size: int = 32, seed: int = 42) -> DataLoader:
    """Stratified subset of ImageNet val (one sample per class up to num_samples)."""
    from utils.utils import get_imagenet_val_dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    _, ds = get_imagenet_val_dataset(data_path=imagenet_root,
                                      transform=transform)

    # Stratify: for num_samples in [1, 1000], take first sample of each
    # consecutive class (val is sorted alphabetically, labels are consecutive
    # 50/class, so taking every 50th from the start gives one per class).
    g = torch.Generator().manual_seed(seed)
    if num_samples <= 1000:
        # Take one per class (50 images per class in val; first 1000 indices
        # cover classes 0–19 only since val is sorted by label, so use stride).
        stride = 50  # 50 val images per class -> stride to span all 1000
        class_indices = list(range(0, len(ds), stride))[:num_samples]
        indices = class_indices
    else:
        indices = torch.randperm(len(ds), generator=g)[:num_samples].tolist()

    return DataLoader(Subset(ds, indices), batch_size=batch_size,
                      shuffle=False, num_workers=0)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True,
                   choices=sorted(CHECKPOINT_REGISTRY.keys()))
    p.add_argument("--verify", default=None,
                   help="Just load this checkpoint alias and run KM completeness "
                        "check; exit 0 if pass, exit 1 if fail. Skips pair compute.")
    p.add_argument("--ckpt-i", default=None,
                   help="Alias of first checkpoint in the pair (see registry).")
    p.add_argument("--ckpt-j", default=None,
                   help="Alias of second checkpoint in the pair.")
    p.add_argument("--num-samples", type=int, default=1000,
                   help="ImageNet val samples per pair (stratified one-per-class up to 1000).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Forward batch size for penultimate/logit extraction.")
    p.add_argument("--matrix-batch-size", type=int, default=1024,
                   help="KnowledgeMatrixComputer batch size.")
    p.add_argument("--output-dir", default="results/cross_model")
    p.add_argument("--imagenet-root",
                   default="/datashare/imagenet/ILSVRC2012")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 5 samples, batch=2.")
    p.add_argument("--list-checkpoints", action="store_true",
                   help="List available checkpoints for --arch and exit.")
    args = p.parse_args()
    if args.smoke:
        args.num_samples = 5
        args.batch_size = 2
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")
    args = parse_args()

    if args.list_checkpoints:
        for alias in list_checkpoints(args.arch):
            _, recipe = CHECKPOINT_REGISTRY[args.arch][alias]
            print(f"  {alias:<10s}  {recipe}")
        return 0

    if args.verify is not None:
        # Lightweight remap-correctness check (no Jacobian materialized).
        # Login-node-safe: ~2 GB peak vs the full KM compute's 8+ GB.
        try:
            ok, diff = verify_remap_logit_match(args.arch, args.verify, args.device)
            if ok:
                print(f"PASS: {args.arch}/{args.verify}  max|km - src logits| = {diff:.3e}")
                return 0
            print(f"FAIL: {args.arch}/{args.verify}  max|km - src logits| = {diff:.3e}")
            return 1
        except Exception as e:
            print(f"ERROR loading {args.arch}/{args.verify}: {e}")
            return 2

    if args.ckpt_i is None or args.ckpt_j is None:
        print("ERROR: must specify --ckpt-i and --ckpt-j (or --verify, or --list-checkpoints).",
              file=sys.stderr)
        return 2
    if args.ckpt_i == args.ckpt_j:
        print("ERROR: --ckpt-i and --ckpt-j must differ", file=sys.stderr)
        return 2

    dataloader = load_imagenet_val(args.imagenet_root, args.num_samples,
                                    batch_size=args.batch_size)
    result = run_pair(args.arch, args.ckpt_i, args.ckpt_j, dataloader,
                      args.device, matrix_batch_size=args.matrix_batch_size)

    out_dir = Path(args.output_dir) / args.arch / "per_pair"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.ckpt_i}__{args.ckpt_j}.json"
    # Atomic write
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    tmp_path.rename(out_path)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
