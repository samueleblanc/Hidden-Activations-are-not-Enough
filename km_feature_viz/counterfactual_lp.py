"""Step 05: box-constrained closed-form L1 counterfactual via extract_weff.

The LP is `min ‖δ‖₁ s.t. ⟨W_eff[t] - W_eff[s], δ⟩ ≥ swing` with the
*corrected* per-coord box `δ_i ∈ [-x_pixel_i / std_c, (1 - x_pixel_i) / std_c]`,
which makes `x_pixel + δ_pixel ∈ [0, 1]` element-wise — i.e., `x + δ` is a
valid pixel image after un-normalization.

Solved via a greedy multi-coord saturation (closed form; no LP solver):
sort coordinates by |c_i| descending, saturate each in the helpful direction
until swing is delivered. Provably the L1-min optimum because the cost-per-
unit-swing of each coord is `1 / |c_i|`, independent of the bound (the bound
sets the magnitude of the contribution, not the marginal rate).

Methodology note: the box ensures δ corresponds to a renderable pixel
perturbation (visualizable as an image — the saturated pixels show "where
the model is sensitive"). However, with `region_ok = False` for ~all samples
on ImageNet (Round 7 evidence), the linearization breaks well before x+δ
reaches the LP optimum, so f(x+δ) is governed by a different W_eff than
the one solved against. The δ is a faithful M(x)-direction-with-pixel-box,
not a guaranteed model-flipping perturbation. See in_region() for the per-
sample faithfulness check.
"""
import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from km_feature_viz import paths, state
from km_feature_viz.compute_kms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    load_image,
)
from km_feature_viz.manifest import (
    TIER_A_CLASSES,
    read_manifest,
    sample_key,
)

logger = logging.getLogger(__name__)


def patch_available() -> bool:
    sig = inspect.signature(KnowledgeMatrixComputer.forward)
    return "extract_weff" in sig.parameters


def extract_weff_and_beff(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W_eff of shape (out, in), b_eff of shape (out,))."""
    computer = KnowledgeMatrixComputer(model, batch_size=512)
    A = computer.forward(x)
    b_eff = A[:, -1]
    W_eff = computer.forward(x, extract_weff=True)
    return W_eff, b_eff


def solve_l1_lp(
    W_eff: torch.Tensor,
    b_eff: torch.Tensor,
    x_flat: torch.Tensor,
    out_true: torch.Tensor,
    source: int,
    target: int,
    margin: float,
) -> torch.Tensor:
    """Box-constrained closed-form L1-minimal direction.

    LP:  min ‖δ‖₁  s.t.  ⟨c, δ⟩ ≥ swing,  δ_i ∈ [δ_min_i, δ_max_i]
    with c = W_eff[t] - W_eff[s], swing = margin - (f[t] - f[s]), and per-coord
    bounds derived from `x_pixel + δ_pixel ∈ [0, 1]`:

        δ_max_i = (1 - x_pixel_i) / std_c   (≥ 0 since x_pixel ≤ 1)
        δ_min_i = -x_pixel_i / std_c        (≤ 0 since x_pixel ≥ 0)

    where x_pixel_i = x_normalized_i · std_c + mean_c  (per-channel).

    Greedy closed-form: rank coordinates by |c_i| descending and saturate
    each in the helpful direction (sign(c_i)) until swing is delivered. This
    is provably L1-optimal because the cost-per-unit-swing of every coord is
    `1 / |c_i|` — independent of the bound. Shorter |c_i| ⇒ more L1 cost per
    unit of margin movement, so the top-|c| coords come first.

    Vectorized on GPU: argsort + cumsum + searchsorted. ~ms on a 150k-dim
    constraint vector. Replaces the prior unbounded single-coord closed form,
    which produced ‖δ‖_∞ ≈ 70 (way outside [0,1] pixel space). With the
    corrected box, ‖δ‖_∞ ≤ max_c (1/std_c) ≈ 4.4, and δ corresponds to a
    valid renderable pixel image.
    """
    device = W_eff.device
    dtype = W_eff.dtype
    n = W_eff.shape[1]

    swing = float(margin - (out_true[target] - out_true[source]).item())
    if swing <= 0:
        return torch.zeros(n, dtype=dtype, device=device)

    constraint = W_eff[target] - W_eff[source]                              # (n,)
    abs_c = constraint.abs()

    # Per-coord pixel value: x is CHW-flat (3·H·W,), 50176 coords per channel.
    n_per_channel = n // 3
    if n_per_channel * 3 != n:
        raise RuntimeError(
            f"Expected n divisible by 3 (CHW input); got n={n}. "
            f"This solver assumes 3-channel inputs."
        )
    mean_t = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device)
    std_t = torch.tensor(IMAGENET_STD, dtype=dtype, device=device)
    coord_idx = torch.arange(n, device=device)
    channel = coord_idx // n_per_channel                                    # 0..2
    mean_per_coord = mean_t[channel]
    std_per_coord = std_t[channel]

    x_pixel = x_flat * std_per_coord + mean_per_coord                       # ≈ [0, 1]
    # Clamp pos_max/neg_max ≥ 0 to absorb float-noise where x_pixel slips
    # outside [0, 1] by epsilon (numerical inverse of the Normalize transform).
    pos_max = ((1.0 - x_pixel) / std_per_coord).clamp(min=0)                # +δ cap
    neg_max = (x_pixel / std_per_coord).clamp(min=0)                        # |-δ cap|

    # Helpful displacement magnitude per coord (always ≥ 0):
    helpful_disp = torch.where(constraint >= 0, pos_max, neg_max)
    max_contribution = abs_c * helpful_disp                                 # ≥ 0

    # Sort coords by |c| descending; greedy fill until cumsum reaches swing.
    sorted_abs_c, sorted_idx = abs_c.sort(descending=True)
    sorted_max_contrib = max_contribution[sorted_idx]
    cumsum = sorted_max_contrib.cumsum(dim=0)

    total_possible = float(cumsum[-1].item())
    if total_possible < swing:
        raise RuntimeError(
            f"Box-constrained LP infeasible: max swing achievable in [0,1]-"
            f"pixel box is {total_possible:.3e}, but (margin - logit_gap) "
            f"requires {swing:.3e}. Either decrease --margin, or accept that "
            f"this (source={source}, target={target}) pair cannot be flipped "
            f"within the linearization."
        )

    # First k where cumsum[k] ≥ swing. searchsorted returns the leftmost
    # insertion point i such that cumsum[i-1] < swing ≤ cumsum[i].
    k = int(torch.searchsorted(
        cumsum, torch.tensor(swing, dtype=dtype, device=device)
    ).item())

    delta = torch.zeros(n, dtype=dtype, device=device)

    # Coords sorted_idx[:k] are fully saturated in helpful direction.
    if k > 0:
        full_idx = sorted_idx[:k]
        sign_full = torch.where(
            constraint[full_idx] >= 0,
            torch.ones_like(constraint[full_idx]),
            -torch.ones_like(constraint[full_idx]),
        )
        delta[full_idx] = sign_full * helpful_disp[full_idx]

    # Coord sorted_idx[k] is partially saturated to deliver the leftover.
    delivered_so_far = float(cumsum[k - 1].item()) if k > 0 else 0.0
    leftover = swing - delivered_so_far
    if leftover > 0:
        partial_i = int(sorted_idx[k].item())
        c_partial = float(constraint[partial_i].item())
        abs_c_partial = abs(c_partial)
        if abs_c_partial < 1e-12:
            # Should not happen given total_possible ≥ swing, but guard.
            raise RuntimeError(
                f"Greedy fill landed on a |c|=0 coord with leftover={leftover:.3e}. "
                f"This indicates a numerical bug in cumsum/searchsorted alignment."
            )
        partial_disp = leftover / abs_c_partial
        delta[partial_i] = partial_disp if c_partial > 0 else -partial_disp

    return delta


def collect_activation_pattern(model, x: torch.Tensor) -> dict:
    """Capture ReLU sign patterns and MaxPool argmax indices.
    Used by in_region() to verify that x and x+delta land in the same region.
    Implementation reads model.pre_acts and model.maxpool_indices after a
    save=True forward pass."""
    model.save = True
    _ = model(x)
    model.save = False
    pattern = {
        "pre_act_signs": {
            i: (pa > 0).cpu() for i, pa in enumerate(model.pre_acts) if pa is not None
        },
        "pool_indices": {
            i: idx.cpu() for i, idx in enumerate(model.maxpool_indices) if idx is not None
        },
    }
    return pattern


def in_region(model, x: torch.Tensor, x_perturbed: torch.Tensor) -> bool:
    pat_a = collect_activation_pattern(model, x)
    pat_b = collect_activation_pattern(model, x_perturbed)
    if set(pat_a["pre_act_signs"]) != set(pat_b["pre_act_signs"]):
        return False
    for i in pat_a["pre_act_signs"]:
        if not torch.equal(pat_a["pre_act_signs"][i], pat_b["pre_act_signs"][i]):
            return False
    for i in pat_a["pool_indices"]:
        if not torch.equal(pat_a["pool_indices"][i], pat_b["pool_indices"][i]):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--n-source-images", type=int, default=3,
                        help="Per (model, class) pair, how many images to use as LP sources")
    parser.add_argument("--n-targets-per-source", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not patch_available():
        print("ERROR: knowledgematrix patch missing.", file=sys.stderr)
        return 2

    entries = read_manifest(args.manifest)

    by_model_class = {}
    for e in entries:
        by_model_class.setdefault((e.model, e.class_id), []).append(e)

    completed = state.load_completed(paths.state_path("05_counterfactual"))

    n_attempted = 0
    n_success = 0

    print(f"counterfactual_lp: {len(by_model_class)} (model, class) groups; "
          f"{len(completed)} prior completed", flush=True)

    last_model = None
    for (model_name, source_class), samples in by_model_class.items():
        if model_name != last_model:
            t0 = time.perf_counter()
            model = build_model(model_name, args.device)
            print(f"  built {model_name}: {time.perf_counter() - t0:.1f}s",
                  flush=True)
            last_model = model_name

        for e in samples[: args.n_source_images]:
            # W_eff extraction is the expensive part — wrap it separately so
            # we don't recompute for each target. If it fails, skip the whole
            # source image; if LP fails for one target, continue to the next.
            try:
                x = load_image(e.image_path).to(args.device)
                t0 = time.perf_counter()
                W_eff, b_eff = extract_weff_and_beff(model, x)
                out_true = model.forward(x).flatten()
                weff_t = time.perf_counter() - t0
                print(f"  W_eff for {sample_key(e)}: {weff_t:.1f}s", flush=True)
            except Exception as exc:
                state.log_error(
                    paths.errors_path(), step="05_counterfactual",
                    sample_id=sample_key(e),
                    error_type=type(exc).__name__, message=str(exc),
                    tb=state.capture_traceback(),
                )
                continue

            target_pool = [c for c in TIER_A_CLASSES if c != source_class]
            targets = target_pool[: args.n_targets_per_source]
            for target in targets:
                key = f"{sample_key(e)}__to_{target}"
                if key in completed:
                    continue
                n_attempted += 1
                try:
                    t_lp = time.perf_counter()
                    delta = solve_l1_lp(
                        W_eff, b_eff, x.flatten(), out_true,
                        source=source_class, target=target, margin=args.margin,
                    )
                    lp_t = time.perf_counter() - t_lp
                    # No clamp: x_perturbed = x + delta is reported in the
                    # same input-space coordinate frame as x, but is not
                    # constrained to [0,1]. Clamping would silently re-introduce
                    # a box the LP did not enforce, making new_logits and
                    # region_ok misleading. The caller interprets delta as a
                    # mathematical direction (logit-lens-style), not a pixel
                    # perturbation.
                    t_post = time.perf_counter()
                    x_perturbed = (x.flatten() + delta).reshape(x.shape)
                    region_ok = in_region(model, x, x_perturbed)
                    new_logits = model.forward(x_perturbed).flatten()
                    post_t = time.perf_counter() - t_post
                    out_path = paths.counterfactual_path(
                        model_name, source_class, e.image_id, target=target
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "delta_l1": float(delta.abs().sum()),
                        "delta_linf": float(delta.abs().max()),
                        "delta_l2": float(delta.norm()),
                        "new_logit_target": float(new_logits[target]),
                        "new_logit_source": float(new_logits[source_class]),
                        "region_ok": bool(region_ok),
                    }
                    with out_path.open("w") as f:
                        json.dump(payload, f, indent=2)
                    state.mark_completed(paths.state_path("05_counterfactual"), key)
                    n_success += 1
                    print(f"    done {key}  lp={lp_t*1000:.1f}ms  "
                          f"post={post_t:.2f}s  region_ok={region_ok}",
                          flush=True)
                except Exception as exc:
                    state.log_error(
                        paths.errors_path(), step="05_counterfactual",
                        sample_id=key,
                        error_type=type(exc).__name__, message=str(exc),
                        tb=state.capture_traceback(),
                    )
        # Note: we DON'T del model after every (model, class) group — we keep
        # it alive across all classes of the same arch (3 classes per arch),
        # avoiding 2 redundant rebuilds per arch. Only del when arch changes.
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"counterfactual_lp: {n_success}/{n_attempted} succeeded "
          f"(prior completed: {len(completed)})", flush=True)
    if n_attempted > 0 and n_success == 0:
        print("ERROR: zero samples completed this run; failing the step.",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
