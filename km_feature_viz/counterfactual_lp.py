"""Step 05: closed-form L1 counterfactual via the patched extract_weff API."""
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
from km_feature_viz.compute_kms import build_model, load_image
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
    """Closed-form L1-minimal direction satisfying  ⟨c, δ⟩ ≥ swing,  no box.

    For c = W_eff[t] - W_eff[s] and swing = margin - (f[t] - f[s]):
      * swing ≤ 0 ⇒ δ = 0 is feasible and optimal.
      * swing > 0 ⇒ optimum is single-coordinate:  δ_k = swing / c_k,  where
        k = argmax |c|, and all other coords are 0. Then ‖δ‖₁ = swing/|c_k|;
        any other choice has a strictly larger L1 cost (1/|c_j| ≥ 1/|c_k|).

    Equivalent to the LP `min ‖δ‖₁ s.t. ⟨c, δ⟩ ≥ swing` solved previously via
    scipy.optimize.linprog with the L1 doubling trick. Closed form is O(n)
    on GPU vs. scipy presolve/simplex bookkeeping over 2n=~300k variables —
    the LP solver was the bottleneck of the prior 1-hour SLURM timeout.
    """
    n = W_eff.shape[1]
    swing = (margin - (out_true[target] - out_true[source])).item()
    if swing <= 0:
        return torch.zeros(n, dtype=W_eff.dtype, device=W_eff.device)

    constraint = W_eff[target] - W_eff[source]
    abs_c = constraint.abs()
    k = int(abs_c.argmax().item())
    c_k = constraint[k].item()
    if abs(c_k) < 1e-12:
        raise RuntimeError(
            f"Constraint vector (W[t]-W[s]) is effectively zero: |c|_max={abs(c_k):.3e}"
        )

    delta = torch.zeros(n, dtype=W_eff.dtype, device=W_eff.device)
    delta[k] = swing / c_k
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
