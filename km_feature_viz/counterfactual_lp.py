"""Step 05: LP counterfactual via the patched extract_weff API."""
import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy
import scipy.optimize
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
    """Solve  min ||delta||_1  s.t.  (W[t] - W[s]) . delta >= m - (f[t] - f[s]).

    No box constraint on (x + delta): the returned delta is the L1-minimal
    direction in input-space that, *within the source's linear region*,
    achieves the target margin. It is a mathematical direction in M(x)-space
    (analogous to a logit-lens projection), not a valid pixel-space
    perturbation. Whether x + delta stays in the source's linear region is
    a separate question, answered by the caller via in_region().

    Without the box, the LP is trivially feasible whenever (W[t] - W[s]) is
    not the zero vector — the closed-form optimum is delta = (swing / |c_k|)
    e_k, k = argmax |c|. We still solve it via linprog so the same code path
    handles edge cases (swing <= 0 ⇒ delta = 0).
    """
    n = W_eff.shape[1]
    swing = (margin - (out_true[target] - out_true[source])).item()
    constraint = (W_eff[target] - W_eff[source]).cpu().numpy()

    # L1 trick: delta = delta+ - delta-, both >= 0.
    # Variables: [delta+ (n), delta- (n)]. Objective: sum(delta+ + delta-).
    c = np.ones(2 * n)
    A_ub = -np.concatenate([constraint, -constraint]).reshape(1, 2 * n)
    b_ub = np.array([-swing])
    bounds = [(0, None)] * (2 * n)

    _scipy_version = tuple(int(v) for v in scipy.__version__.split(".")[:2])
    lp_method = "highs" if _scipy_version >= (1, 7) else "interior-point"
    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=lp_method)
    if not res.success:
        raise RuntimeError(f"LP infeasible: {res.message}")
    delta = res.x[:n] - res.x[n:]
    return torch.from_numpy(delta).to(W_eff.dtype)


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

    for (model_name, source_class), samples in by_model_class.items():
        model = build_model(model_name, args.device)
        for e in samples[: args.n_source_images]:
            # W_eff extraction is the expensive part — wrap it separately so
            # we don't recompute for each target. If it fails, skip the whole
            # source image; if LP fails for one target, continue to the next.
            try:
                x = load_image(e.image_path).to(args.device)
                W_eff, b_eff = extract_weff_and_beff(model, x)
                out_true = model.forward(x).flatten()
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
                    delta = solve_l1_lp(
                        W_eff, b_eff, x.flatten(), out_true,
                        source=source_class, target=target, margin=args.margin,
                    )
                    # No clamp: x_perturbed = x + delta is reported in the
                    # same input-space coordinate frame as x, but is not
                    # constrained to [0,1]. Clamping would silently re-introduce
                    # a box the LP did not enforce, making new_logits and
                    # region_ok misleading. The caller interprets delta as a
                    # mathematical direction (logit-lens-style), not a pixel
                    # perturbation.
                    x_perturbed = (x.flatten() + delta).reshape(x.shape)
                    region_ok = in_region(model, x, x_perturbed)
                    new_logits = model.forward(x_perturbed).flatten()
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
                    logger.info("done %s", key)
                except Exception as exc:
                    state.log_error(
                        paths.errors_path(), step="05_counterfactual",
                        sample_id=key,
                        error_type=type(exc).__name__, message=str(exc),
                        tb=state.capture_traceback(),
                    )
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
