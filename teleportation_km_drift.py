#!/usr/bin/env python
"""D2 / NTH-3: measure KM drift under neural teleportation (Study 1b's missing column).

Claim structure being instrumented (plan PDF, Part C, Sec. 9 'Teleportation'):
  (i)  exact isomorphism            => KM drift = 0            [theorem; not asserted, MEASURED]
  (ii) approximate teleportation    => VISIBLE drift ||(M_t - M) 1|| = ||f_t(x) - f(x)|| <= gate
  (iii) INVISIBLE drift             => no a-priori bound exists (no-go theorem); measured, and
        regressed against the per-teleport max_logit_diff (the function-approximation control).

Per (teleport t, sample i) this script records:
  d_M      = ||M_t(x_i) - M(x_i)||_F
  d_vis    = ||f_t(x_i) - f(x_i)||_2            (= ||(M_t - M) 1|| exactly; identity checked)
  m_vis2   = d_vis^2 / (d+1)                     (visible mass)
  m_inv    = sqrt(max(d_M^2 - m_vis2, 0))        (invisible mass — the quantity 1b must report)
plus per-teleport max_logit_diff over the gate set and the completeness residual of each KM model.

Integration: COB models + teleport via neuralteleportation (as teleportation_experiment.py);
teleported weights loaded into the knowledgematrix wrapper via the stratified remap +
bifurcated-residual copy (as cross_model_experiment.py). Wired archs: resnet152, densenet121
(the two with KM alt-weight loading; googlenet needs the same wiring Step E declined).

Usage (cluster, mirrors job_teleportation.sh conventions — see job_km_drift.sh):
  python teleportation_km_drift.py --arch resnet152 --num_teleportations 5 \
      --num_km_samples 50 --num_gate_samples 500 --data <samples.pth> \
      --out results/teleportation_km_drift --device cuda --seed 0
Cost estimate: (T+1) * N_km KM extractions; resnet152 @ ~104 s/KM on H100 => T=5, N=50 ~ 8.7 h.

Resume (added for the 8h-wall policy; measurement logic untouched): after each
teleport t the accumulated results are flushed to {out}/{arch}_km_drift.partial.json
(tmp+rename). On restart with matching (arch, T, N_km, N_gate, seed) the completed
teleports are skipped; teleport seeds are index-keyed (seed*1000+t) so the schedule
is unchanged. M_base/f_base/gate_base are recomputed each slot (deterministic
weights/data/eval-mode; fp32 recompute noise ~1e-7 sits orders of magnitude below
teleportation drift), so per-teleport rows always compare against a same-slot base.
The partial file is removed once the final JSON is written.

Local logic-validation: ../resubmission-artifacts-2026-06-11/scripts/teleportation_km_drift_smoke.py.
"""
import argparse, copy, json, os, sys, time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# --- HAaNE-repo imports (run from the Hidden-Activations-are-not-Enough root) ---
from cross_model_experiment import _stratified_remap, verify_km_completeness  # noqa: E402
from utils.km_models import build_model                                        # noqa: E402
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer            # noqa: E402
from teleportation_experiment import load_pretrained_cob, teleport_model       # noqa: E402

D_PLUS_1 = 3 * 224 * 224 + 1


def load_cob_into_km(arch: str, cob_model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Load a (possibly teleported) COB model's weights into the KM wrapper.

    Same recipe as cross_model_experiment.build_km_model_with_alt_weights, with the
    in-memory COB state_dict as the source. The strict stratum-cardinality check is the
    intended failure mode on any mismatch — investigate, never force.
    """
    km_model = build_model(arch, device)
    km_model.eval()
    km_sd = km_model.state_dict()
    src_sd = {k: v for k, v in cob_model.state_dict().items()}
    remapped = _stratified_remap(arch, "teleported-cob", km_sd, src_sd)
    missing, unexpected = km_model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict missing={missing[:5]} unexpected={unexpected[:5]}")
    # Bifurcated residual storage: copy residual_modules -> residuals (forward path).
    rm_iter = iter(getattr(km_model, "residual_modules", []))
    for entries in getattr(km_model, "residuals", {}).values():
        for _start, projection in entries:
            for sub in projection:
                if isinstance(sub, torch.nn.Identity):
                    continue
                rm_module = next(rm_iter)
                sub.load_state_dict(rm_module.state_dict())
                sub.to(device)
    return km_model


def km_matrices(km_model, xs, device, batch_size=1800):
    out = []
    kmc = KnowledgeMatrixComputer(km_model, batch_size=batch_size, device=device)
    for x in xs:
        out.append(kmc.forward(x.to(device)).detach().cpu())
    return out


def _atomic_json_dump(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet152", "densenet121"])
    ap.add_argument("--num_teleportations", type=int, default=5)
    ap.add_argument("--num_km_samples", type=int, default=50)
    ap.add_argument("--num_gate_samples", type=int, default=500)
    ap.add_argument("--data", required=True, help=".pth tensor (N,3,224,224), Step-B sample set")
    ap.add_argument("--out", default="results/teleportation_km_drift")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--km_batch", type=int, default=1800)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    final = os.path.join(args.out, f"{args.arch}_km_drift.json")
    partial = os.path.join(args.out, f"{args.arch}_km_drift.partial.json")
    if os.path.exists(final):
        print(f"{final} already exists — nothing to do.", flush=True)
        return

    data = torch.load(args.data, map_location="cpu")
    gate_x = data[: args.num_gate_samples]
    km_x = data[: args.num_km_samples]

    # Base COB model (pretrained, eval) via the shared offline-safe loader (the
    # SAME path Step B uses). Kept on CPU: the gate forward below runs on CPU
    # `gate_x`, teleport_model deepcopies this CPU model, and load_cob_into_km
    # reads the (device-agnostic) state_dict into a fresh KM model on args.device
    # — so the COB model's own device is irrelevant to the KM compute.
    #
    # NOT factory(pretrained=True): the COB factories download via their legacy
    # model_urls (resnet152 -> orphan V1 resnet152-b121ed2d.pth), which differs
    # from the torchvision DEFAULT (V2 resnet152-f82ba261.pth) that Phase-0d
    # pre-caches -> would miss the cache and crash on a no-internet compute node.
    # Using the DEFAULT here also aligns D2's measured checkpoint with Steps B/C.
    base = load_pretrained_cob(args.arch, device="cpu")

    # Base KM model + matrices (recomputed every slot; see module docstring).
    km_base_model = load_cob_into_km(args.arch, base, args.device)
    ok, resid = verify_km_completeness(km_base_model, args.device)
    base_resid = resid
    t0 = time.time()
    M_base = km_matrices(km_base_model, km_x, args.device, args.km_batch)
    with torch.no_grad():
        f_base = torch.stack([km_base_model(x.to(args.device)).reshape(-1).cpu() for x in km_x])
        gate_base = torch.stack([base(x.unsqueeze(0)).reshape(-1) for x in gate_x])
    print(f"[base] completeness resid={resid:.3e}; {len(M_base)} KMs in {time.time()-t0:.0f}s", flush=True)

    results = {"arch": args.arch, "T": args.num_teleportations, "N_km": args.num_km_samples,
               "N_gate": args.num_gate_samples, "seed": args.seed, "d_plus_1": D_PLUS_1,
               "base_completeness_resid": base_resid, "per_teleport": []}

    # Resume: adopt a matching partial (same config), skip its completed teleports.
    start_t = 0
    if os.path.exists(partial):
        try:
            with open(partial) as fh:
                prev = json.load(fh)
        except Exception as e:
            prev = None
            print(f"[resume] unreadable partial ({e}) — starting fresh", flush=True)
        if prev is not None:
            same_cfg = all(prev.get(k) == results[k]
                           for k in ("arch", "T", "N_km", "N_gate", "seed"))
            if same_cfg:
                results["per_teleport"] = prev.get("per_teleport", [])
                start_t = len(results["per_teleport"])
                # base_completeness_resid stays the CURRENT slot's value by design.
                print(f"[resume] {start_t}/{args.num_teleportations} teleports already done", flush=True)
            else:
                print("[resume] partial config mismatch — starting fresh", flush=True)

    for t in range(start_t, args.num_teleportations):
        tele = teleport_model(base, input_shape=(1, 3, 224, 224), seed=args.seed * 1000 + t)
        tele.eval()
        with torch.no_grad():
            gate_t = torch.stack([tele(x.unsqueeze(0)).reshape(-1) for x in gate_x])
        max_logit_diff = (gate_t - gate_base).abs().max().item()

        km_t_model = load_cob_into_km(args.arch, tele, args.device)
        ok, resid_t = verify_km_completeness(km_t_model, args.device)
        M_t = km_matrices(km_t_model, km_x, args.device, args.km_batch)
        with torch.no_grad():
            f_t = torch.stack([km_t_model(x.to(args.device)).reshape(-1).cpu() for x in km_x])

        rows = []
        for i in range(len(km_x)):
            dM = (M_t[i] - M_base[i]).double().norm().item()
            dvis = (f_t[i] - f_base[i]).double().norm().item()
            rowsum_id = ((M_t[i] - M_base[i]).double().sum(1)
                         - (f_t[i] - f_base[i]).double()).abs().max().item()
            mvis2 = dvis * dvis / D_PLUS_1
            minv = (max(dM * dM - mvis2, 0.0)) ** 0.5
            rows.append(dict(d_M=dM, d_vis=dvis, m_inv=minv, rowsum_identity_err=rowsum_id))
        results["per_teleport"].append(dict(
            teleport=t, max_logit_diff=max_logit_diff, completeness_resid=resid_t,
            samples=rows,
            d_M_mean=float(np.mean([r["d_M"] for r in rows])),
            m_inv_mean=float(np.mean([r["m_inv"] for r in rows])),
            d_vis_mean=float(np.mean([r["d_vis"] for r in rows])),
        ))
        print(f"[teleport {t}] max_logit_diff={max_logit_diff:.3e}  "
              f"d_M mean={results['per_teleport'][-1]['d_M_mean']:.4g}  "
              f"invisible mean={results['per_teleport'][-1]['m_inv_mean']:.4g}", flush=True)
        _atomic_json_dump(results, partial)

    # Function-approximation control: regression of invisible drift on the gate violation.
    xs = [pt["max_logit_diff"] for pt in results["per_teleport"]]
    ys = [pt["m_inv_mean"] for pt in results["per_teleport"]]
    if len(xs) >= 2 and np.std(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        results["control_regression"] = dict(slope=float(slope), intercept=float(intercept), pearson_r=r)
    _atomic_json_dump(results, final)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"wrote {final}", flush=True)


if __name__ == "__main__":
    main()
