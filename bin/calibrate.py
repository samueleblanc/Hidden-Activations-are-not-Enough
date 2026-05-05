"""Per-architecture three-tier KM batch-size calibration.

Restored from legacy/calibrate.py. Computes optimal KM-extraction batch
size at three GPU-memory utilization tiers (85%, 90%, 93%) per
architecture, persists to experiments/calibration/{arch}_imagenet/calibration.json.

Used by:
- All Phase 1 workers (S1, S2, S3) -- read active_tier batch size at startup.
- bin/sentinel.sh -- step active tier down on CUDA-OOM.
"""
import os
import sys
import json
import time
import math
import tempfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

TIERS = ("85", "90", "93")
TIER_FRACTIONS = {"85": 0.85, "90": 0.90, "93": 0.93}
DEFAULT_ACTIVE_TIER = "93"


def write_calibration_json(out_path, arch, gpu_name, gpu_memory_bytes, tiers,
                           avg_seconds_per_km, input_shape, active_tier=DEFAULT_ACTIVE_TIER):
    """Write the canonical 3-tier calibration JSON."""
    from utils.atomic_io import atomic_json_dump
    data = {
        "arch": arch,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "tiers": tiers,
        "active_tier": active_tier,
        "avg_seconds_per_km": avg_seconds_per_km,
        "input_shape": list(input_shape),
        "timestamp": datetime.now().isoformat(),
    }
    atomic_json_dump(out_path, data)


def get_active_batch_size(calib_path):
    """Read calibration.json and return the batch size for the active tier."""
    with open(calib_path) as f:
        data = json.load(f)
    tier = data["active_tier"]
    return data["tiers"][tier]["km_batch_size"]


def step_down_tier(calib_path):
    """Move the active tier one step down (93 -> 90, 90 -> 85). Raise on below-85."""
    from utils.atomic_io import atomic_json_dump
    with open(calib_path) as f:
        data = json.load(f)
    current = data["active_tier"]
    next_tier = {"93": "90", "90": "85", "85": None}[current]
    if next_tier is None:
        raise RuntimeError(f"Cannot step below 85% tier for {data.get('arch')}")
    data["active_tier"] = next_tier
    atomic_json_dump(calib_path, data)
    return next_tier


def probe_batch_size(model, sample_input, batch_size, device):
    """Try a single KM forward at the given batch_size; return (success, peak_mem_bytes)."""
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
        mc.forward(sample_input.to(device))
        return True, torch.cuda.max_memory_allocated(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0
        raise


def find_three_tier_batch_sizes(model, sample_input, device):
    """Single binary-search sweep that records the largest batch size for each tier."""
    total_mem = torch.cuda.get_device_properties(device).total_memory
    targets = {tier: int(total_mem * frac) for tier, frac in TIER_FRACTIONS.items()}

    # Phase 1: exponential growth to find an upper bound
    bs = 256
    best = {tier: (64, 0) for tier in TIERS}  # (batch_size, peak_mem)
    while True:
        success, peak = probe_batch_size(model, sample_input, bs, device)
        if not success:
            print(f"  bs={bs}: OOM", flush=True)
            break
        for tier in TIERS:
            if peak <= targets[tier] and bs > best[tier][0]:
                best[tier] = (bs, peak)
        if peak > targets["93"]:
            print(f"  bs={bs}: over all targets (peak={peak/1e9:.2f}GB)", flush=True)
            break
        print(f"  bs={bs}: ok (peak={peak/1e9:.2f}GB, {peak/total_mem*100:.1f}%)", flush=True)
        bs *= 2

    # Phase 2: binary search between max-ok-93 and the OOM/over-93 batch
    lo, hi = best["93"][0], bs
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        success, peak = probe_batch_size(model, sample_input, mid, device)
        if success:
            for tier in TIERS:
                if peak <= targets[tier] and mid > best[tier][0]:
                    best[tier] = (mid, peak)
            if peak <= targets["93"]:
                lo = mid
            else:
                hi = mid
        else:
            hi = mid

    return {
        tier: {
            "km_batch_size": int(best[tier][0]),
            "peak_memory_bytes": int(best[tier][1]),
            "fraction": TIER_FRACTIONS[tier],
        }
        for tier in TIERS
    }


def time_km_computation(model, sample_inputs, batch_size, device, n_samples=50):
    """Time KM computation on n_samples; return average seconds per matrix."""
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    model.eval()
    times = []
    n = min(n_samples, len(sample_inputs))
    for i in range(n):
        torch.cuda.empty_cache()
        x = sample_inputs[i].to(device)
        start = time.time()
        mc.forward(x)
        times.append(time.time() - start)
    return sum(times) / len(times)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timing_samples", type=int, default=50)
    args = parser.parse_args()

    if os.path.exists(args.out_path) and not args.force:
        print(f"Calibration exists: {args.out_path} (use --force to re-run)")
        return

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    # Lazy import: utils.utils pulls in knowledgematrix model classes; don't
    # force that during the JSON-only unit tests.
    from utils.utils import get_device
    device = get_device()

    # Load pretrained arch (on CPU first, then transfer to GPU)
    import torchvision.models as tvm
    arch_loader = {
        "resnet152": tvm.resnet152,
        "densenet121": tvm.densenet121,
        "googlenet": lambda **kw: tvm.googlenet(aux_logits=False, **kw),
    }[args.arch]
    model = arch_loader(weights="DEFAULT").to(device).eval()
    model.input_shape = (3, 224, 224)

    # Sample input -- random ImageNet-shaped tensor for calibration purposes
    sample = torch.randn(3, 224, 224)

    print(f"Calibrating {args.arch} on {torch.cuda.get_device_name(device)}", flush=True)
    tiers = find_three_tier_batch_sizes(model, sample, device)

    # Time the 93% tier batch size
    timing_inputs = [torch.randn(3, 224, 224) for _ in range(args.timing_samples)]
    avg_t = time_km_computation(model, timing_inputs, tiers["93"]["km_batch_size"], device,
                                 n_samples=args.timing_samples)

    write_calibration_json(
        out_path=args.out_path,
        arch=args.arch,
        gpu_name=torch.cuda.get_device_name(device),
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        tiers=tiers,
        avg_seconds_per_km=round(avg_t, 4),
        input_shape=[3, 224, 224],
    )

    print(f"Calibration saved: {args.out_path}", flush=True)
    for tier in TIERS:
        print(f"  tier {tier}%: bs={tiers[tier]['km_batch_size']}, "
              f"peak={tiers[tier]['peak_memory_bytes']/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
