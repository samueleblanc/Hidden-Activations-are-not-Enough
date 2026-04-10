"""
GH200 Benchmark — Batch size optimization and memory budget for knowledge matrix computation.

Determines:
1. Optimal KM batch_size for a given model on GH200 (binary search at 90% GPU memory)
2. Whether concurrent KM computation is feasible
3. Whether the full training dataset fits in system memory for Step E
4. Per-matrix wall-clock time at optimal batch_size

Usage:
    python benchmark_gh200.py --experiment_name vgg_cifar10
    python benchmark_gh200.py --experiment_name vgg_cifar10 --target_utilization 0.85
    python benchmark_gh200.py --experiment_name vgg_cifar10 --force
"""

import gc
import json
import os
import sys
import time
import torch
from argparse import ArgumentParser
from datetime import datetime

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import (
    get_architecture, get_dataset, get_input_shape,
    get_num_classes, get_device, subset
)


def parse_args():
    parser = ArgumentParser(description="GH200 benchmark for KM computation")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, default=None)
    parser.add_argument("--target_utilization", type=float, default=0.90)
    parser.add_argument("--timing_samples", type=int, default=20,
                        help="Number of matrices to compute for timing")
    parser.add_argument("--max_concurrent", type=int, default=5,
                        help="Max concurrent KM results to test holding in memory")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing benchmark results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: experiments/{exp}/benchmark_gh200.json)")
    return parser.parse_args()


def get_system_memory_bytes():
    """Get total system memory in bytes."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # /proc/meminfo reports in kB
    except FileNotFoundError:
        pass
    # macOS fallback
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception:
        return 0


def probe_batch_size(model, sample_input, batch_size, device):
    """
    Try computing one matrix with the given batch_size.
    Returns (success, peak_memory_bytes, elapsed_seconds).
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
        start = time.perf_counter()
        mc.forward(sample_input.to(device))
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        del mc
        return True, peak, elapsed
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return False, 0, 0.0
        raise


def find_optimal_batch_size(model, sample_input, device, target_utilization=0.90):
    """
    Binary search for the largest batch_size that keeps GPU memory <= target.
    Returns (batch_size, peak_memory_bytes, sweep_results).
    """
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_memory * target_utilization)

    C, H, W = model.input_shape
    total_positions = C * H * W

    print(f"GPU total memory: {total_memory / 1e9:.1f} GB", flush=True)
    print(f"Target memory ({target_utilization*100:.0f}%): {target_mem / 1e9:.1f} GB", flush=True)
    print(f"Total input positions: {total_positions}", flush=True)

    sweep = []

    # Phase 1: exponential growth to find upper bound
    best_bs = 64
    best_peak = 0
    best_time = 0.0
    test_bs = 256

    print("Phase 1: Finding upper bound...", flush=True)
    while test_bs <= total_positions:
        success, peak, elapsed = probe_batch_size(model, sample_input, test_bs, device)
        entry = {
            "batch_size": test_bs,
            "success": success,
            "peak_memory_gb": round(peak / 1e9, 3) if success else None,
            "wall_clock_seconds": round(elapsed, 3) if success else None,
            "memory_utilization_pct": round(peak / total_memory * 100, 1) if success else None,
        }
        sweep.append(entry)

        if success and peak <= target_mem:
            best_bs = test_bs
            best_peak = peak
            best_time = elapsed
            print(f"  batch_size={test_bs}: OK "
                  f"(peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%, "
                  f"{elapsed:.2f}s)", flush=True)
            test_bs *= 2
        else:
            if success:
                print(f"  batch_size={test_bs}: Over target "
                      f"(peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%)", flush=True)
            else:
                print(f"  batch_size={test_bs}: OOM", flush=True)
            break

    # Phase 2: binary search between best_bs and test_bs
    low = best_bs
    high = min(test_bs, total_positions)

    print(f"Phase 2: Binary search [{low}, {high}]...", flush=True)
    while low <= high:
        mid = (low + high) // 2
        if mid == best_bs:
            break

        success, peak, elapsed = probe_batch_size(model, sample_input, mid, device)
        entry = {
            "batch_size": mid,
            "success": success,
            "peak_memory_gb": round(peak / 1e9, 3) if success else None,
            "wall_clock_seconds": round(elapsed, 3) if success else None,
            "memory_utilization_pct": round(peak / total_memory * 100, 1) if success else None,
        }
        sweep.append(entry)

        if success and peak <= target_mem:
            best_bs = mid
            best_peak = peak
            best_time = elapsed
            print(f"  batch_size={mid}: OK "
                  f"(peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%, "
                  f"{elapsed:.2f}s)", flush=True)
            low = mid + 1
        else:
            if success:
                print(f"  batch_size={mid}: Over target "
                      f"(peak={peak/1e9:.2f} GB)", flush=True)
            else:
                print(f"  batch_size={mid}: OOM", flush=True)
            high = mid - 1

    print(f"Optimal batch_size: {best_bs} "
          f"(peak={best_peak/1e9:.2f} GB, {best_peak/total_memory*100:.1f}%, "
          f"{best_time:.2f}s)", flush=True)
    return best_bs, best_peak, sweep


def test_concurrent_km(model, samples, batch_size, max_concurrent, device):
    """
    Test holding multiple KM result tensors in GPU memory simultaneously.
    Computes KMs one at a time, keeping results on GPU to measure cumulative memory.
    """
    print(f"\nConcurrent KM test (batch_size={batch_size}, max={max_concurrent})...", flush=True)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    results = []
    times = []

    for i in range(min(max_concurrent, len(samples))):
        start = time.perf_counter()
        try:
            matrix = mc.forward(samples[i].to(device))
            results.append(matrix)  # Keep result tensor on GPU
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            peak = torch.cuda.max_memory_allocated(device)
            print(f"  KM #{i+1}: peak={peak/1e9:.2f} GB "
                  f"({peak/total_memory*100:.1f}%), {elapsed:.2f}s", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  KM #{i+1}: OOM — max concurrent = {i}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                break
            raise

    n_computed = len(results)
    peak = torch.cuda.max_memory_allocated(device)

    # Measure result tensor size
    result_size = results[0].element_size() * results[0].nelement() if results else 0

    # Clean up
    del results, mc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "max_concurrent_computed": n_computed,
        "peak_memory_gb": round(peak / 1e9, 3),
        "peak_memory_utilization_pct": round(peak / total_memory * 100, 1),
        "result_tensor_shape": list(results[0].shape) if n_computed > 0 else None,
        "result_tensor_bytes": result_size,
        "result_tensor_mb": round(result_size / 1e6, 3),
        "times_seconds": [round(t, 3) for t in times],
        "note": ("Result tensors are small; the bottleneck is computation workspace, "
                 "not stored results"),
    } if n_computed > 0 else {"max_concurrent_computed": 0, "error": "OOM on first KM"}


def time_matrix_computation(model, samples, batch_size, device, n_samples=20):
    """Compute n_samples matrices, return average seconds per matrix."""
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    n = min(n_samples, len(samples))
    times = []

    for i in range(n):
        im = samples[i].to(device)
        gc.collect()
        torch.cuda.empty_cache()

        start = time.perf_counter()
        mc.forward(im)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 5 == 0:
            avg_so_far = sum(times) / len(times)
            print(f"  Timed {i+1}/{n} matrices (avg: {avg_so_far:.2f}s)", flush=True)

    avg = sum(times) / len(times)
    print(f"Average time per matrix: {avg:.3f}s ({n} samples)", flush=True)
    return avg, times


def compute_memory_budget(model, dataset_name, system_memory_bytes):
    """Compute memory budget for holding all training KMs in system memory (Step E)."""
    num_classes = get_num_classes(dataset_name)
    matrix_shape = model.get_matrix_shape()  # (rows, cols) = (output_size, input_size + 1)
    km_elements = matrix_shape[0] * matrix_shape[1]
    km_bytes = km_elements * 4  # float32

    # CIFAR-10: 5000 train samples per class, CIFAR-100: 500 per class
    if dataset_name == "cifar10":
        total_train_samples = 50000
        samples_per_class = 5000
    elif dataset_name == "cifar100":
        total_train_samples = 50000
        samples_per_class = 500
    else:
        total_train_samples = 50000  # conservative default
        samples_per_class = total_train_samples // num_classes

    total_km_bytes = total_train_samples * km_bytes
    reserve = 0.10  # Keep 10% for OS + detectors + other data
    available = int(system_memory_bytes * (1 - reserve))
    fits = total_km_bytes < available

    return {
        "matrix_shape": list(matrix_shape),
        "km_elements_per_sample": km_elements,
        "km_bytes_per_sample": km_bytes,
        "km_mb_per_sample": round(km_bytes / 1e6, 3),
        "num_classes": num_classes,
        "samples_per_class": samples_per_class,
        "total_train_samples": total_train_samples,
        "total_km_gb": round(total_km_bytes / 1e9, 2),
        "system_memory_gb": round(system_memory_bytes / 1e9, 1),
        "available_gb_90pct": round(available / 1e9, 1),
        "fits_in_system_memory": fits,
        "utilization_pct": round(total_km_bytes / system_memory_bytes * 100, 1),
    }


def main():
    args = parse_args()
    experiment = args.experiment_name

    if experiment not in DEFAULT_EXPERIMENTS:
        print(f"ERROR: '{experiment}' not found in DEFAULT_EXPERIMENTS")
        print(f"Available: {list(DEFAULT_EXPERIMENTS.keys())}")
        sys.exit(1)

    exp_config = DEFAULT_EXPERIMENTS[experiment]

    # Output path
    output_path = args.output or f"experiments/{experiment}/benchmark_gh200.json"
    if os.path.exists(output_path) and not args.force:
        print(f"Benchmark already exists: {output_path}")
        print("Use --force to re-run.")
        with open(output_path) as f:
            data = json.load(f)
        print(f"  optimal_batch_size: {data.get('optimal_batch_size')}")
        print(f"  avg_seconds_per_matrix: {data.get('avg_seconds_per_matrix')}")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmark requires a GPU.")
        sys.exit(1)

    device = get_device()
    dataset_name = exp_config['dataset']
    architecture_index = exp_config['architecture_index']
    input_shape = get_input_shape(dataset_name)
    num_classes = get_num_classes(dataset_name)
    system_memory = get_system_memory_bytes()

    gpu_props = torch.cuda.get_device_properties(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = gpu_props.total_memory

    C, H, W = input_shape
    total_positions = C * H * W

    print("=" * 60, flush=True)
    print(f"  GH200 Benchmark: {experiment}", flush=True)
    print(f"  GPU: {gpu_name} ({gpu_mem / 1e9:.1f} GB)", flush=True)
    print(f"  System memory: {system_memory / 1e9:.1f} GB", flush=True)
    print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}", flush=True)
    print(f"  Architecture index: {architecture_index}", flush=True)
    print(f"  Dataset: {dataset_name}", flush=True)
    print(f"  Input shape: {input_shape}", flush=True)
    print(f"  Total positions: {total_positions}", flush=True)
    print("=" * 60, flush=True)

    # --- Load dataset ---
    print("\nLoading dataset...", flush=True)
    train_set, _ = get_dataset(
        data_set=dataset_name,
        data_loader=False,
        data_path=args.temp_dir
    )
    n_needed = max(args.timing_samples, args.max_concurrent) + 5
    sample_data, _ = subset(train_set, min(n_needed, len(train_set)), input_shape)
    print(f"Loaded {len(sample_data)} samples for benchmarking", flush=True)

    # --- Create model ---
    print("\nCreating model...", flush=True)
    model = get_architecture(
        input_shape, num_classes, architecture_index,
        pretrained=False, freeze_features=False
    ).to(device)
    model.input_shape = input_shape
    model.eval()
    print(f"Model input_shape: {model.input_shape}", flush=True)
    print(f"Model matrix_shape: {model.get_matrix_shape()}", flush=True)

    # --- Step 1: Batch size sweep ---
    print("\n--- Step 1: Batch size sweep ---", flush=True)
    sample_input = sample_data[0].to(device)
    optimal_bs, peak_mem, sweep_results = find_optimal_batch_size(
        model, sample_input, device, args.target_utilization
    )

    # --- Step 2: Timing at optimal batch_size ---
    print(f"\n--- Step 2: Timing ({args.timing_samples} matrices at batch_size={optimal_bs}) ---",
          flush=True)
    avg_time, all_times = time_matrix_computation(
        model, sample_data, optimal_bs, device, args.timing_samples
    )

    # --- Step 3: Concurrent KM test ---
    print("\n--- Step 3: Concurrent KM computation ---", flush=True)
    concurrent_results = test_concurrent_km(
        model, sample_data, optimal_bs, args.max_concurrent, device
    )

    # --- Step 4: Memory budget for full dataset ---
    print("\n--- Step 4: Full dataset memory budget ---", flush=True)
    memory_budget = compute_memory_budget(model, dataset_name, system_memory)
    print(f"  KM per sample: {memory_budget['km_mb_per_sample']} MB", flush=True)
    print(f"  Total for {memory_budget['total_train_samples']} samples: "
          f"{memory_budget['total_km_gb']} GB", flush=True)
    print(f"  System memory: {memory_budget['system_memory_gb']} GB", flush=True)
    print(f"  Fits in 90% system memory: {memory_budget['fits_in_system_memory']}", flush=True)

    # --- Estimate total Step B time ---
    total_train = memory_budget['total_train_samples']
    est_step_b_seconds = avg_time * total_train
    est_step_b_hours = est_step_b_seconds / 3600

    # --- Save results ---
    results = {
        "experiment_name": experiment,
        "gpu": {
            "name": gpu_name,
            "hbm_gb": round(gpu_mem / 1e9, 1),
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
        },
        "system_memory_gb": round(system_memory / 1e9, 1),
        "model": type(model).__name__,
        "dataset": dataset_name,
        "input_shape": list(input_shape),
        "total_positions": total_positions,
        "target_utilization": args.target_utilization,
        "optimal_batch_size": optimal_bs,
        "peak_memory_gb": round(peak_mem / 1e9, 3),
        "peak_memory_utilization_pct": round(peak_mem / gpu_mem * 100, 1),
        "avg_seconds_per_matrix": round(avg_time, 4),
        "all_times_seconds": [round(t, 4) for t in all_times],
        "batch_size_sweep": sweep_results,
        "concurrent_km": concurrent_results,
        "memory_budget": memory_budget,
        "estimated_step_b": {
            "total_matrices": total_train,
            "total_seconds": round(est_step_b_seconds, 1),
            "total_hours": round(est_step_b_hours, 2),
            "note": "Single-GPU sequential estimate (no chunking overhead)",
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Results saved to: {output_path}", flush=True)
    print(f"  optimal_batch_size = {optimal_bs}", flush=True)
    print(f"  GPU utilization = {peak_mem / gpu_mem * 100:.1f}%", flush=True)
    print(f"  avg time/matrix = {avg_time:.3f}s", flush=True)
    print(f"  full dataset ({total_train} matrices) = {est_step_b_hours:.1f}h", flush=True)
    print(f"  full dataset KMs in memory = {memory_budget['total_km_gb']} GB "
          f"({'OK' if memory_budget['fits_in_system_memory'] else 'EXCEEDS MEMORY'})", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
