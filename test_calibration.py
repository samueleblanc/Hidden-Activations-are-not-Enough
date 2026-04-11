"""
Comprehensive GH200 Matrix Benchmark — 1k matrices per experiment.

Benchmarks knowledge matrix computation across all 4 main experiments
(alexnet_cifar10, resnet_cifar10, resnet_cifar100, vgg_cifar100) to
estimate full pipeline wall-clock time and memory on the GH200.

For each experiment:
  1. Calibrates optimal batch_size (binary search with cliff detection)
  2. Computes 1000 matrices with per-matrix timing
  3. Monitors GPU + system memory via background thread
  4. Extrapolates to full pipeline (Step B + Step D)

ALL output goes to stdout (flush=True) so the SLURM .out file is
self-contained for copy-paste analysis.

Usage:
    python test_calibration.py                    # all 4 experiments, 1000 matrices each
    python test_calibration.py --num_matrices 500 # fewer matrices
    python test_calibration.py --experiments alexnet_cifar10 resnet_cifar10  # subset
"""

from utils.unified_memory import init_unified_memory
init_unified_memory()

import gc
import json
import os
import resource
import subprocess
import sys
import statistics
import time
import threading
from argparse import ArgumentParser
from datetime import datetime

import torch
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.utils import get_architecture, get_dataset, get_input_shape, get_num_classes, get_device, subset, move_residuals_to_device


# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    parser = ArgumentParser(description="GH200 matrix computation benchmark (1k matrices)")
    parser.add_argument("--experiments", nargs="+",
                        default=["alexnet_cifar10", "resnet_cifar10", "resnet_cifar100", "vgg_cifar100"],
                        help="Experiments to benchmark (default: all 4 main)")
    parser.add_argument("--num_matrices", type=int, default=1000,
                        help="Number of matrices to compute per experiment (default: 1000)")
    parser.add_argument("--max_batch_size", type=int, default=8192,
                        help="Upper bound for batch_size search (default: 8192)")
    parser.add_argument("--target_utilization", type=float, default=0.70,
                        help="Target GPU utilization for batch_size search (default: 0.70)")
    parser.add_argument("--num_samples_per_class", type=int, default=500,
                        help="Assumed samples per class for pipeline extrapolation (default: 500)")
    parser.add_argument("--samples_per_attack", type=int, default=500,
                        help="Assumed samples per attack for pipeline extrapolation (default: 500)")
    parser.add_argument("--total_chunks", type=int, default=1,
                        help="Assumed chunk count for pipeline extrapolation (default: 1)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for dataset (e.g. $SLURM_TMPDIR)")
    parser.add_argument("--output", type=str, default="experiments/benchmark_1k.json",
                        help="Path for JSON report (default: experiments/benchmark_1k.json)")
    return parser.parse_args()


# ── Memory Monitor ───────────────────────────────────────────────

class MemoryMonitor:
    """Background thread that samples GPU + system memory every interval seconds."""

    def __init__(self, interval=10):
        self.interval = interval
        self.samples = []  # [(timestamp, gpu_mem_mb, gpu_util_pct, sys_avail_mb, rss_mb)]
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def _run(self):
        while not self._stop.is_set():
            try:
                sample = self._sample()
                self.samples.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _sample(self):
        ts = time.time()

        # GPU memory + utilization via nvidia-smi
        gpu_mem_mb = 0.0
        gpu_util = 0.0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_mem_mb = float(parts[0].strip())
                gpu_util = float(parts[1].strip())
        except Exception:
            pass

        # System available memory from /proc/meminfo
        sys_avail_mb = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        sys_avail_mb = int(line.split()[1]) / 1024  # kB -> MB
                        break
        except Exception:
            pass

        # Process RSS
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # kB -> MB on Linux

        return (ts, gpu_mem_mb, gpu_util, sys_avail_mb, rss_mb)

    def peak_gpu_mem_mb(self):
        if not self.samples:
            return 0.0
        return max(s[1] for s in self.samples)

    def peak_gpu_util(self):
        if not self.samples:
            return 0.0
        return max(s[2] for s in self.samples)

    def min_sys_avail_mb(self):
        if not self.samples:
            return 0.0
        return min(s[3] for s in self.samples)

    def peak_rss_mb(self):
        if not self.samples:
            return 0.0
        return max(s[4] for s in self.samples)

    def summary_dict(self):
        if not self.samples:
            return {}
        gpu_mems = [s[1] for s in self.samples]
        gpu_utils = [s[2] for s in self.samples]
        sys_avails = [s[3] for s in self.samples]
        return {
            "num_samples": len(self.samples),
            "gpu_mem_mb_peak": max(gpu_mems),
            "gpu_mem_mb_avg": statistics.mean(gpu_mems),
            "gpu_util_pct_peak": max(gpu_utils),
            "gpu_util_pct_avg": statistics.mean(gpu_utils),
            "sys_avail_mb_min": min(sys_avails),
            "sys_avail_mb_avg": statistics.mean(sys_avails),
            "process_rss_mb_peak": self.peak_rss_mb(),
        }


# ── Batch Size Calibration (from calibrate.py) ──────────────────

def probe_batch_size(model, sample_input, batch_size, device):
    """Try computing one matrix with the given batch_size. Returns (success, peak_bytes)."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass

    try:
        mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
        mc.forward(sample_input.to(device))
        try:
            peak = torch.cuda.max_memory_allocated(device)
        except Exception:
            peak = 0
        del mc
        gc.collect()
        torch.cuda.empty_cache()
        return True, peak
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return False, 0
        raise


def find_optimal_batch_size(model, sample_input, device, target_utilization=0.70,
                            max_batch_size=8192):
    """Binary search for largest batch_size that doesn't OOM, with cliff detection."""
    from utils.unified_memory import is_unified_memory_active, get_total_memory_bytes

    unified = is_unified_memory_active()
    total_memory = get_total_memory_bytes(device)
    hbm_memory = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(hbm_memory * target_utilization)

    C, H, W = model.input_shape
    total_positions = C * H * W

    if unified:
        print(f"  Unified memory pool: {total_memory / 1e9:.1f} GB (OOM-based search)", flush=True)
    else:
        print(f"  GPU total memory: {hbm_memory / 1e9:.1f} GB", flush=True)
        print(f"  Target memory ({target_utilization*100:.0f}%): {target_mem / 1e9:.1f} GB", flush=True)
    print(f"  Total input positions: {total_positions}", flush=True)

    def is_acceptable(success, peak):
        if not success:
            return False
        if unified:
            return True
        return peak <= target_mem

    # Phase 1: exponential growth
    best_bs = 64
    best_peak = 0
    test_bs = 256
    prev_probe_time = None
    cliff_detected = False
    PROBE_SLOWDOWN_FACTOR = 3.0
    probes = []

    upper_limit = min(total_positions, max_batch_size)
    if total_positions > 50000:
        upper_limit = min(upper_limit, 4096)
    print(f"  Phase 1: Finding upper bound (cap={upper_limit})...", flush=True)

    while test_bs <= upper_limit:
        probe_start = time.time()
        success, peak = probe_batch_size(model, sample_input, test_bs, device)
        probe_elapsed = time.time() - probe_start

        if is_acceptable(success, peak):
            if prev_probe_time is not None and probe_elapsed > prev_probe_time * PROBE_SLOWDOWN_FACTOR:
                print(f"    batch_size={test_bs}: OK but {probe_elapsed:.1f}s "
                      f"(>{PROBE_SLOWDOWN_FACTOR}x previous {prev_probe_time:.1f}s) — "
                      f"cliff detected, stopping", flush=True)
                cliff_detected = True
                break

            best_bs = test_bs
            best_peak = peak
            probes.append((test_bs, probe_elapsed, peak))
            print(f"    batch_size={test_bs}: OK ({probe_elapsed:.1f}s)", flush=True)
            prev_probe_time = probe_elapsed
            test_bs *= 2
        else:
            if success:
                print(f"    batch_size={test_bs}: Over target (peak={peak/1e9:.2f} GB)", flush=True)
            else:
                print(f"    batch_size={test_bs}: OOM", flush=True)
            break

    # Phase 2: binary search
    low = best_bs
    high = min(test_bs, upper_limit)

    if cliff_detected:
        best_probe_time = min(t for _, t, _ in probes)
        print(f"  Phase 2: Throughput search [{low}, {high}] "
              f"(best so far: {best_probe_time:.1f}s)...", flush=True)
        while low <= high:
            mid = (low + high) // 2
            if mid == best_bs:
                break
            probe_start = time.time()
            success, peak = probe_batch_size(model, sample_input, mid, device)
            probe_elapsed = time.time() - probe_start
            if not success:
                print(f"    batch_size={mid}: OOM", flush=True)
                high = mid - 1
                continue
            if probe_elapsed <= best_probe_time * PROBE_SLOWDOWN_FACTOR:
                probes.append((mid, probe_elapsed, peak))
                print(f"    batch_size={mid}: OK ({probe_elapsed:.1f}s)", flush=True)
                low = mid + 1
            else:
                print(f"    batch_size={mid}: {probe_elapsed:.1f}s — rejected", flush=True)
                high = mid - 1

        fastest_time = min(t for _, t, _ in probes)
        NOISE_THRESHOLD = 1.10
        acceptable = [(bs, t, p) for bs, t, p in probes if t <= fastest_time * NOISE_THRESHOLD]
        best_bs, _, best_peak = max(acceptable, key=lambda x: x[0])
        print(f"  Optimal batch_size: {best_bs} (throughput-optimized)", flush=True)
    else:
        print(f"  Phase 2: Binary search [{low}, {high}]...", flush=True)
        while low <= high:
            mid = (low + high) // 2
            if mid == best_bs:
                break
            success, peak = probe_batch_size(model, sample_input, mid, device)
            if is_acceptable(success, peak):
                best_bs = mid
                best_peak = peak
                print(f"    batch_size={mid}: OK", flush=True)
                low = mid + 1
            else:
                print(f"    batch_size={mid}: {'OOM' if not success else 'Over target'}", flush=True)
                high = mid - 1
        print(f"  Optimal batch_size: {best_bs}", flush=True)

    return best_bs, best_peak


# ── Core Benchmark ───────────────────────────────────────────────

def benchmark_experiment(experiment_name, num_matrices, max_batch_size, target_utilization,
                         num_samples_per_class, samples_per_attack, total_chunks, temp_dir):
    """Run the full benchmark for one experiment. Returns a results dict."""

    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset_name = exp_config['dataset']
    architecture_index = exp_config['architecture_index']
    input_shape = get_input_shape(dataset_name)
    num_classes = get_num_classes(dataset_name)
    C, H, W = input_shape
    total_positions = C * H * W

    print(f"\n{'='*70}", flush=True)
    print(f"  BENCHMARK: {experiment_name}", flush=True)
    print(f"  Architecture index: {architecture_index}", flush=True)
    print(f"  Dataset: {dataset_name} ({num_classes} classes)", flush=True)
    print(f"  Input shape: {input_shape} ({total_positions} positions)", flush=True)
    print(f"  Matrices to compute: {num_matrices}", flush=True)
    print(f"{'='*70}", flush=True)

    device = get_device()

    # ── Load dataset ──
    print("\n  Loading dataset...", flush=True)
    ds, _ = get_dataset(dataset_name, data_loader=False, data_path=temp_dir)
    data, labels = subset(ds, num_matrices, input_shape)
    print(f"  Dataset loaded: {data.shape}", flush=True)

    # ── Create model ──
    print("  Creating model (random weights)...", flush=True)
    model = get_architecture(input_shape, num_classes, architecture_index,
                             pretrained=False, freeze_features=False).to(device)
    move_residuals_to_device(model, device)
    model.input_shape = input_shape
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}", flush=True)

    # ── Calibrate batch_size ──
    print("\n  --- Batch Size Calibration ---", flush=True)
    sample_input = data[0].to(device)
    batch_size, peak_mem = find_optimal_batch_size(
        model, sample_input, device, target_utilization, max_batch_size
    )
    print(f"  Selected batch_size: {batch_size}", flush=True)

    # ── Warmup (2 matrices, not timed) ──
    print("\n  --- Warmup (2 matrices) ---", flush=True)
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    for i in range(min(2, num_matrices)):
        torch.cuda.empty_cache()
        mc.forward(data[i].to(device))
    print("  Warmup done.", flush=True)

    # ── Start memory monitor ──
    monitor = MemoryMonitor(interval=10)
    monitor.start()

    # ── Compute matrices ──
    print(f"\n  --- Computing {num_matrices} matrices (batch_size={batch_size}) ---", flush=True)
    per_matrix_times = []
    bench_start = time.perf_counter()

    for i in range(num_matrices):
        torch.cuda.empty_cache()
        gc.collect()

        t0 = time.perf_counter()
        matrix = mc.forward(data[i].to(device))
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_matrix_times.append(elapsed)
        del matrix

        if (i + 1) % 100 == 0 or i == 0:
            avg_so_far = statistics.mean(per_matrix_times)
            eta_sec = avg_so_far * (num_matrices - i - 1)
            mem_snap = monitor.samples[-1] if monitor.samples else None
            gpu_str = f"GPU={mem_snap[1]:.0f}MB/{mem_snap[2]:.0f}%" if mem_snap else "N/A"
            print(f"  [{i+1:>5}/{num_matrices}]  this={elapsed:.2f}s  "
                  f"avg={avg_so_far:.2f}s  ETA={eta_sec/60:.1f}min  {gpu_str}", flush=True)

    total_bench_time = time.perf_counter() - bench_start
    monitor.stop()

    # ── Statistics ──
    times = per_matrix_times
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    min_t = min(times)
    max_t = max(times)
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    mem_summary = monitor.summary_dict()

    print(f"\n  --- Results: {experiment_name} ---", flush=True)
    print(f"  Total wall time:    {total_bench_time:.1f}s ({total_bench_time/60:.1f} min)", flush=True)
    print(f"  Matrices computed:  {num_matrices}", flush=True)
    print(f"  Batch size:         {batch_size}", flush=True)
    print(f"  Throughput:         {num_matrices/total_bench_time:.2f} matrices/sec", flush=True)
    print(f"", flush=True)
    print(f"  Per-matrix time (seconds):", flush=True)
    print(f"    Mean:   {mean_t:.3f}", flush=True)
    print(f"    Std:    {std_t:.3f}", flush=True)
    print(f"    Min:    {min_t:.3f}", flush=True)
    print(f"    Max:    {max_t:.3f}", flush=True)
    print(f"    P50:    {p50:.3f}", flush=True)
    print(f"    P95:    {p95:.3f}", flush=True)
    print(f"    P99:    {p99:.3f}", flush=True)
    print(f"", flush=True)
    print(f"  Memory:", flush=True)
    print(f"    GPU HBM peak:       {mem_summary.get('gpu_mem_mb_peak', 0):.0f} MB "
          f"({mem_summary.get('gpu_mem_mb_peak', 0)/1024:.1f} GB)", flush=True)
    print(f"    GPU HBM avg:        {mem_summary.get('gpu_mem_mb_avg', 0):.0f} MB", flush=True)
    print(f"    GPU utilization:    peak={mem_summary.get('gpu_util_pct_peak', 0):.0f}% "
          f"avg={mem_summary.get('gpu_util_pct_avg', 0):.0f}%", flush=True)
    print(f"    Sys RAM available:  min={mem_summary.get('sys_avail_mb_min', 0)/1024:.1f} GB "
          f"avg={mem_summary.get('sys_avail_mb_avg', 0)/1024:.1f} GB", flush=True)
    print(f"    Process RSS peak:   {mem_summary.get('process_rss_mb_peak', 0)/1024:.1f} GB", flush=True)

    # ── Pipeline Extrapolation ──
    num_attacks = len(ATTACKS) + 1  # +1 for "test"
    step_b_total = num_classes * num_samples_per_class
    step_b_per_chunk = step_b_total / total_chunks
    step_d_total = num_attacks * samples_per_attack
    step_d_per_chunk = step_d_total / total_chunks

    OVERHEAD_FACTOR = 1.15  # 15% overhead for I/O, checkpointing, etc.

    step_b_time = mean_t * step_b_total * OVERHEAD_FACTOR
    step_b_chunk_time = mean_t * step_b_per_chunk * OVERHEAD_FACTOR
    step_d_time = mean_t * step_d_total * OVERHEAD_FACTOR
    step_d_chunk_time = mean_t * step_d_per_chunk * OVERHEAD_FACTOR

    def fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m ({seconds:.0f}s)"

    def slurm_time(seconds):
        seconds = int(seconds * 1.2)  # extra 20% buffer for SLURM
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(f"\n  --- Pipeline Extrapolation (overhead={OVERHEAD_FACTOR}x) ---", flush=True)
    print(f"  Assumptions: {num_samples_per_class} samples/class, {samples_per_attack} samples/attack, "
          f"{total_chunks} chunk(s)", flush=True)
    print(f"", flush=True)
    print(f"  Step B (clean matrices):  {step_b_total} total matrices", flush=True)
    print(f"    Total time:       {fmt_time(step_b_time)}", flush=True)
    print(f"    Per chunk:        {fmt_time(step_b_chunk_time)}", flush=True)
    print(f"    SLURM --time:     {slurm_time(step_b_chunk_time)}", flush=True)
    print(f"", flush=True)
    print(f"  Step D (adv matrices):    {step_d_total} total matrices", flush=True)
    print(f"    Total time:       {fmt_time(step_d_time)}", flush=True)
    print(f"    Per chunk:        {fmt_time(step_d_chunk_time)}", flush=True)
    print(f"    SLURM --time:     {slurm_time(step_d_chunk_time)}", flush=True)
    print(f"", flush=True)
    print(f"  Recommended SLURM --mem:  480G (unified memory)", flush=True)

    # Cleanup
    del mc, model, data, labels
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "experiment": experiment_name,
        "architecture_index": architecture_index,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
        "total_positions": total_positions,
        "total_params": total_params,
        "batch_size": batch_size,
        "num_matrices": num_matrices,
        "total_wall_time_sec": round(total_bench_time, 2),
        "throughput_matrices_per_sec": round(num_matrices / total_bench_time, 4),
        "per_matrix_seconds": {
            "mean": round(mean_t, 4),
            "std": round(std_t, 4),
            "min": round(min_t, 4),
            "max": round(max_t, 4),
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
        },
        "memory": mem_summary,
        "pipeline_extrapolation": {
            "overhead_factor": OVERHEAD_FACTOR,
            "num_samples_per_class": num_samples_per_class,
            "samples_per_attack": samples_per_attack,
            "total_chunks": total_chunks,
            "step_b_total_matrices": step_b_total,
            "step_b_total_seconds": round(step_b_time, 1),
            "step_b_per_chunk_seconds": round(step_b_chunk_time, 1),
            "step_b_slurm_time": slurm_time(step_b_chunk_time),
            "step_d_total_matrices": step_d_total,
            "step_d_total_seconds": round(step_d_time, 1),
            "step_d_per_chunk_seconds": round(step_d_chunk_time, 1),
            "step_d_slurm_time": slurm_time(step_d_chunk_time),
        },
        "per_matrix_times_raw": [round(t, 4) for t in per_matrix_times],
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate experiments
    for exp in args.experiments:
        if exp not in DEFAULT_EXPERIMENTS:
            print(f"ERROR: '{exp}' not in DEFAULT_EXPERIMENTS", flush=True)
            sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.", flush=True)
        sys.exit(1)

    device = get_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory

    from utils.unified_memory import is_unified_memory_active, get_total_memory_bytes
    unified = is_unified_memory_active()
    total_mem = get_total_memory_bytes(device)

    print("=" * 70, flush=True)
    print("  GH200 MATRIX COMPUTATION BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(f"  Date:          {datetime.now().isoformat()}", flush=True)
    print(f"  GPU:           {gpu_name}", flush=True)
    print(f"  GPU HBM:       {gpu_mem / 1e9:.1f} GB", flush=True)
    print(f"  Unified mem:   {'YES' if unified else 'NO'} ({total_mem / 1e9:.1f} GB total)", flush=True)
    print(f"  Experiments:   {', '.join(args.experiments)}", flush=True)
    print(f"  Matrices/exp:  {args.num_matrices}", flush=True)
    print(f"  Max batch_size:{args.max_batch_size}", flush=True)
    print(f"  Extrapolation: {args.num_samples_per_class} samples/class, "
          f"{args.samples_per_attack} samples/attack, {args.total_chunks} chunk(s)", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    overall_start = time.perf_counter()

    for exp_name in args.experiments:
        result = benchmark_experiment(
            experiment_name=exp_name,
            num_matrices=args.num_matrices,
            max_batch_size=args.max_batch_size,
            target_utilization=args.target_utilization,
            num_samples_per_class=args.num_samples_per_class,
            samples_per_attack=args.samples_per_attack,
            total_chunks=args.total_chunks,
            temp_dir=args.temp_dir,
        )
        all_results.append(result)

    overall_elapsed = time.perf_counter() - overall_start

    # ── Final Summary ──
    print(f"\n\n{'='*70}", flush=True)
    print(f"  FINAL SUMMARY — ALL EXPERIMENTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total benchmark time: {overall_elapsed/60:.1f} min ({overall_elapsed/3600:.2f} h)", flush=True)
    print(f"", flush=True)

    # Header
    print(f"  {'Experiment':<20} {'Arch':>5} {'Positions':>10} {'BS':>6} "
          f"{'Mean(s)':>8} {'P95(s)':>8} {'GPU MB':>8} "
          f"{'StepB':>12} {'StepD':>12}", flush=True)
    print(f"  {'-'*18:<20} {'---':>5} {'-'*8:>10} {'---':>6} "
          f"{'---':>8} {'---':>8} {'---':>8} "
          f"{'---':>12} {'---':>12}", flush=True)

    for r in all_results:
        ext = r["pipeline_extrapolation"]
        step_b_h = ext["step_b_per_chunk_seconds"] / 3600
        step_d_h = ext["step_d_per_chunk_seconds"] / 3600
        print(f"  {r['experiment']:<20} {r['architecture_index']:>5} {r['total_positions']:>10} "
              f"{r['batch_size']:>6} {r['per_matrix_seconds']['mean']:>8.3f} "
              f"{r['per_matrix_seconds']['p95']:>8.3f} "
              f"{r['memory'].get('gpu_mem_mb_peak', 0):>8.0f} "
              f"{step_b_h:>11.1f}h {step_d_h:>11.1f}h", flush=True)

    print(f"", flush=True)
    print(f"  Pipeline time estimates (per chunk, with 1.15x overhead + 1.2x SLURM buffer):", flush=True)
    print(f"", flush=True)
    for r in all_results:
        ext = r["pipeline_extrapolation"]
        print(f"  {r['experiment']}:", flush=True)
        print(f"    Step B ({ext['step_b_total_matrices']} matrices): "
              f"SLURM --time={ext['step_b_slurm_time']}  --mem=480G  --gres=gpu:1", flush=True)
        print(f"    Step D ({ext['step_d_total_matrices']} matrices): "
              f"SLURM --time={ext['step_d_slurm_time']}  --mem=480G  --gres=gpu:1", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  BENCHMARK COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Save JSON ──
    report = {
        "benchmark_date": datetime.now().isoformat(),
        "gpu_name": gpu_name,
        "gpu_hbm_bytes": gpu_mem,
        "unified_memory": unified,
        "total_memory_bytes": total_mem,
        "total_benchmark_seconds": round(overall_elapsed, 2),
        "experiments": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
