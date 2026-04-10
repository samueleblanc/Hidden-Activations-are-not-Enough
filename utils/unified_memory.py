"""
GH200 Unified Memory Initialization via RMM (RAPIDS Memory Manager).

The GH200 has ~600GB unified CPU+GPU memory (144GB HBM3e + 465GB LPDDR5X)
connected via NVLink C2C at 900 GB/s. By replacing PyTorch's CUDA allocator
with RMM's ManagedMemoryResource (cudaMallocManaged), all tensor allocations
become unified memory — the CUDA driver migrates pages automatically.

MUST be called before any torch.cuda operation (including torch.cuda.is_available()).

Disable with: export DISABLE_UNIFIED_MEMORY=1
"""

import os

_unified_memory_initialized = False


def init_unified_memory():
    """Initialize RMM managed memory for GH200 unified CPU+GPU memory pool.

    Returns True if unified memory was successfully enabled, False otherwise.
    """
    if os.environ.get("DISABLE_UNIFIED_MEMORY", "0") == "1":
        return False

    try:
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator
        import torch

        managed = rmm.mr.ManagedMemoryResource()
        pool = rmm.mr.PoolMemoryResource(
            managed,
            initial_pool_size=2**30,   # 1 GB initial
            maximum_pool_size=2**40,   # 1 TB max (covers full unified pool)
        )
        rmm.mr.set_current_device_resource(pool)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        global _unified_memory_initialized
        _unified_memory_initialized = True
        print("[RMM] Unified memory enabled", flush=True)
        return True
    except ImportError:
        print("[RMM] rmm not installed — using default CUDA allocator", flush=True)
        return False
    except RuntimeError as e:
        print(f"[RMM] Failed to initialize: {e}", flush=True)
        return False


def is_unified_memory_active():
    """Check if RMM unified memory was successfully initialized."""
    return _unified_memory_initialized


def get_system_memory_bytes():
    """Read total system memory from /proc/meminfo (Linux) or sysctl (macOS)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except FileNotFoundError:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def get_total_memory_bytes(device=None):
    """Return total available memory for GPU computation.

    With unified memory active: system_memory + gpu_hbm (the full unified pool).
    Without unified memory: gpu_hbm only.
    """
    import torch

    gpu_mem = torch.cuda.get_device_properties(device).total_memory

    if is_unified_memory_active():
        sys_mem = get_system_memory_bytes()
        if sys_mem > 0:
            # On GH200, /proc/meminfo reports LPDDR5X only (~465GB), not HBM.
            # The sum gives the full unified pool. If the OS ever includes HBM
            # in MemTotal, this would overcount — but current GH200 kernels don't.
            total = sys_mem + gpu_mem
            print(f"[RMM] Unified memory pool: {total / 1e9:.1f} GB "
                  f"(GPU HBM: {gpu_mem / 1e9:.1f} GB + System: {sys_mem / 1e9:.1f} GB)",
                  flush=True)
            return total

    return gpu_mem
