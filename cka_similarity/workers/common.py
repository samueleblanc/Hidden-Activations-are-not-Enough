"""Shared utilities for the Phase 1 sub-study workers."""
import json
import os
from typing import Tuple


def chunk_slice(chunk_id: int, total: int, num_chunks: int) -> Tuple[int, int]:
    """Compute the [start, end) slice index for a SLURM array task.

    Last chunk absorbs any remainder so all samples are covered exactly once.
    """
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range [0, {num_chunks})")
    base_size = total // num_chunks
    start = chunk_id * base_size
    end = (chunk_id + 1) * base_size if chunk_id < num_chunks - 1 else total
    return start, end


def load_active_km_batch_size(calibration_path: str) -> int:
    """Read calibration.json and return the active-tier KM batch size."""
    with open(calibration_path) as f:
        data = json.load(f)
    return data["tiers"][data["active_tier"]]["km_batch_size"]


def calibration_path_for(arch: str) -> str:
    return f"experiments/calibration/{arch}_imagenet/calibration.json"


def imagenet_val_subset_indices(n: int = 25000) -> list:
    """The fixed first-N-by-sorted-filename ImageNet val sample set used in Phase 1.

    Indices into the dataset's sorted-by-filename order. Reproducibility seed:
    we always use the first 25K validation images by sorted filename.
    """
    return list(range(n))
