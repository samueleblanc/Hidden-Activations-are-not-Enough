"""Phase-1 Step C — aggregate chunk artifacts → final per-cell measure values."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from cka_similarity.measures.panel import PANEL


def _measure_finalize(measure_name, chunk_accs: List):
    """Find the right MeasureBase subclass by name and call finalize."""
    for cls in PANEL:
        m = cls()
        if m.name == measure_name:
            return m.finalize(chunk_accs)
    raise ValueError(f"Unknown measure: {measure_name}")


def _finalize_panel(panel_files: List) -> Dict:
    """Load each chunk file once, then finalize every measure in its panel.

    The previous per-measure ``torch.load`` re-read every file once per
    measure (×9 I/O — ~4.3 TB of redundant deserialization across the 150
    S1 combos), risking the reduce walltime. Holding one combo's 64 chunk
    files (~3 GB of feature blocks) is well within the job's memory.
    """
    loaded = [torch.load(f)["accumulators"] for f in panel_files]
    results = {}
    for mname in loaded[0].keys():
        chunk_accs = [d[mname] for d in loaded]
        r = _measure_finalize(mname, chunk_accs)
        results[mname] = {"value": r.value, "extras": r.extras}
    return results


def aggregate_s1(s1_dir: str, archs: List[str], num_teleports: int, num_chunks: int) -> Dict:
    """For each (arch, teleport_id), glob chunk files and finalize each measure.

    Returns: {(arch, teleport_id): {measure_name: {"value": ..., "extras": ...}}}
    """
    out = {}
    for arch in archs:
        for tid in range(num_teleports):
            chunk_files = sorted(Path(s1_dir).glob(f"{arch}_teleport{tid}_chunk*.pt"))
            if len(chunk_files) != num_chunks:
                print(f"WARN: {arch} teleport {tid} has {len(chunk_files)}/{num_chunks} chunks")
                if not chunk_files:
                    continue

            out[(arch, tid)] = _finalize_panel(chunk_files)
    return out


def aggregate_s2(s2_dir: str, archs: List[str], num_chunks: int) -> Dict:
    """For each unordered arch pair, aggregate KM Frobenius distances + D1/D2 panels.

    Per-pair KM Frobenius distances are stored on disk in both raw units
    ({pname}_KM_chunk*.json) and per-coordinate RMS units
    ({pname}_KM_rms_chunk*.json — written by
    cka_similarity/workers/s2_cross_architecture.py since 2026-05-10).
    If RMS files are missing (pre-2026-05-10 chunks), fall back to deriving
    RMS values from raw using utils.scaling.km_numel — this preserves the
    canonical metric across legacy and fresh data.
    """
    from itertools import combinations
    from utils.scaling import (
        IMAGENET_INPUT_NUMEL, IMAGENET_NUM_CLASSES, km_numel,
    )
    short_names = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}
    s_KM = 1.0 / (km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL) ** 0.5)
    out = {}
    for a, b in combinations(archs, 2):
        pname = f"{short_names[a]}_{short_names[b]}"

        # KM Frobenius — concatenate per-chunk lists (raw)
        km_files = sorted(Path(s2_dir).glob(f"{pname}_KM_chunk*.json"))
        all_distances = []
        for f in km_files:
            all_distances.extend(json.loads(Path(f).read_text()))

        # KM Frobenius — RMS variant. Always derived from the raw list: rms is
        # the deterministic rescale ``raw * s_KM`` (the s2 worker writes exactly
        # ``[v * s_KM for v in km_distances]``). We deliberately do NOT read the
        # on-disk ``*_KM_rms`` files: a partially-regenerated rms set (observed
        # 2026-06-06: RN_DN had 39/64 rms files after a cross-version rerun while
        # raw had 64/64) would otherwise be aggregated over fewer chunks than the
        # raw list, silently desyncing km_mean_rms from km_mean. Deriving from
        # raw keeps the two in lock-step by construction.
        all_distances_rms = [d * s_KM for d in all_distances]

        # D1 panel
        d1_files = sorted(Path(s2_dir).glob(f"{pname}_D1_chunk*.pt"))
        d1_results = _finalize_panel(d1_files) if d1_files else {}

        # D2 panel
        d2_files = sorted(Path(s2_dir).glob(f"{pname}_D2_chunk*.pt"))
        d2_results = _finalize_panel(d2_files) if d2_files else {}

        out[pname] = {
            "km_distances": all_distances,
            "km_distances_rms": all_distances_rms,
            "km_mean": sum(all_distances) / max(1, len(all_distances)),
            "km_mean_rms": sum(all_distances_rms) / max(1, len(all_distances_rms)),
            "km_n": len(all_distances),
            "D1": d1_results,
            "D2": d2_results,
        }
    return out


def aggregate_s3(s3_dir: str, archs: List[str], attacks: List[str], num_chunks: int) -> Dict:
    """For each (arch, attack), aggregate per-pair distances + panel accumulators."""
    out = {}
    for arch in archs:
        for attack in attacks:
            # Per-pair distance lists
            dist_files = sorted(Path(s3_dir).glob(f"{arch}_{attack}_chunk*.json"))
            all_pairs = []
            for f in dist_files:
                all_pairs.extend(json.loads(Path(f).read_text()))

            # Panel accumulators. S3 writes n_pairs=0 markers for chunks past
            # the pair n_done (deepfool's partial pairs.pth — see
            # workers/s3_distance_amplification.py). Filter those out before
            # finalizing measures so an empty-marker chunk0 doesn't blank the
            # measure_names list.
            panel_files = sorted(Path(s3_dir).glob(f"{arch}_{attack}_panel_chunk*.pt"))
            loaded = [torch.load(f) for f in panel_files]
            non_empty = [d for d in loaded if d.get("n_pairs", 0) > 0]
            panel_results = {}
            if non_empty:
                measure_names = list(non_empty[0]["accumulators"].keys())
                for mname in measure_names:
                    chunk_accs = [d["accumulators"][mname] for d in non_empty]
                    r = _measure_finalize(mname, chunk_accs)
                    panel_results[mname] = {"value": r.value, "extras": r.extras}

            out[(arch, attack)] = {
                "per_pair": all_pairs,
                "n_pairs": len(all_pairs),
                "panel": panel_results,
            }
    return out
