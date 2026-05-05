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

            # Per-measure aggregation
            measure_results = {}
            measure_names = list(torch.load(chunk_files[0])["accumulators"].keys())
            for mname in measure_names:
                chunk_accs = [torch.load(f)["accumulators"][mname] for f in chunk_files]
                result = _measure_finalize(mname, chunk_accs)
                measure_results[mname] = {"value": result.value, "extras": result.extras}
            out[(arch, tid)] = measure_results
    return out


def aggregate_s2(s2_dir: str, archs: List[str], num_chunks: int) -> Dict:
    """For each unordered arch pair, aggregate KM Frobenius distances + D1/D2 panels."""
    from itertools import combinations
    short_names = {"resnet152": "RN", "densenet121": "DN", "googlenet": "GN"}
    out = {}
    for a, b in combinations(archs, 2):
        pname = f"{short_names[a]}_{short_names[b]}"

        # KM Frobenius — concatenate per-chunk lists
        km_files = sorted(Path(s2_dir).glob(f"{pname}_KM_chunk*.json"))
        all_distances = []
        for f in km_files:
            all_distances.extend(json.loads(Path(f).read_text()))

        # D1 panel
        d1_files = sorted(Path(s2_dir).glob(f"{pname}_D1_chunk*.pt"))
        d1_results = {}
        if d1_files:
            d1_measure_names = list(torch.load(d1_files[0])["accumulators"].keys())
            for mname in d1_measure_names:
                chunk_accs = [torch.load(f)["accumulators"][mname] for f in d1_files]
                r = _measure_finalize(mname, chunk_accs)
                d1_results[mname] = {"value": r.value, "extras": r.extras}

        # D2 panel
        d2_files = sorted(Path(s2_dir).glob(f"{pname}_D2_chunk*.pt"))
        d2_results = {}
        if d2_files:
            d2_measure_names = list(torch.load(d2_files[0])["accumulators"].keys())
            for mname in d2_measure_names:
                chunk_accs = [torch.load(f)["accumulators"][mname] for f in d2_files]
                r = _measure_finalize(mname, chunk_accs)
                d2_results[mname] = {"value": r.value, "extras": r.extras}

        out[pname] = {
            "km_distances": all_distances,
            "km_mean": sum(all_distances) / max(1, len(all_distances)),
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

            # Panel accumulators
            panel_files = sorted(Path(s3_dir).glob(f"{arch}_{attack}_panel_chunk*.pt"))
            panel_results = {}
            if panel_files:
                measure_names = list(torch.load(panel_files[0])["accumulators"].keys())
                for mname in measure_names:
                    chunk_accs = [torch.load(f)["accumulators"][mname] for f in panel_files]
                    r = _measure_finalize(mname, chunk_accs)
                    panel_results[mname] = {"value": r.value, "extras": r.extras}

            out[(arch, attack)] = {
                "per_pair": all_pairs,
                "n_pairs": len(all_pairs),
                "panel": panel_results,
            }
    return out
