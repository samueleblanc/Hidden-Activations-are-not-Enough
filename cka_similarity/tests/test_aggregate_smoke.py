"""Smoke test for the chunk-aggregator (Phase-1 Step C, Task 4.1)."""
import torch
import pytest


def test_aggregate_s1_sums_chunk_accumulators(tmp_path):
    from cka_similarity.reduce.aggregate import aggregate_s1
    from cka_similarity.measures.cka import DebiasedLinearCKA

    # Create 4 fake chunk files
    s1_dir = tmp_path / "s1"
    s1_dir.mkdir()
    cls = DebiasedLinearCKA
    measure = cls()
    for chunk_id in range(4):
        X = torch.randn(50, 64)
        Y = torch.randn(50, 64)
        acc = {measure.name: measure.accumulate(X, Y)}
        torch.save({
            "chunk_id": chunk_id, "arch": "resnet152", "teleport_id": 0,
            "n_samples": 50, "accumulators": acc,
        }, s1_dir / f"resnet152_teleport0_chunk{chunk_id}.pt")

    results = aggregate_s1(str(s1_dir), archs=["resnet152"], num_teleports=1, num_chunks=4)
    assert ("resnet152", 0) in results
    assert measure.name in results[("resnet152", 0)]
    assert "value" in results[("resnet152", 0)][measure.name]
