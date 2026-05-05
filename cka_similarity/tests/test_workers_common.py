import pytest


def test_chunk_slice_basic():
    from cka_similarity.workers.common import chunk_slice
    # 25000 samples, 64 chunks
    start, end = chunk_slice(0, total=25000, num_chunks=64)
    assert start == 0 and end == 390

    start, end = chunk_slice(63, total=25000, num_chunks=64)
    # Last chunk gets the remainder
    assert end == 25000
    assert start == 63 * (25000 // 64)


def test_chunk_slice_uniform_division():
    from cka_similarity.workers.common import chunk_slice
    # Easy case: evenly divisible
    start, end = chunk_slice(3, total=64, num_chunks=8)
    assert start == 24 and end == 32


def test_load_active_batch_size(tmp_path):
    import json
    from cka_similarity.workers.common import load_active_km_batch_size
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "active_tier": "90",
        "tiers": {
            "85": {"km_batch_size": 768},
            "90": {"km_batch_size": 896},
            "93": {"km_batch_size": 1024},
        },
    }))
    assert load_active_km_batch_size(str(calib)) == 896
