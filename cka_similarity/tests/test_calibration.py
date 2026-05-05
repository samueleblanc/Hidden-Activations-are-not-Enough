import json, os
from pathlib import Path
import pytest


def test_calibration_three_tier_schema(tmp_path):
    """Calibration JSON must have tiers 85/90/93 with positive batch sizes."""
    from bin.calibrate import write_calibration_json
    out = tmp_path / "calibration.json"
    write_calibration_json(
        out_path=str(out),
        arch="resnet152",
        gpu_name="NVIDIA H100",
        gpu_memory_bytes=80_000_000_000,
        tiers={
            "85": {"km_batch_size": 768, "peak_memory_bytes": 68_000_000_000, "fraction": 0.85},
            "90": {"km_batch_size": 896, "peak_memory_bytes": 72_000_000_000, "fraction": 0.90},
            "93": {"km_batch_size": 1024, "peak_memory_bytes": 74_000_000_000, "fraction": 0.93},
        },
        avg_seconds_per_km=0.087,
        input_shape=[3, 224, 224],
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["arch"] == "resnet152"
    assert data["active_tier"] == "93"
    assert set(data["tiers"].keys()) == {"85", "90", "93"}
    for tier in ["85", "90", "93"]:
        assert data["tiers"][tier]["km_batch_size"] > 0
        assert data["tiers"][tier]["peak_memory_bytes"] > 0


def test_calibration_active_tier_lookup(tmp_path):
    from bin.calibrate import get_active_batch_size
    out = tmp_path / "calibration.json"
    out.write_text(json.dumps({
        "arch": "resnet152",
        "tiers": {
            "85": {"km_batch_size": 768},
            "90": {"km_batch_size": 896},
            "93": {"km_batch_size": 1024},
        },
        "active_tier": "90",
    }))
    assert get_active_batch_size(str(out)) == 896


def test_calibration_step_down_tier(tmp_path):
    from bin.calibrate import step_down_tier
    out = tmp_path / "calibration.json"
    out.write_text(json.dumps({
        "active_tier": "93",
        "tiers": {"85": {}, "90": {}, "93": {}},
    }))
    new_tier = step_down_tier(str(out))
    assert new_tier == "90"
    data = json.loads(out.read_text())
    assert data["active_tier"] == "90"
    new_tier = step_down_tier(str(out))
    assert new_tier == "85"
    with pytest.raises(RuntimeError, match="below 85"):
        step_down_tier(str(out))
