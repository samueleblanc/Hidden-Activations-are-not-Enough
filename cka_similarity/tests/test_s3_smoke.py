import torch
import pytest


def _fake_s3_env(s3, monkeypatch):
    """Shared mock setup for S3 unit tests."""
    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, 1000)
        def forward(self, x): return self.fc(x.flatten(1))

    fake_model = _FakeModel()
    monkeypatch.setattr(s3, "load_pretrained", lambda arch: fake_model)
    monkeypatch.setattr(s3, "forward_penultimate", lambda m, x: m(x)[:, :64])
    monkeypatch.setattr(s3, "forward_logits", lambda m, x: m(x))
    monkeypatch.setattr(s3, "extract_km_per_sample", lambda model, x, batch_size: torch.randn(x.shape[0], 1000, 100))
    monkeypatch.setattr(s3, "load_active_km_batch_size", lambda path: 32)
    return fake_model


def test_s3_emits_per_pair_distances_and_panel_accumulators(tmp_path, monkeypatch):
    from cka_similarity.workers import s3_distance_amplification as s3

    class _FakeModel(torch.nn.Module):
        def __init__(self, p=64):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, 1000)
        def forward(self, x): return self.fc(x.flatten(1))

    fake_model = _FakeModel()
    monkeypatch.setattr(s3, "load_pretrained", lambda arch: fake_model)
    monkeypatch.setattr(s3, "forward_penultimate", lambda m, x: m(x)[:, :64])
    monkeypatch.setattr(s3, "forward_logits", lambda m, x: m(x))
    monkeypatch.setattr(s3, "extract_km_per_sample", lambda model, x, batch_size: torch.randn(x.shape[0], 1000, 100))
    monkeypatch.setattr(s3, "load_active_km_batch_size", lambda path: 32)

    # Mock adversarial pairs file
    pairs = {
        "x_clean": torch.randn(20, 3, 224, 224),
        "x_adv":   torch.randn(20, 3, 224, 224),
        "y_clean": torch.randint(0, 1000, (20,)),
        "y_adv":   torch.randint(0, 1000, (20,)),
        "n_done":  20,
    }
    pairs_dir = tmp_path / "experiments" / "resnet152_imagenet" / "adversarial_pairs_N5000" / "fgsm"
    pairs_dir.mkdir(parents=True)
    torch.save(pairs, str(pairs_dir / "pairs.pth"))

    out_dir = tmp_path / "results" / "phase1" / "s3"
    s3.run_chunk(
        chunk_id=0, num_chunks=1, total_pairs_per_attack=20,
        archs=["resnet152"], attacks=["fgsm"],
        out_dir=str(out_dir),
        pairs_root=str(tmp_path / "experiments"),
    )

    files = list(out_dir.glob("*.json")) + list(out_dir.glob("*.pt"))
    assert len(files) >= 2


def test_s3_writes_empty_markers_for_chunks_past_n_done(tmp_path, monkeypatch):
    """Partial pairs.pth (deepfool: n_done < target_n) → chunks past n_done
    must emit empty markers so the .complete sentinel can fire and the
    orchestrator stops re-queueing S3.
    """
    from cka_similarity.workers import s3_distance_amplification as s3

    _fake_s3_env(s3, monkeypatch)

    pairs = {
        "x_clean": torch.randn(10, 3, 224, 224),
        "x_adv":   torch.randn(10, 3, 224, 224),
        "y_clean": torch.randint(0, 1000, (10,)),
        "y_adv":   torch.randint(0, 1000, (10,)),
        "n_done":  10,  # partial — target was 100
    }
    pairs_dir = tmp_path / "experiments" / "resnet152_imagenet" / "adversarial_pairs_N5000" / "deepfool"
    pairs_dir.mkdir(parents=True)
    torch.save(pairs, str(pairs_dir / "pairs.pth"))

    out_dir = tmp_path / "results" / "phase1" / "s3"

    # Chunk 0: pairs [0, 50) — clamped to 10, real data
    s3.run_chunk(
        chunk_id=0, num_chunks=2, total_pairs_per_attack=100,
        archs=["resnet152"], attacks=["deepfool"],
        out_dir=str(out_dir), pairs_root=str(tmp_path / "experiments"),
    )
    # Chunk 1: pairs [50, 100) — entirely past n_done; should emit empty markers
    s3.run_chunk(
        chunk_id=1, num_chunks=2, total_pairs_per_attack=100,
        archs=["resnet152"], attacks=["deepfool"],
        out_dir=str(out_dir), pairs_root=str(tmp_path / "experiments"),
    )

    # Both chunk JSONs and both panel .pt files must exist
    for ci in (0, 1):
        assert (out_dir / f"resnet152_deepfool_chunk{ci}.json").exists(), f"missing dist chunk{ci}"
        assert (out_dir / f"resnet152_deepfool_panel_chunk{ci}.pt").exists(), f"missing panel chunk{ci}"

    # Chunk 1 must be marked n_pairs=0 with empty accumulators dict
    empty_panel = torch.load(out_dir / "resnet152_deepfool_panel_chunk1.pt")
    assert empty_panel["n_pairs"] == 0
    assert empty_panel["accumulators"] == {}

    # Sentinel must now fire (full grid is on disk)
    assert (out_dir / ".complete").exists(), "expected .complete sentinel after empty markers"


def test_aggregate_s3_filters_empty_panel_chunks(tmp_path):
    """aggregate_s3 must ignore n_pairs=0 panel chunks (deepfool partial state)
    so the per-(arch, attack) finalize uses only the real-data chunks.
    """
    from cka_similarity.reduce.aggregate import aggregate_s3
    from cka_similarity.measures.panel import PANEL

    s3_dir = tmp_path / "s3"
    s3_dir.mkdir()

    # Real data chunk 0 — build accumulators from one of the panel measures
    real_measure = list(PANEL)[0]()
    acc0 = real_measure.accumulate(torch.randn(5, 64), torch.randn(5, 64))
    torch.save({
        "chunk_id": 0, "arch": "resnet152", "attack": "deepfool",
        "n_pairs": 5, "accumulators": {real_measure.name: acc0},
    }, str(s3_dir / "resnet152_deepfool_panel_chunk0.pt"))
    (s3_dir / "resnet152_deepfool_chunk0.json").write_text("[]")

    # Empty marker chunk 1 — accumulators={} would crash the old reducer
    torch.save({
        "chunk_id": 1, "arch": "resnet152", "attack": "deepfool",
        "n_pairs": 0, "accumulators": {},
    }, str(s3_dir / "resnet152_deepfool_panel_chunk1.pt"))
    (s3_dir / "resnet152_deepfool_chunk1.json").write_text("[]")

    out = aggregate_s3(str(s3_dir), archs=["resnet152"], attacks=["deepfool"], num_chunks=2)
    assert ("resnet152", "deepfool") in out
    # Panel results must include the real measure name (not blanked by the empty chunk)
    assert real_measure.name in out[("resnet152", "deepfool")]["panel"]
