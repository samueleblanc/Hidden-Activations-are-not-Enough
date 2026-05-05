import torch
import pytest


def test_s2_emits_km_distance_lists_and_d1_d2_accumulators(tmp_path, monkeypatch):
    from cka_similarity.workers import s2_cross_architecture as s2

    # Mock the heavy bits
    class _FakeModel(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.fc = torch.nn.Linear(3 * 224 * 224, p)
        def forward(self, x): return self.fc(x.flatten(1))

    fake_models = {
        "resnet152":   _FakeModel(p=2048),
        "densenet121": _FakeModel(p=1024),
        "googlenet":   _FakeModel(p=1024),
    }
    monkeypatch.setattr(s2, "load_pretrained", lambda arch: fake_models[arch])
    monkeypatch.setattr(s2, "forward_penultimate", lambda m, x: m(x))
    monkeypatch.setattr(s2, "forward_logits", lambda m, x: m(x)[:, :1000] if m(x).shape[1] >= 1000 else torch.cat([m(x), torch.zeros(m(x).shape[0], 1000 - m(x).shape[1])], dim=1))

    # Fake KM: small uniform per-sample tensor
    monkeypatch.setattr(s2, "extract_km_per_sample",
                        lambda model, x, batch_size: torch.randn(x.shape[0], 1000, 100))

    fake_inputs = torch.randn(10, 3, 224, 224)
    monkeypatch.setattr(s2, "load_imagenet_val_chunk", lambda *args, **kwargs: fake_inputs)
    monkeypatch.setattr(s2, "load_active_km_batch_size", lambda path: 64)

    out_dir = tmp_path / "results" / "phase1" / "s2"
    s2.run_chunk(
        chunk_id=0, num_chunks=1, num_samples_total=10,
        archs=["resnet152", "densenet121", "googlenet"],
        out_dir=str(out_dir), data_dir="/dummy",
    )

    # Should produce 3 KM-distance JSON list files (one per arch pair)
    pair_files = list(out_dir.glob("*_KM_chunk*.json"))
    assert len(pair_files) == 3   # RN_DN, RN_GN, DN_GN
    # Should produce penultimate-panel accumulator files
    panel_files = list(out_dir.glob("*_D1_chunk*.pt"))
    assert len(panel_files) == 3
