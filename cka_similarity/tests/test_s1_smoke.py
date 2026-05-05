"""S1 worker smoke test: small synthetic input, mock teleportation, verify
the per-chunk accumulator file is written with the expected keys."""
import os
import torch
import pytest


def test_s1_writes_accumulator_per_chunk(tmp_path, monkeypatch):
    from cka_similarity.workers import s1_within_arch_invariance as s1

    # Mock the model loader and teleporter to keep the test fast
    class _FakeModel(torch.nn.Module):
        def __init__(self, p=64):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, p)
        def forward(self, x):
            return self.fc(x.flatten(1))

    fake_model = _FakeModel()
    monkeypatch.setattr(s1, "load_pretrained", lambda arch: fake_model)
    monkeypatch.setattr(s1, "teleport", lambda model, seed: fake_model)
    monkeypatch.setattr(s1, "forward_penultimate", lambda m, x: m(x))
    monkeypatch.setattr(s1, "forward_logits", lambda m, x: m(x))

    fake_inputs = torch.randn(10, 3, 224, 224)
    monkeypatch.setattr(s1, "load_imagenet_val_chunk", lambda start, end, data_dir: fake_inputs[:end-start])

    out_dir = tmp_path / "results" / "phase1" / "s1"
    s1.run_chunk(
        chunk_id=0, num_chunks=2, num_samples_total=10,
        archs=["resnet152"], num_teleports=2,
        out_dir=str(out_dir), data_dir="/dummy",
    )

    # Should produce 1 arch × 2 teleports = 2 files
    files = list(out_dir.glob("*.pt"))
    assert len(files) == 2
    acc = torch.load(files[0])
    assert "accumulators" in acc
    # Some measure stores its block under either A_block or B_block
    any_acc = next(iter(acc["accumulators"].values()))
    assert isinstance(any_acc, dict)
