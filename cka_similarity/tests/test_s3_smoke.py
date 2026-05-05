import torch
import pytest


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
