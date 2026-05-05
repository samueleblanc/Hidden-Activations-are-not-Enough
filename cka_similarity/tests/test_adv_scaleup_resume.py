"""Adversarial scale-up checkpointing contract: a partially-completed run
must be resumable, with the resumed run extending the existing tensor
rather than overwriting it.
"""
import os
import torch
import pytest


def test_resume_from_partial_tensor(tmp_path):
    from generate_adversarial_pairs_scaleup import resume_state

    # Simulate a partially-completed run
    path = tmp_path / "fgsm" / "pairs.pth"
    path.parent.mkdir(parents=True)
    partial = {
        "x_clean": torch.randn(100, 3, 224, 224),
        "x_adv":   torch.randn(100, 3, 224, 224),
        "y_clean": torch.randint(0, 1000, (100,)),
        "y_adv":   torch.randint(0, 1000, (100,)),
        "n_done":  100,
        "attack_kwargs": {"eps": 8/255},
    }
    torch.save(partial, str(path))

    state = resume_state(str(path))
    assert state["n_done"] == 100
    assert state["x_clean"].shape[0] == 100


def test_no_partial_returns_none(tmp_path):
    from generate_adversarial_pairs_scaleup import resume_state
    state = resume_state(str(tmp_path / "missing.pth"))
    assert state is None
