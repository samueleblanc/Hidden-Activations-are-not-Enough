"""Tests for slurm.train_trial training trial script."""
import pytest
import torch
import torch.nn as nn

from slurm.train_trial import (
    load_pretrained_model,
    freeze_backbone,
    unfreeze_all,
    build_optimizer,
    build_scheduler,
)


# ---------------------------------------------------------------------------
# load_pretrained_model
# ---------------------------------------------------------------------------


class TestLoadPretrainedModel:
    def test_load_pretrained_model_resnet18(self):
        """Load resnet18 with num_classes=10, verify model.fc.out_features == 10."""
        model = load_pretrained_model("resnet18", num_classes=10)
        assert model.fc.out_features == 10

    def test_load_pretrained_model_vgg11_bn(self):
        """Load vgg11_bn with num_classes=100, verify last classifier output."""
        model = load_pretrained_model("vgg11_bn", num_classes=100)
        assert model.classifier[-1].out_features == 100

    def test_load_pretrained_model_resnet152(self):
        """Load resnet152 with num_classes=200, verify model.fc.out_features == 200."""
        model = load_pretrained_model("resnet152", num_classes=200)
        assert model.fc.out_features == 200


# ---------------------------------------------------------------------------
# freeze_backbone / unfreeze_all
# ---------------------------------------------------------------------------


class TestFreezeBackbone:
    def test_freeze_backbone_resnet(self):
        """Freeze backbone: conv layers frozen, fc trainable. Then unfreeze all."""
        model = load_pretrained_model("resnet18", num_classes=10)
        freeze_backbone(model, "resnet18")

        # fc (head) should be trainable
        for p in model.fc.parameters():
            assert p.requires_grad is True, "fc should be trainable after freeze_backbone"

        # At least one conv layer should be frozen
        frozen_count = sum(
            1 for p in model.parameters() if not p.requires_grad
        )
        assert frozen_count > 0, "Some parameters should be frozen"

        # Unfreeze all
        unfreeze_all(model)
        for p in model.parameters():
            assert p.requires_grad is True, "All params should be trainable after unfreeze_all"

    def test_freeze_backbone_vgg(self):
        """Freeze backbone: features frozen, last classifier layer trainable."""
        model = load_pretrained_model("vgg11_bn", num_classes=100)
        freeze_backbone(model, "vgg11_bn")

        # features should be frozen
        for p in model.features.parameters():
            assert p.requires_grad is False, "features should be frozen"

        # Last classifier layer should be trainable
        last_layer = model.classifier[-1]
        for p in last_layer.parameters():
            assert p.requires_grad is True, "classifier[-1] should be trainable"


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_build_optimizer_adam(self):
        """Build Adam optimizer."""
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, "adam", lr=1e-3, weight_decay=1e-4)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["lr"] == 1e-3

    def test_build_optimizer_sgd_with_momentum(self):
        """Build SGD with momentum."""
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, "sgd", lr=1e-2, weight_decay=1e-4, momentum=0.9)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["lr"] == 1e-2
        assert opt.defaults["momentum"] == 0.9

    def test_build_optimizer_adamw(self):
        """Build AdamW optimizer."""
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, "adamw", lr=1e-3, weight_decay=1e-2)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["lr"] == 1e-3


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_build_scheduler_cosine(self):
        """Build CosineAnnealingLR scheduler."""
        model = nn.Linear(10, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler(opt, "cosine", max_epochs=50)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_scheduler_step(self):
        """Build StepLR scheduler."""
        model = nn.Linear(10, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler(opt, "step", max_epochs=30)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
