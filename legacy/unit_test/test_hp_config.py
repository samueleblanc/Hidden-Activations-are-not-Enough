"""Tests for slurm.hp_config shared configuration module."""
import pytest
from unittest.mock import MagicMock

from slurm.hp_config import (
    ARCHITECTURES,
    DATASETS,
    DATASET_NUM_CLASSES,
    ALL_CONFIGS,
    arch_dataset_from_config,
    MIG_TIERS,
    next_mig_tier,
    mig_resources,
    get_train_transform,
    get_test_transform,
    sample_hyperparameters,
    EXIT_CUDA_OOM,
)


class TestAllConfigs:
    def test_all_configs_generated(self):
        """ALL_CONFIGS has 27 entries (9 arch x 3 datasets)."""
        assert len(ALL_CONFIGS) == 27

    def test_config_names_format(self):
        """All config names contain a dataset suffix."""
        for config_name in ALL_CONFIGS:
            assert any(config_name.endswith(ds) for ds in DATASETS), (
                f"Config '{config_name}' does not end with a known dataset"
            )


class TestMigTiers:
    def test_mig_tiers_ordered(self):
        """4 tiers with increasing GPU memory."""
        assert len(MIG_TIERS) == 4
        memories = [tier["gpu_mem_gb"] for tier in MIG_TIERS]
        assert memories == sorted(memories), "MIG tiers should be ordered by GPU memory"
        assert all(memories[i] < memories[i + 1] for i in range(len(memories) - 1))

    def test_next_mig_tier(self):
        """H100-1g.10gb -> H100-2g.20gb -> ... -> None at max."""
        assert next_mig_tier("H100-1g.10gb") == "H100-2g.20gb"
        assert next_mig_tier("H100-2g.20gb") == "H100-3g.40gb"
        assert next_mig_tier("H100-3g.40gb") == "H100-80gb"
        assert next_mig_tier("H100-80gb") is None

    def test_mig_tier_resources(self):
        """Returns correct gres, cpus, mem for each tier."""
        t1 = mig_resources("H100-1g.10gb")
        assert t1["gres"] == "gpu:h100:1g.10gb:1"
        assert t1["gpu_mem_gb"] == 10
        assert t1["cpus"] == 2
        assert t1["mem"] == "15G"

        t2 = mig_resources("H100-2g.20gb")
        assert t2["gres"] == "gpu:h100:2g.20gb:1"
        assert t2["gpu_mem_gb"] == 20
        assert t2["cpus"] == 4
        assert t2["mem"] == "31G"

        t3 = mig_resources("H100-3g.40gb")
        assert t3["gres"] == "gpu:h100:3g.40gb:1"
        assert t3["gpu_mem_gb"] == 40
        assert t3["cpus"] == 8
        assert t3["mem"] == "62G"

        t4 = mig_resources("H100-80gb")
        assert t4["gres"] == "gpu:h100:1"
        assert t4["gpu_mem_gb"] == 80
        assert t4["cpus"] == 16
        assert t4["mem"] == "124G"

    def test_mig_resources_unknown_tier(self):
        """Unknown tier name returns None."""
        assert mig_resources("nonexistent") is None


class TestTransforms:
    def test_get_train_transform(self):
        """Returns non-None for cifar10, cifar100, tiny_imagenet."""
        for ds in ["cifar10", "cifar100", "tiny_imagenet"]:
            t = get_train_transform(ds)
            assert t is not None, f"get_train_transform('{ds}') returned None"

    def test_get_test_transform(self):
        """Returns non-None for all datasets."""
        for ds in ["cifar10", "cifar100", "tiny_imagenet"]:
            t = get_test_transform(ds)
            assert t is not None, f"get_test_transform('{ds}') returned None"


class TestSearchSpace:
    def test_hp_search_space_keys(self):
        """sample_hyperparameters returns dict with all expected p1_ and p2_ keys."""
        trial = MagicMock()
        # Configure suggest methods to return valid values
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        trial.suggest_float.side_effect = lambda name, low, high, **kw: low
        trial.suggest_int.side_effect = lambda name, low, high: low

        params = sample_hyperparameters(trial)
        assert isinstance(params, dict)

        # Phase 1 keys
        expected_p1 = {"p1_optimizer", "p1_lr", "p1_epochs", "p1_weight_decay"}
        # Phase 2 keys
        expected_p2 = {
            "p2_optimizer",
            "p2_lr",
            "p2_weight_decay",
            "p2_scheduler",
            "p2_max_epochs",
            "p2_patience",
        }

        for key in expected_p1:
            assert key in params, f"Missing key: {key}"
        for key in expected_p2:
            assert key in params, f"Missing key: {key}"


class TestArchDatasetFromConfig:
    def test_simple_config(self):
        """Correctly parses 'resnet18_cifar10'."""
        arch, ds = arch_dataset_from_config("resnet18_cifar10")
        assert arch == "resnet18"
        assert ds == "cifar10"

    def test_bn_config(self):
        """Correctly parses 'vgg16_bn_cifar100'."""
        arch, ds = arch_dataset_from_config("vgg16_bn_cifar100")
        assert arch == "vgg16_bn"
        assert ds == "cifar100"

    def test_tiny_imagenet_config(self):
        """Correctly parses 'resnet152_tiny_imagenet'."""
        arch, ds = arch_dataset_from_config("resnet152_tiny_imagenet")
        assert arch == "resnet152"
        assert ds == "tiny_imagenet"

    def test_vgg_bn_tiny_imagenet(self):
        """Correctly parses 'vgg19_bn_tiny_imagenet'."""
        arch, ds = arch_dataset_from_config("vgg19_bn_tiny_imagenet")
        assert arch == "vgg19_bn"
        assert ds == "tiny_imagenet"


class TestConstants:
    def test_exit_cuda_oom(self):
        assert EXIT_CUDA_OOM == 42

    def test_dataset_num_classes(self):
        assert DATASET_NUM_CLASSES["cifar10"] == 10
        assert DATASET_NUM_CLASSES["cifar100"] == 100
        assert DATASET_NUM_CLASSES["tiny_imagenet"] == 200
