"""Tests for slurm.hp_planner hyperparameter tuning planner."""
import json
import os
import pytest

from slurm.hp_config import ALL_CONFIGS, MIG_TIERS
from slurm.hp_planner import (
    create_initial_registry,
    process_errors,
    select_jobs,
    finalize_completed_configs,
    _parse_mem,
    _format_mem,
    _parse_time,
    _format_time,
)


# ---------------------------------------------------------------------------
# create_initial_registry
# ---------------------------------------------------------------------------


class TestCreateInitialRegistry:
    def test_create_initial_registry(self):
        """Verify 27 configs x 50 trials, all status 'pending', correct MIG tier."""
        registry = create_initial_registry(
            trials_per_config=50,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        assert registry["cycle"] == 1
        assert registry["max_cycles"] == 5
        configs = registry["configs"]
        assert len(configs) == 27

        for config_name, config in configs.items():
            assert config_name in ALL_CONFIGS
            assert config["mig_tier"] == "H100-1g.10gb"
            assert config["system_mem"] == "15G"
            assert config["time_limit"] == "6:00:00"
            assert config["cpus"] == MIG_TIERS[0]["cpus"]
            assert config["target_trials"] == 50
            assert config["complete"] is False
            assert len(config["trials"]) == 50
            for trial_id, trial in config["trials"].items():
                assert trial["status"] == "pending"


# ---------------------------------------------------------------------------
# process_errors — escalation
# ---------------------------------------------------------------------------


class TestEscalateMigOnCudaOom:
    def test_escalate_mig_on_cuda_oom(self):
        """Config with a failed cuda_oom trial should escalate MIG tier."""
        registry = create_initial_registry(
            trials_per_config=50,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        # Pick first config, mark trial 0 as failed with cuda_oom
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "cuda_oom",
        }

        process_errors(registry["configs"])

        config = registry["configs"][config_name]
        assert config["mig_tier"] == "H100-2g.20gb"
        assert config["cpus"] == 4
        assert config["system_mem"] == "31G"
        # Failed trial should be marked for retry
        assert registry["configs"][config_name]["trials"]["0"]["retry"] is True


class TestDoubleMemOnSystemOom:
    def test_double_mem_on_system_oom(self):
        """Config with system_oom failure should double system_mem."""
        registry = create_initial_registry(
            trials_per_config=50,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "system_oom",
        }

        process_errors(registry["configs"])

        assert registry["configs"][config_name]["system_mem"] == "30G"
        assert registry["configs"][config_name]["trials"]["0"]["retry"] is True


class TestDoubleTimeOnTimeout:
    def test_double_time_on_timeout(self):
        """Config with timeout failure should double time_limit."""
        registry = create_initial_registry(
            trials_per_config=50,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "timeout",
        }

        process_errors(registry["configs"])

        assert registry["configs"][config_name]["time_limit"] == "12:00:00"
        assert registry["configs"][config_name]["trials"]["0"]["retry"] is True


# ---------------------------------------------------------------------------
# select_jobs
# ---------------------------------------------------------------------------


class TestSelectJobsRespectsMax:
    def test_select_jobs_respects_max(self):
        """With 50 pending trials, select_jobs(max_jobs=10) returns exactly 10."""
        registry = create_initial_registry(
            trials_per_config=50,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        jobs = select_jobs(registry["configs"], max_jobs=10)
        assert len(jobs) == 10


class TestSelectJobsRoundRobin:
    def test_select_jobs_round_robin(self):
        """Two configs with 5 pending each, max_jobs=6 -> 3 from each."""
        # Build a minimal registry with only 2 configs
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        # Keep only first 2 configs
        first_two = list(ALL_CONFIGS[:2])
        registry["configs"] = {
            k: v for k, v in registry["configs"].items() if k in first_two
        }

        jobs = select_jobs(registry["configs"], max_jobs=6)
        assert len(jobs) == 6

        # Count per config
        counts = {}
        for job in jobs:
            counts[job["config"]] = counts.get(job["config"], 0) + 1
        assert counts[first_two[0]] == 3
        assert counts[first_two[1]] == 3


class TestSelectJobsPrioritizesRetries:
    def test_select_jobs_prioritizes_retries(self):
        """Failed trial marked with retry=True should appear first in jobs list."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        # Keep only one config for clarity
        config_name = ALL_CONFIGS[0]
        registry["configs"] = {
            k: v for k, v in registry["configs"].items() if k == config_name
        }
        # Mark trial 3 as needing retry
        registry["configs"][config_name]["trials"]["3"] = {
            "status": "failed",
            "error": "timeout",
            "retry": True,
        }

        jobs = select_jobs(registry["configs"], max_jobs=5)
        # First job should be the retry (trial 3)
        assert jobs[0]["trial_id"] == "3"
        assert jobs[0]["config"] == config_name


class TestSelectJobsSkipsCompleteConfigs:
    def test_select_jobs_skips_complete_configs(self):
        """Config with complete=True should produce 0 jobs."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        # Mark all configs as complete
        for config in registry["configs"].values():
            config["complete"] = True

        jobs = select_jobs(registry["configs"], max_jobs=100)
        assert len(jobs) == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_mem(self):
        assert _parse_mem("15G") == 15
        assert _parse_mem("480G") == 480

    def test_format_mem(self):
        assert _format_mem(15) == "15G"
        assert _format_mem(480) == "480G"

    def test_parse_time(self):
        assert _parse_time("6:00:00") == 21600
        assert _parse_time("12:00:00") == 43200
        assert _parse_time("1:30:00") == 5400

    def test_format_time(self):
        assert _format_time(21600) == "6:00:00"
        assert _format_time(43200) == "12:00:00"
        assert _format_time(5400) == "1:30:00"


# ---------------------------------------------------------------------------
# Edge cases for process_errors
# ---------------------------------------------------------------------------


class TestProcessErrorsEdgeCases:
    def test_cuda_oom_at_max_tier_no_escalation(self):
        """CUDA OOM at max MIG tier does not crash."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-80gb",
            initial_mem="124G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "cuda_oom",
        }
        # Should not raise
        process_errors(registry["configs"])
        # Tier stays the same since there is no next tier
        assert registry["configs"][config_name]["mig_tier"] == "H100-80gb"
        # Trial should still be marked for retry
        assert registry["configs"][config_name]["trials"]["0"]["retry"] is True

    def test_system_mem_cap_at_480(self):
        """System memory doubling caps at 480G."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="300G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "system_oom",
        }
        process_errors(registry["configs"])
        # 300 * 2 = 600, but capped at 480
        assert registry["configs"][config_name]["system_mem"] == "480G"

    def test_time_cap_at_48h(self):
        """Time doubling caps at 48 hours."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="30:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "timeout",
        }
        process_errors(registry["configs"])
        # 30h * 2 = 60h, capped at 48h
        assert registry["configs"][config_name]["time_limit"] == "48:00:00"

    def test_skip_complete_configs(self):
        """process_errors does not touch complete configs."""
        registry = create_initial_registry(
            trials_per_config=5,
            initial_mig="H100-1g.10gb",
            initial_mem="15G",
            initial_time="6:00:00",
            max_cycles=5,
        )
        config_name = ALL_CONFIGS[0]
        registry["configs"][config_name]["complete"] = True
        registry["configs"][config_name]["trials"]["0"] = {
            "status": "failed",
            "error": "cuda_oom",
        }
        process_errors(registry["configs"])
        # Should still be at original tier since config is complete
        assert registry["configs"][config_name]["mig_tier"] == "H100-1g.10gb"
