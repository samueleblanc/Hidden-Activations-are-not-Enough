"""HP tuning planner: registry management, error escalation, and job selection.

Reads/creates a job_registry.json, processes errors from completed trials,
selects the next batch of jobs to run, and writes a manifest.json for the
executor. Called by the sentinel between pipeline cycles.
"""

import argparse
import json
import os
from pathlib import Path

from slurm.hp_config import ALL_CONFIGS, arch_dataset_from_config, next_mig_tier, mig_resources
from utils.atomic_io import atomic_json_dump


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

MAX_SYSTEM_MEM_GB = 480
MAX_TIME_SECONDS = 48 * 3600  # 48 hours


def _parse_mem(mem_str):
    """Parse memory string like '15G' to integer GB."""
    if mem_str.endswith("G"):
        return int(mem_str[:-1])
    raise ValueError(f"Cannot parse memory string: {mem_str}")


def _format_mem(mem_gb):
    """Format integer GB to memory string like '15G'."""
    return f"{mem_gb}G"


def _parse_time(time_str):
    """Parse time string like '6:00:00' to total seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse time string: {time_str}")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _format_time(total_seconds):
    """Format total seconds to time string like '6:00:00'."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Registry creation
# ---------------------------------------------------------------------------


def create_initial_registry(trials_per_config, initial_mig, initial_mem,
                            initial_time, max_cycles):
    """Create a fresh registry with all 27 configs from ALL_CONFIGS.

    Each config has:
        mig_tier, system_mem, time_limit, cpus, target_trials,
        complete=False, trials={0..N-1: {status: "pending"}}

    Returns dict with cycle=1, max_cycles, configs.
    """
    tier = mig_resources(initial_mig)
    cpus = tier["cpus"] if tier else 2

    configs = {}
    for config_name in ALL_CONFIGS:
        trials = {}
        for i in range(trials_per_config):
            trials[str(i)] = {"status": "pending"}
        configs[config_name] = {
            "mig_tier": initial_mig,
            "system_mem": initial_mem,
            "time_limit": initial_time,
            "cpus": cpus,
            "target_trials": trials_per_config,
            "complete": False,
            "trials": trials,
        }

    return {
        "cycle": 1,
        "max_cycles": max_cycles,
        "configs": configs,
    }


# ---------------------------------------------------------------------------
# Error processing
# ---------------------------------------------------------------------------


def process_errors(configs):
    """Scan all non-complete configs for failed trials and apply escalation.

    - cuda_oom: escalate MIG tier to next level, update cpus and system_mem
    - system_oom: double system_mem (cap at 480G)
    - timeout: double time_limit (cap at 48h)

    Failed trials are marked with retry=True.
    """
    for config_name, config in configs.items():
        if config["complete"]:
            continue

        has_cuda_oom = False
        has_system_oom = False
        has_timeout = False

        for trial_id, trial in config["trials"].items():
            if trial.get("status") != "failed":
                continue

            error = trial.get("error", "")
            if error == "cuda_oom":
                has_cuda_oom = True
                trial["retry"] = True
            elif error == "system_oom":
                has_system_oom = True
                trial["retry"] = True
            elif error == "timeout":
                has_timeout = True
                trial["retry"] = True

        # Apply escalations at the config level
        if has_cuda_oom:
            new_tier = next_mig_tier(config["mig_tier"])
            if new_tier is not None:
                config["mig_tier"] = new_tier
                resources = mig_resources(new_tier)
                config["cpus"] = resources["cpus"]
                config["system_mem"] = resources["mem"]

        if has_system_oom:
            current_mem = _parse_mem(config["system_mem"])
            new_mem = min(current_mem * 2, MAX_SYSTEM_MEM_GB)
            config["system_mem"] = _format_mem(new_mem)

        if has_timeout:
            current_time = _parse_time(config["time_limit"])
            new_time = min(current_time * 2, MAX_TIME_SECONDS)
            config["time_limit"] = _format_time(new_time)


# ---------------------------------------------------------------------------
# Job selection
# ---------------------------------------------------------------------------


def select_jobs(configs, max_jobs):
    """Select next batch of jobs to run from the registry.

    Phase 1: Collect all trials with retry=True (prioritized).
    Phase 2: Round-robin pending trials across configs, sorted by fewest completed.

    Each job is a dict: {config, trial_id, mig, mem, time, cpus}.
    Returns list of up to max_jobs.
    """
    jobs = []

    # Phase 1: retry jobs first
    for config_name, config in configs.items():
        if config["complete"]:
            continue
        for trial_id, trial in config["trials"].items():
            if trial.get("retry"):
                jobs.append({
                    "config": config_name,
                    "trial_id": trial_id,
                    "mig": config["mig_tier"],
                    "mem": config["system_mem"],
                    "time": config["time_limit"],
                    "cpus": config["cpus"],
                })
    if len(jobs) >= max_jobs:
        return jobs[:max_jobs]

    # Phase 2: round-robin pending trials
    # Sort configs by fewest completed trials (to balance progress)
    active_configs = []
    for config_name, config in configs.items():
        if config["complete"]:
            continue
        pending_trials = [
            (trial_id, trial)
            for trial_id, trial in config["trials"].items()
            if trial.get("status") == "pending"
        ]
        if pending_trials:
            completed_count = sum(
                1 for t in config["trials"].values()
                if t.get("status") == "completed"
            )
            active_configs.append((config_name, config, pending_trials, completed_count))

    # Sort by fewest completed (least progress first)
    active_configs.sort(key=lambda x: x[3])

    # Round-robin: take one pending trial from each config in turn
    remaining = max_jobs - len(jobs)
    # Track index into each config's pending list
    indices = [0] * len(active_configs)
    while remaining > 0:
        added_this_round = 0
        for i, (config_name, config, pending_trials, _) in enumerate(active_configs):
            if remaining <= 0:
                break
            if indices[i] < len(pending_trials):
                trial_id, trial = pending_trials[indices[i]]
                jobs.append({
                    "config": config_name,
                    "trial_id": trial_id,
                    "mig": config["mig_tier"],
                    "mem": config["system_mem"],
                    "time": config["time_limit"],
                    "cpus": config["cpus"],
                })
                indices[i] += 1
                remaining -= 1
                added_this_round += 1
        if added_this_round == 0:
            break  # No more pending trials anywhere

    return jobs


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_completed_configs(configs, base_dir):
    """For configs where all trials are done, find best trial and record results.

    Reads trial checkpoint JSONs from disk, selects the trial with the
    highest validation accuracy, writes best_hps.json, appends entry to
    constants/constants.py, and marks the config as complete=True.
    """
    base_dir = Path(base_dir)

    for config_name, config in configs.items():
        if config["complete"]:
            continue

        # Check if all trials are terminal (completed or permanently failed)
        all_terminal = all(
            t.get("status") in ("completed", "permanently_failed")
            for t in config["trials"].values()
        )
        if not all_terminal:
            continue

        # Find best trial by reading checkpoint JSONs
        best_trial = None
        best_val_acc = -1.0

        for trial_id, trial in config["trials"].items():
            if trial.get("status") != "completed":
                continue
            checkpoint_path = (
                base_dir / "hp_tuning" / config_name / f"trial_{trial_id}"
                / "checkpoint.json"
            )
            if not checkpoint_path.exists():
                continue
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            val_acc = ckpt.get("best_val_acc", -1.0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_trial = ckpt
                best_trial["trial_id"] = trial_id

        if best_trial is not None:
            # Write best_hps.json
            output_dir = base_dir / "hp_tuning" / config_name
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(best_trial, output_dir / "best_hps.json")

            # Append to constants.py
            constants_path = base_dir / "constants" / "constants.py"
            if constants_path.exists():
                _append_to_constants(constants_path, config_name, best_trial)

        config["complete"] = True


def _append_to_constants(constants_path, config_name, best_trial):
    """Append a new experiment entry to DEFAULT_EXPERIMENTS in constants.py.

    Reads the file, finds the closing brace of DEFAULT_EXPERIMENTS,
    and inserts the new entry before it.
    """
    constants_path = Path(constants_path)
    content = constants_path.read_text()

    # Find DEFAULT_EXPERIMENTS = { and count braces to locate closing }
    start_marker = "DEFAULT_EXPERIMENTS = {"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"WARNING: Could not find DEFAULT_EXPERIMENTS in {constants_path}")
        return

    # Walk from the opening brace, counting brace depth
    brace_start = start_idx + len(start_marker) - 1  # index of {
    depth = 1
    pos = brace_start + 1
    while pos < len(content) and depth > 0:
        if content[pos] == "{":
            depth += 1
        elif content[pos] == "}":
            depth -= 1
        pos += 1

    # pos is now one past the closing }
    closing_brace_idx = pos - 1

    # Build the new entry
    train_acc = best_trial.get("train_acc", "N/A")
    val_acc = best_trial.get("best_val_acc", "N/A")
    test_acc = best_trial.get("test_acc", "N/A")

    params = best_trial.get("params", {})
    arch, dataset = arch_dataset_from_config(config_name)

    entry_lines = [
        f"    # HP-tuned: Train={train_acc}, Val={val_acc}, Test={test_acc}",
        f"    '{config_name}': {{",
        f"        'pretrained': True,",
        f"        'dataset': '{dataset}',",
        f"        'architecture': '{arch}',",
    ]
    for key, value in sorted(params.items()):
        entry_lines.append(f"        '{key}': {repr(value)},")
    entry_lines.append(f"    }},")
    entry_lines.append("")

    new_entry = "\n".join(entry_lines)

    # Insert before closing brace
    new_content = content[:closing_brace_idx] + new_entry + "\n" + content[closing_brace_idx:]
    constants_path.write_text(new_content)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Load or create registry, process errors, finalize, select jobs, write manifest."""
    parser = argparse.ArgumentParser(
        description="HP tuning planner: manage registry and select jobs"
    )
    parser.add_argument(
        "--registry", type=str, default="hp_tuning/job_registry.json",
        help="Path to the job registry JSON file",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=800,
        help="Maximum number of jobs to select per cycle",
    )
    parser.add_argument(
        "--trials-per-config", type=int, default=50,
        help="Number of trials per config (for initial registry creation)",
    )
    parser.add_argument(
        "--initial-mig", type=str, default="H100-1g.10gb",
        help="Initial MIG tier name",
    )
    parser.add_argument(
        "--initial-mem", type=str, default="15G",
        help="Initial system memory",
    )
    parser.add_argument(
        "--initial-time", type=str, default="6:00:00",
        help="Initial time limit",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=5,
        help="Maximum number of sentinel cycles",
    )
    parser.add_argument(
        "--manifest", type=str, default="hp_tuning/manifest.json",
        help="Path to write the job manifest",
    )
    parser.add_argument(
        "--base-dir", type=str, default=".",
        help="Base directory for experiment outputs",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    manifest_path = Path(args.manifest)

    # Load or create registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        print(f"Loaded registry from {registry_path} (cycle {registry['cycle']})")
    else:
        registry = create_initial_registry(
            trials_per_config=args.trials_per_config,
            initial_mig=args.initial_mig,
            initial_mem=args.initial_mem,
            initial_time=args.initial_time,
            max_cycles=args.max_cycles,
        )
        print(f"Created initial registry with {len(registry['configs'])} configs")

    # Process errors from completed trials
    process_errors(registry["configs"])

    # Finalize completed configs
    finalize_completed_configs(registry["configs"], args.base_dir)

    # Select next batch of jobs
    jobs = select_jobs(registry["configs"], max_jobs=args.max_jobs)
    print(f"Selected {len(jobs)} jobs for cycle {registry['cycle']}")

    # Write manifest
    manifest = {
        "cycle": registry["cycle"],
        "jobs": jobs,
    }
    atomic_json_dump(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}")

    # Save registry
    atomic_json_dump(registry, registry_path)
    print(f"Registry saved to {registry_path}")

    return jobs


if __name__ == "__main__":
    main()
