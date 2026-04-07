"""HP tuning sentinel: cycle management and planner re-submission.

Runs as the last job in each cycle (afterany on error_collector). Checks the
registry for remaining work, decides whether to start another planner->executor
cycle, and writes a summary when tuning is complete or max cycles are reached.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for direct execution (python slurm/hp_sentinel.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.atomic_io import atomic_json_dump


# ---------------------------------------------------------------------------
# Slurm script templates (CPU jobs for planner and executor)
# ---------------------------------------------------------------------------

PLANNER_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={cpu_account}
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --output=slurm_out/HP_planner_cycle_{cycle}_%j.out
#SBATCH --error=slurm_err/HP_planner_cycle_{cycle}_%j.err
module load {modules}
source {venv}/bin/activate
cd {submit_dir}
python -m slurm.hp_planner --registry {registry} --max-jobs {max_jobs} --trials-per-config {trials_per_config} --initial-mig {initial_mig} --initial-mem {initial_mem} --initial-time {initial_time} --max-cycles {max_cycles} --manifest {manifest} --base-dir {base_dir}
"""

EXECUTOR_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={cpu_account}
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --output=slurm_out/HP_executor_cycle_{cycle}_%j.out
#SBATCH --error=slurm_err/HP_executor_cycle_{cycle}_%j.err
module load {modules}
source {venv}/bin/activate
cd {submit_dir}
python -m slurm.hp_executor --manifest {manifest} --registry {registry} --gpu-account {gpu_account} --cpu-account {cpu_account} --max-cycles {max_cycles} --max-jobs {max_jobs} --trials-per-config {trials_per_config} --initial-mig {initial_mig} --initial-mem {initial_mem} --initial-time {initial_time} --base-dir {base_dir}
"""


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(registry, reason):
    """Write experiments/hp_tuning/summary.json with final status.

    Args:
        registry: The full registry dict.
        reason: One of "all_complete" or "max_cycles_reached".
    """
    configs_summary = {}
    for config_name, config in registry.get("configs", {}).items():
        trials = config.get("trials", {})
        completed_count = sum(
            1 for t in trials.values() if t.get("status") == "completed"
        )
        failed_count = sum(
            1 for t in trials.values()
            if t.get("status") in ("failed", "permanently_failed")
        )
        configs_summary[config_name] = {
            "complete": config.get("complete", False),
            "completed_trials": completed_count,
            "failed_trials": failed_count,
            "mig_tier": config.get("mig_tier", "unknown"),
        }

    summary = {
        "reason": reason,
        "cycle": registry.get("cycle", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configs": configs_summary,
    }

    summary_path = Path("experiments") / "hp_tuning" / "summary.json"
    atomic_json_dump(summary, summary_path)
    print(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def _submit_job(script_path, dependency=None, dry_run=False):
    """Submit a Slurm job script via sbatch.

    Args:
        script_path: Path to the Slurm job script.
        dependency: Optional dependency string (e.g. "afterok:12345").
        dry_run: If True, print instead of submitting.

    Returns:
        Job ID string, or "DRY_RUN" if dry_run is True.
    """
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script_path))

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return "DRY_RUN"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ERROR: sbatch failed for {script_path}")
        print(f"  stdout: {result.stdout.strip()}")
        print(f"  stderr: {result.stderr.strip()}")
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}")

    job_id = result.stdout.strip()
    print(f"  Submitted {script_path.name} -> {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Check registry for remaining work and re-submit planner+executor if needed."""
    parser = argparse.ArgumentParser(
        description="HP tuning sentinel: cycle management and planner re-submission"
    )
    parser.add_argument(
        "--registry", type=str, default="hp_tuning/job_registry.json",
        help="Path to the job registry JSON file",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=5,
        help="Maximum number of sentinel cycles",
    )
    parser.add_argument(
        "--gpu-account", type=str, required=True,
        help="SLURM billing account for GPU jobs",
    )
    parser.add_argument(
        "--cpu-account", type=str, required=True,
        help="SLURM billing account for CPU jobs",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=800,
        help="Maximum jobs per cycle",
    )
    parser.add_argument(
        "--trials-per-config", type=int, default=50,
        help="Trials per config",
    )
    parser.add_argument(
        "--initial-mig", type=str, default="H100-1g.10gb",
        help="Initial MIG tier",
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
        "--modules", type=str, default="StdEnv/2023 python/3.11.5 scipy-stack/2025a",
        help="Space-separated module list for 'module load'",
    )
    parser.add_argument(
        "--venv", type=str, default="env",
        help="Path to Python virtual environment",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate scripts but do not submit",
    )
    args = parser.parse_args(argv)

    registry_path = Path(args.registry)

    # ---- Load registry ----
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    cycle = registry.get("cycle", 1)
    max_cycles = args.max_cycles
    configs = registry.get("configs", {})

    # ---- Count trial statuses across all non-complete configs ----
    completed = 0
    failed_retryable = 0
    failed_permanent = 0
    pending = 0
    submitted = 0

    for config_name, config in configs.items():
        if config.get("complete", False):
            # Count completed config trials for the summary but skip work check
            for trial in config.get("trials", {}).values():
                if trial.get("status") == "completed":
                    completed += 1
            continue

        for trial in config.get("trials", {}).values():
            status = trial.get("status", "pending")
            if status == "completed":
                completed += 1
            elif status == "failed":
                error = trial.get("error", "")
                if error in ("cuda_oom", "system_oom", "timeout"):
                    failed_retryable += 1
                else:
                    failed_permanent += 1
            elif status == "permanently_failed":
                failed_permanent += 1
            elif status == "pending":
                pending += 1
            elif status == "submitted":
                submitted += 1

    remaining = failed_retryable + pending + submitted

    # ---- Print summary ----
    print(f"Sentinel cycle {cycle}/{max_cycles}")
    print(f"  Completed:         {completed}")
    print(f"  Failed (retryable): {failed_retryable}")
    print(f"  Failed (permanent): {failed_permanent}")
    print(f"  Pending:           {pending}")
    print(f"  Submitted:         {submitted}")
    print(f"  Remaining work:    {remaining}")

    # ---- Decision logic ----
    if remaining == 0:
        print("All work complete. Writing summary and exiting.")
        _write_summary(registry, "all_complete")
        return

    if cycle >= max_cycles:
        print(f"Max cycles ({max_cycles}) reached. Writing summary and exiting.")
        _write_summary(registry, "max_cycles_reached")
        return

    # ---- Remaining work exists and cycles remain: start new cycle ----
    new_cycle = cycle + 1
    registry["cycle"] = new_cycle
    atomic_json_dump(registry, registry_path)
    print(f"Registry updated: cycle {cycle} -> {new_cycle}")

    # ---- Generate and submit planner + executor scripts ----
    submit_dir = os.getcwd()
    base_dir = "experiments/hp_tuning"
    manifest = "experiments/hp_tuning/manifest.json"

    scripts_dir = Path(base_dir) / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Ensure slurm log directories exist
    Path("slurm_out").mkdir(exist_ok=True)
    Path("slurm_err").mkdir(exist_ok=True)

    # Planner script
    planner_content = PLANNER_TEMPLATE.format(
        cpu_account=args.cpu_account,
        cycle=new_cycle,
        modules=args.modules,
        venv=args.venv,
        submit_dir=submit_dir,
        registry=str(registry_path),
        max_jobs=args.max_jobs,
        trials_per_config=args.trials_per_config,
        initial_mig=args.initial_mig,
        initial_mem=args.initial_mem,
        initial_time=args.initial_time,
        max_cycles=max_cycles,
        manifest=manifest,
        base_dir=base_dir,
    )
    planner_path = scripts_dir / f"planner_cycle_{new_cycle}.sh"
    planner_path.write_text(planner_content)

    # Executor script
    executor_content = EXECUTOR_TEMPLATE.format(
        cpu_account=args.cpu_account,
        cycle=new_cycle,
        modules=args.modules,
        venv=args.venv,
        submit_dir=submit_dir,
        manifest=manifest,
        registry=str(registry_path),
        gpu_account=args.gpu_account,
        max_cycles=max_cycles,
        max_jobs=args.max_jobs,
        trials_per_config=args.trials_per_config,
        initial_mig=args.initial_mig,
        initial_mem=args.initial_mem,
        initial_time=args.initial_time,
        base_dir=base_dir,
    )
    executor_path = scripts_dir / f"executor_cycle_{new_cycle}.sh"
    executor_path.write_text(executor_content)

    # Submit: planner first, then executor afterok on planner
    planner_job_id = _submit_job(planner_path, dry_run=args.dry_run)
    executor_dep = f"afterok:{planner_job_id}"
    executor_job_id = _submit_job(
        executor_path, dependency=executor_dep, dry_run=args.dry_run
    )

    print(f"Cycle {new_cycle} submitted:")
    print(f"  Planner:  {planner_job_id}")
    print(f"  Executor: {executor_job_id} (afterok on planner)")


if __name__ == "__main__":
    main()
