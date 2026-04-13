"""HP tuning executor: Slurm script generation, job submission, and dependency wiring.

Reads a manifest.json (produced by hp_planner), generates per-trial Slurm scripts,
submits training jobs with GPU resources, then chains error_collector and sentinel
as afterany dependencies. Updates the job registry with submitted Slurm IDs.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path for direct execution (python slurm/hp_executor.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import optuna
except ImportError:
    optuna = None

from slurm.hp_config import mig_resources, arch_dataset_from_config
from utils.atomic_io import atomic_json_dump


# ---------------------------------------------------------------------------
# Slurm script templates
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={gpu_account}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output=slurm_out/HP_{config}_trial_{trial_id}_%j.out
#SBATCH --error=slurm_err/HP_{config}_trial_{trial_id}_%j.err
module load {modules}
source {venv}/bin/activate
mkdir -p $SLURM_TMPDIR/data
{data_copy_commands}
cd {submit_dir}
python slurm/train_trial.py --config {config} --study-db {study_db} --output-dir {output_dir} --data-dir $SLURM_TMPDIR/data --seed 42
"""

ERROR_COLLECTOR_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={cpu_account}
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --output=slurm_out/HP_error_collector_%j.out
#SBATCH --error=slurm_err/HP_error_collector_%j.err
module load {modules}
source {venv}/bin/activate
cd {submit_dir}
python -m slurm.hp_error_collector --registry {registry} --stderr-dir slurm_err
"""

SENTINEL_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={cpu_account}
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --output=slurm_out/HP_sentinel_%j.out
#SBATCH --error=slurm_err/HP_sentinel_%j.err
module load {modules}
source {venv}/bin/activate
cd {submit_dir}
python -m slurm.hp_sentinel --registry {registry} --max-cycles {max_cycles} --gpu-account {gpu_account} --cpu-account {cpu_account} --max-jobs {max_jobs} --trials-per-config {trials_per_config} --initial-mig {initial_mig} --initial-mem {initial_mem} --initial-time {initial_time}
"""


# ---------------------------------------------------------------------------
# Data copy commands
# ---------------------------------------------------------------------------


def generate_data_copy_commands(dataset_name, submit_dir):
    """Generate shell commands to copy dataset from persistent storage to $SLURM_TMPDIR.

    Args:
        dataset_name: One of "cifar10", "cifar100", "tiny_imagenet".
        submit_dir: The SLURM_SUBMIT_DIR path (project root on persistent storage).

    Returns:
        Multi-line string of shell commands for the Slurm script.
    """
    if dataset_name == "cifar10":
        return (
            f"mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/\n"
            f"cp -r {submit_dir}/data/cifar-10-batches-py/* "
            f"$SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true"
        )
    elif dataset_name == "cifar100":
        return (
            f"mkdir -p $SLURM_TMPDIR/data/cifar-100-python/\n"
            f"cp -r {submit_dir}/data/cifar-100-python/* "
            f"$SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true"
        )
    elif dataset_name == "tiny_imagenet":
        return (
            f"cp -r {submit_dir}/data/tiny-imagenet-200 "
            f"$SLURM_TMPDIR/data/ 2>/dev/null || true"
        )
    else:
        raise ValueError(f"Unknown dataset for data copy: {dataset_name}")


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def submit_job(script_path, dependency=None, dry_run=False):
    """Submit a Slurm job script via sbatch.

    Args:
        script_path: Path to the Slurm job script.
        dependency: Optional dependency string (e.g. "afterany:12345:12346").
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
# Seed initial trial
# ---------------------------------------------------------------------------

# Known-good HP configurations from existing experiments.
# These are enqueued as the first trial for matching configs so Optuna
# starts from a reasonable baseline rather than purely random sampling.
_SEED_PARAMS = {
    "resnet18_cifar10": {
        "p1_optimizer": "adam",
        "p1_lr": 1e-3,
        "p1_epochs": 10,
        "p1_weight_decay": 1e-5,
        "p2_optimizer": "sgd",
        "p2_lr": 1e-5,
        "p2_momentum": 0.9,
        "p2_weight_decay": 5e-4,
        "p2_scheduler": "cosine",
        "p2_max_epochs": 60,
        "p2_patience": 10,
    },
    "resnet18_cifar100": {
        "p1_optimizer": "adam",
        "p1_lr": 1e-3,
        "p1_epochs": 10,
        "p1_weight_decay": 1e-5,
        "p2_optimizer": "sgd",
        "p2_lr": 1e-5,
        "p2_momentum": 0.9,
        "p2_weight_decay": 5e-4,
        "p2_scheduler": "cosine",
        "p2_max_epochs": 60,
        "p2_patience": 10,
    },
    "vgg11_bn_cifar10": {
        "p1_optimizer": "adam",
        "p1_lr": 1e-3,
        "p1_epochs": 10,
        "p1_weight_decay": 1e-5,
        "p2_optimizer": "adam",
        "p2_lr": 1.2e-4,
        "p2_weight_decay": 1.4e-4,
        "p2_scheduler": "cosine",
        "p2_max_epochs": 60,
        "p2_patience": 10,
    },
    "vgg11_bn_cifar100": {
        "p1_optimizer": "adam",
        "p1_lr": 1e-3,
        "p1_epochs": 10,
        "p1_weight_decay": 1e-5,
        "p2_optimizer": "adam",
        "p2_lr": 1.2e-4,
        "p2_weight_decay": 1.4e-4,
        "p2_scheduler": "cosine",
        "p2_max_epochs": 60,
        "p2_patience": 10,
    },
}


def _seed_initial_trial(study, config_name):
    """Enqueue a known-good HP configuration as the first trial if available.

    Args:
        study: An Optuna study instance.
        config_name: Config name (e.g. "resnet18_cifar10").
    """
    params = _SEED_PARAMS.get(config_name)
    if params is not None:
        study.enqueue_trial(params)
        print(f"  Seeded initial trial for {config_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Read manifest, generate Slurm scripts, submit training + collector + sentinel."""
    parser = argparse.ArgumentParser(
        description="HP tuning executor: generate and submit Slurm jobs"
    )
    parser.add_argument(
        "--manifest", type=str, default="hp_tuning/manifest.json",
        help="Path to the job manifest JSON (from hp_planner)",
    )
    parser.add_argument(
        "--registry", type=str, default="hp_tuning/job_registry.json",
        help="Path to the job registry JSON file",
    )
    parser.add_argument(
        "--gpu-account", type=str, required=True,
        help="SLURM billing account for GPU jobs",
    )
    parser.add_argument(
        "--cpu-account", type=str, required=True,
        help="SLURM billing account for CPU jobs (error collector, sentinel)",
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
        "--base-dir", type=str, default="experiments/hp_tuning",
        help="Base directory for HP tuning outputs",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=5,
        help="Maximum sentinel cycles (passed to sentinel)",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=800,
        help="Maximum jobs per cycle (passed to sentinel)",
    )
    parser.add_argument(
        "--trials-per-config", type=int, default=50,
        help="Trials per config (passed to sentinel)",
    )
    parser.add_argument(
        "--initial-mig", type=str, default="H100-1g.10gb",
        help="Initial MIG tier (passed to sentinel)",
    )
    parser.add_argument(
        "--initial-mem", type=str, default="15G",
        help="Initial system memory (passed to sentinel)",
    )
    parser.add_argument(
        "--initial-time", type=str, default="6:00:00",
        help="Initial time limit (passed to sentinel)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate scripts but do not submit",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    registry_path = Path(args.registry)
    base_dir = Path(args.base_dir)
    submit_dir = os.getcwd()

    # ---- Load manifest ----
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    jobs = manifest.get("jobs", [])
    cycle = manifest.get("cycle", 1)
    print(f"Executor: cycle {cycle}, {len(jobs)} job(s) in manifest")

    if not jobs:
        print("No jobs to submit. Exiting.")
        return

    # ---- Load registry ----
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return
    with open(registry_path) as f:
        registry = json.load(f)

    # ---- Create output directories ----
    scripts_dir = base_dir / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Ensure slurm log directories exist
    Path("slurm_out").mkdir(exist_ok=True)
    Path("slurm_err").mkdir(exist_ok=True)

    # ---- Create Optuna studies and submit training jobs ----
    training_job_ids = []

    for job in jobs:
        config_name = job["config"]
        trial_id = job["trial_id"]
        mig_tier = job["mig"]
        mem = job["mem"]
        time_limit = job["time"]
        cpus = job["cpus"]

        # Parse dataset from config name
        arch, dataset = arch_dataset_from_config(config_name)

        # Resolve MIG resources
        tier = mig_resources(mig_tier)
        if tier is None:
            print(f"WARNING: Unknown MIG tier {mig_tier} for {config_name}, skipping")
            continue
        gres = tier["gres"]

        # Create per-config output directory
        output_dir = base_dir / config_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create or load Optuna study (SQLite-backed for cross-job coordination)
        study_db = str(output_dir / "optuna_study.db")
        if optuna is not None:
            storage = f"sqlite:///{study_db}"
            study = optuna.create_study(
                study_name=config_name,
                storage=storage,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=10
                ),
                load_if_exists=True,
            )
            # Seed initial trial for new studies
            if len(study.trials) == 0:
                _seed_initial_trial(study, config_name)
        else:
            print("WARNING: optuna not installed, skipping study creation")

        # Generate data copy commands
        data_copy_commands = generate_data_copy_commands(dataset, submit_dir)

        # Render Slurm script
        script_content = SLURM_TEMPLATE.format(
            gpu_account=args.gpu_account,
            gres=gres,
            cpus=cpus,
            time=time_limit,
            mem=mem,
            config=config_name,
            trial_id=trial_id,
            modules=args.modules,
            venv=args.venv,
            submit_dir=submit_dir,
            data_copy_commands=data_copy_commands,
            study_db=study_db,
            output_dir=str(output_dir),
        )

        script_path = scripts_dir / f"train_{config_name}_trial_{trial_id}.sh"
        script_path.write_text(script_content)

        # Submit
        job_id = submit_job(script_path, dry_run=args.dry_run)
        training_job_ids.append(job_id)

        # Update registry
        trial_ref = (
            registry.get("configs", {})
            .get(config_name, {})
            .get("trials", {})
            .get(str(trial_id))
        )
        if trial_ref is not None:
            trial_ref["status"] = "submitted"
            trial_ref["slurm_id"] = job_id
            # Clear retry flag if it was set
            trial_ref.pop("retry", None)

    print(f"Submitted {len(training_job_ids)} training job(s)")

    # ---- Submit error collector (afterany on all training jobs) ----
    if training_job_ids:
        collector_script = ERROR_COLLECTOR_TEMPLATE.format(
            cpu_account=args.cpu_account,
            modules=args.modules,
            venv=args.venv,
            submit_dir=submit_dir,
            registry=str(registry_path),
        )
        collector_path = scripts_dir / f"error_collector_cycle_{cycle}.sh"
        collector_path.write_text(collector_script)

        dep_ids = ":".join(training_job_ids)
        collector_dep = f"afterany:{dep_ids}"
        collector_job_id = submit_job(
            collector_path, dependency=collector_dep, dry_run=args.dry_run
        )
        print(f"Error collector: {collector_job_id} (afterany on {len(training_job_ids)} jobs)")

        # ---- Submit sentinel (afterany on error collector) ----
        sentinel_script = SENTINEL_TEMPLATE.format(
            cpu_account=args.cpu_account,
            modules=args.modules,
            venv=args.venv,
            submit_dir=submit_dir,
            registry=str(registry_path),
            max_cycles=args.max_cycles,
            gpu_account=args.gpu_account,
            max_jobs=args.max_jobs,
            trials_per_config=args.trials_per_config,
            initial_mig=args.initial_mig,
            initial_mem=args.initial_mem,
            initial_time=args.initial_time,
        )
        sentinel_path = scripts_dir / f"sentinel_cycle_{cycle}.sh"
        sentinel_path.write_text(sentinel_script)

        sentinel_dep = f"afterany:{collector_job_id}"
        sentinel_job_id = submit_job(
            sentinel_path, dependency=sentinel_dep, dry_run=args.dry_run
        )
        print(f"Sentinel: {sentinel_job_id} (afterany on error collector)")

    # ---- Save updated registry ----
    atomic_json_dump(registry, registry_path)
    print(f"Registry updated at {registry_path}")


if __name__ == "__main__":
    main()
