"""HP tuning error collector: sacct parsing and job outcome classification.

Scans sacct output and stderr logs to classify job outcomes (completed,
cuda_oom, system_oom, timeout, other). Updates the job registry so the
planner can process errors in the next cycle.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path for direct execution (python slurm/hp_error_collector.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slurm.hp_config import EXIT_CUDA_OOM
from utils.atomic_io import atomic_json_dump


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_job(state, exit_code, stderr_tail):
    """Classify a job outcome from its Slurm state, exit code, and stderr.

    Priority order:
        1. COMPLETED + 0:0 -> "completed"
        2. OUT_OF_MEMORY state -> "system_oom"
        3. TIMEOUT state -> "timeout"
        4. Exit code 42 (main) -> "cuda_oom"
        5. Exit code 137 (either position) -> "system_oom"
        6. Exit code 140 (either position) -> "timeout"
        7. stderr contains CUDA OOM patterns -> "cuda_oom"
        8. stderr contains system OOM patterns -> "system_oom"
        9. Else -> "other"

    Args:
        state: Slurm job state string (e.g. "COMPLETED", "FAILED").
        exit_code: Slurm exit code string "main:signal" (e.g. "0:0", "42:0").
        stderr_tail: Last N lines of the job's stderr log.

    Returns:
        One of: "completed", "cuda_oom", "system_oom", "timeout", "other".
    """
    # 1. Completed successfully
    if state == "COMPLETED" and exit_code == "0:0":
        return "completed"

    # 2. Slurm state-based classification
    if state == "OUT_OF_MEMORY":
        return "system_oom"
    if state == "TIMEOUT":
        return "timeout"

    # 3. Exit code-based classification
    main_code, signal_code = _parse_exit_code(exit_code)

    if main_code == EXIT_CUDA_OOM:
        return "cuda_oom"
    if main_code == 137 or signal_code == 137:
        return "system_oom"
    if main_code == 140 or signal_code == 140:
        return "timeout"

    # 4. Stderr-based classification
    if stderr_tail:
        if "CUDA out of memory" in stderr_tail or "torch.cuda.OutOfMemoryError" in stderr_tail:
            return "cuda_oom"
        if "oom-kill" in stderr_tail or "Out of memory" in stderr_tail:
            return "system_oom"

    return "other"


def _parse_exit_code(exit_code):
    """Parse Slurm exit code string "main:signal" into (int, int).

    Args:
        exit_code: String like "0:0" or "42:0".

    Returns:
        Tuple of (main_code, signal_code) as integers.
    """
    parts = exit_code.split(":")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return 0, 0


# ---------------------------------------------------------------------------
# sacct parsing
# ---------------------------------------------------------------------------


def parse_sacct_line(line):
    """Parse a pipe-delimited sacct output line.

    Expected format: "job_id|state|exit_code"

    Args:
        line: A single line from sacct --parsable2 output.

    Returns:
        Tuple of (job_id, state, exit_code) as strings.
    """
    parts = line.strip().split("|")
    return parts[0], parts[1], parts[2]


def query_sacct(job_ids):
    """Query sacct for job states and exit codes.

    Runs: sacct --jobs=id1,id2,... --format=JobID,State,ExitCode
          --noheader --parsable2 --allocations

    Args:
        job_ids: List of Slurm job ID strings.

    Returns:
        Dict mapping job_id -> (state, exit_code).
    """
    if not job_ids:
        return {}

    cmd = [
        "sacct",
        f"--jobs={','.join(job_ids)}",
        "--format=JobID,State,ExitCode",
        "--noheader",
        "--parsable2",
        "--allocations",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"WARNING: sacct returned exit code {result.returncode}")
            print(f"  stderr: {result.stderr.strip()}")
            return {}
    except FileNotFoundError:
        print("WARNING: sacct command not found (not on a Slurm cluster?)")
        return {}
    except subprocess.TimeoutExpired:
        print("WARNING: sacct query timed out after 60s")
        return {}

    results = {}
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        job_id, state, exit_code = parse_sacct_line(line)
        results[job_id] = (state, exit_code)

    return results


# ---------------------------------------------------------------------------
# Stderr reading
# ---------------------------------------------------------------------------


def read_stderr_tail(slurm_id, stderr_dir, lines=50):
    """Read the last N lines from a Slurm stderr log file.

    Globs for *_{slurm_id}.err in the stderr directory.

    Args:
        slurm_id: Slurm job ID string.
        stderr_dir: Directory containing stderr log files.
        lines: Number of lines to read from the end.

    Returns:
        String containing the last N lines, or empty string if not found.
    """
    pattern = str(Path(stderr_dir) / f"*_{slurm_id}.err")
    matches = glob.glob(pattern)
    if not matches:
        return ""

    # Use the first match
    filepath = matches[0]
    try:
        with open(filepath) as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Load registry, query sacct for submitted trials, classify, and update."""
    parser = argparse.ArgumentParser(
        description="HP tuning error collector: classify job outcomes via sacct"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="hp_tuning/job_registry.json",
        help="Path to the job registry JSON file",
    )
    parser.add_argument(
        "--stderr-dir",
        type=str,
        default="slurm_err",
        help="Directory containing Slurm stderr log files",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    # Collect all trials with status="submitted" and a slurm_id
    submitted_trials = []
    for config_name, config in registry["configs"].items():
        for trial_id, trial in config["trials"].items():
            if trial.get("status") == "submitted" and trial.get("slurm_id"):
                submitted_trials.append((config_name, trial_id, trial))

    if not submitted_trials:
        print("No submitted trials to check.")
        return

    # Gather unique slurm_ids and query sacct
    slurm_ids = list({t[2]["slurm_id"] for t in submitted_trials})
    print(f"Querying sacct for {len(slurm_ids)} job(s)...")
    sacct_results = query_sacct(slurm_ids)

    # Classify each trial and update registry
    classified_counts = {"completed": 0, "cuda_oom": 0, "system_oom": 0,
                         "timeout": 0, "other": 0, "unknown": 0}

    for config_name, trial_id, trial in submitted_trials:
        slurm_id = trial["slurm_id"]

        if slurm_id not in sacct_results:
            # Job not yet visible in sacct (may still be running or pending)
            classified_counts["unknown"] += 1
            continue

        state, exit_code = sacct_results[slurm_id]

        # Still running or pending
        if state in ("RUNNING", "PENDING", "REQUEUED"):
            classified_counts["unknown"] += 1
            continue

        stderr_tail = read_stderr_tail(slurm_id, args.stderr_dir)
        classification = classify_job(state, exit_code, stderr_tail)

        # Update the trial in the registry
        trial_ref = registry["configs"][config_name]["trials"][trial_id]
        if classification == "completed":
            trial_ref["status"] = "completed"
        else:
            trial_ref["status"] = "failed"
            trial_ref["error"] = classification

        classified_counts[classification] += 1

    # Report
    print(f"Classification results:")
    for category, count in sorted(classified_counts.items()):
        if count > 0:
            print(f"  {category}: {count}")

    # Save registry
    atomic_json_dump(registry, registry_path)
    print(f"Registry updated at {registry_path}")


if __name__ == "__main__":
    main()
