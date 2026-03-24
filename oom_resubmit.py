"""
Automatic OOM retry: detect OOM failures and resubmit affected pipeline chain
with doubled memory.

Reads overall_errors.json (produced by collect_errors.py), identifies OOM
failures, doubles --mem in the affected Slurm scripts, and resubmits the
downstream dependency chain.

Usage:
    python oom_resubmit.py --experiment alexnet_cifar10
    python oom_resubmit.py --experiment alexnet_cifar10 --test
    python oom_resubmit.py --experiment alexnet_cifar10 --dry-run
"""

import os
import re
import sys
import json
import argparse
import subprocess
from collections import defaultdict


# Pipeline dependency graph: step -> set of upstream steps it depends on
DEPENDS_ON = {
    "A": set(),
    "B": {"A"},
    "C": {"A"},
    "G": {"A"},
    "D": {"C"},
    "E": {"A", "B", "C", "D"},
    "F": {"E", "G"},
    "AUDIT": {"E", "G", "F"},
}

# Topological submission order
TOPO_ORDER = ["A", "B", "C", "G", "D", "E", "F", "AUDIT"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic OOM retry with doubled memory"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False,
                        help="Use test-mode directories")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Maximum OOM retries (default: 2)")
    parser.add_argument("--mem-cap", type=int, default=480,
                        help="Memory cap in GB (default: 480)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print sbatch commands without executing")
    return parser.parse_args()


def load_errors(experiment):
    """Read overall_errors.json and return (oom_entries, all_entries)."""
    path = os.path.join("experiments", experiment, "overall_errors.json")
    if not os.path.isfile(path):
        print(f"No error report found at {path}")
        return [], []

    with open(path) as f:
        report = json.load(f)

    all_entries = report.get("steps", [])
    oom_entries = [
        e for e in all_entries
        if e.get("error_detected") and e.get("error_type") == "oom"
    ]
    return oom_entries, all_entries


def check_retry_count(experiment, max_retries):
    """Read/increment retry counter. Returns current count (before increment).

    Aborts (sys.exit) if max retries already reached.
    """
    counter_path = os.path.join("experiments", experiment, "oom_retry_count")

    current = 0
    if os.path.isfile(counter_path):
        try:
            with open(counter_path) as f:
                current = int(f.read().strip())
        except (ValueError, OSError):
            current = 0

    if current >= max_retries:
        print(f"OOM retry limit reached ({current}/{max_retries}). No further retries.")
        sys.exit(0)

    # Increment
    with open(counter_path, "w") as f:
        f.write(str(current + 1))

    print(f"OOM retry {current + 1}/{max_retries}")
    return current


def step_to_script(experiment, step, chunk, test_mode):
    """Map (step, chunk) to the Slurm script path in orchestrator_jobs/."""
    job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")
    if test_mode:
        job_dir = os.path.join("experiments", experiment, "orchestrator_jobs_test")
        if not os.path.isdir(job_dir):
            job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")

    if step == "A":
        return os.path.join(job_dir, "step_A.sh")
    elif step == "B":
        return os.path.join(job_dir, f"step_B_chunk_{chunk}.sh")
    elif step == "C":
        return os.path.join(job_dir, f"step_C_attack_{chunk}.sh")
    elif step == "D":
        return os.path.join(job_dir, f"step_D_chunk_{chunk}.sh")
    elif step == "E":
        return os.path.join(job_dir, "step_E.sh")
    elif step == "G":
        return os.path.join(job_dir, "step_G.sh")
    elif step == "F":
        return os.path.join(job_dir, "step_F.sh")
    elif step == "AUDIT":
        return os.path.join(job_dir, "final_audit.sh")
    else:
        return None


def double_memory(script_path, mem_cap_gb):
    """Double the --mem value in a Slurm script, capped at mem_cap_gb."""
    with open(script_path) as f:
        content = f.read()

    def replacer(m):
        old_val = int(m.group(1))
        new_val = min(old_val * 2, mem_cap_gb)
        return f"#SBATCH --mem={new_val}G"

    new_content, count = re.subn(r"#SBATCH --mem=(\d+)G", replacer, content)
    if count == 0:
        print(f"  WARNING: No --mem=...G found in {script_path}")
        return None

    with open(script_path, "w") as f:
        f.write(new_content)

    # Extract old and new values for reporting
    old_match = re.search(r"#SBATCH --mem=(\d+)G", content)
    new_match = re.search(r"#SBATCH --mem=(\d+)G", new_content)
    old_val = old_match.group(1) if old_match else "?"
    new_val = new_match.group(1) if new_match else "?"
    return (old_val, new_val)


def get_retry_set(oom_entries, all_entries):
    """Compute the set of (step, chunk) pairs that need resubmission.

    Includes the OOM'd jobs plus all transitive downstream steps.
    Returns: (oom_pairs, downstream_pairs, affected_steps)
      - oom_pairs: set of (step, chunk) that OOM'd (need memory doubling)
      - downstream_pairs: set of (step, chunk) downstream (resubmit as-is)
      - affected_steps: set of step letters in the retry
    """
    # Step-level set of OOM'd steps
    oom_steps = {e["step"] for e in oom_entries}
    oom_pairs = {(e["step"], e.get("chunk")) for e in oom_entries}

    # Propagate downstream: any step whose upstream intersects affected set
    affected_steps = set(oom_steps)
    for step in TOPO_ORDER:
        if step in affected_steps:
            continue
        upstream = DEPENDS_ON.get(step, set())
        if upstream & affected_steps:
            affected_steps.add(step)

    # Build downstream pairs from all_entries
    # For downstream steps: include jobs that didn't complete successfully
    downstream_pairs = set()
    completed_steps = set()
    for e in all_entries:
        if e.get("slurm_state") == "COMPLETED" and not e.get("error_detected"):
            completed_steps.add((e["step"], e.get("chunk")))

    for step in affected_steps:
        if step in oom_steps:
            continue  # OOM'd steps are handled via oom_pairs
        # For downstream steps, find their entries
        step_entries = [e for e in all_entries if e["step"] == step]
        if step_entries:
            for e in step_entries:
                pair = (e["step"], e.get("chunk"))
                if pair not in completed_steps:
                    downstream_pairs.add(pair)
        else:
            # Step never ran (e.g., was cancelled) — submit with no chunk
            downstream_pairs.add((step, None))

    return oom_pairs, downstream_pairs, affected_steps


def submit_job(script_path, dep_ids, dry_run):
    """Submit a Slurm job, optionally with dependencies. Returns job ID."""
    cmd = ["sbatch", "--parsable"]
    if dep_ids:
        dep_str = ":".join(dep_ids)
        cmd.append(f"--dependency=afterok:{dep_str}")
    cmd.append(script_path)

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return f"DRY_{os.path.basename(script_path)}"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR submitting {script_path}: {result.stderr.strip()}")
        return None
    job_id = result.stdout.strip().split(";")[0]  # handle array job output
    return job_id


def submit_retry_chain(experiment, oom_pairs, downstream_pairs, affected_steps,
                       mem_cap_gb, test_mode, dry_run):
    """Submit the retry chain in topological order with correct dependencies."""
    # Merge all pairs for lookup
    all_retry = {}  # step -> list of (step, chunk, is_oom)
    for step, chunk in oom_pairs:
        all_retry.setdefault(step, []).append((step, chunk, True))
    for step, chunk in downstream_pairs:
        all_retry.setdefault(step, []).append((step, chunk, False))

    new_job_ids = defaultdict(list)  # step -> [job_ids]
    all_submitted = []

    for step in TOPO_ORDER:
        entries = all_retry.get(step)
        if not entries:
            continue

        # Build dependency list from upstream retries only
        dep_ids = []
        for upstream in DEPENDS_ON.get(step, set()):
            if upstream in new_job_ids:
                dep_ids.extend(new_job_ids[upstream])

        for _, chunk, is_oom in entries:
            script = step_to_script(experiment, step, chunk, test_mode)
            if script is None or not os.path.isfile(script):
                print(f"  WARNING: Script not found for step {step} chunk {chunk}: {script}")
                continue

            if is_oom:
                mem_info = double_memory(script, mem_cap_gb)
                if mem_info:
                    print(f"  [{step}] Doubled memory: {mem_info[0]}G -> {mem_info[1]}G ({os.path.basename(script)})")

            job_id = submit_job(script, dep_ids, dry_run)
            if job_id:
                new_job_ids[step].append(job_id)
                all_submitted.append(job_id)
                action = "OOM-retry" if is_oom else "downstream"
                chunk_str = f" chunk={chunk}" if chunk is not None else ""
                print(f"  [{step}] Submitted {action}{chunk_str}: {job_id}")

    # Submit error_scan with afterany on ALL retry jobs
    if all_submitted:
        errscan_script = step_to_script(experiment, "ERRSCAN", None, test_mode)
        # error_scan.sh lives directly in orchestrator_jobs
        job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")
        if test_mode:
            test_dir = os.path.join("experiments", experiment, "orchestrator_jobs_test")
            if os.path.isdir(test_dir):
                job_dir = test_dir
        errscan_script = os.path.join(job_dir, "error_scan.sh")

        if os.path.isfile(errscan_script):
            cmd = ["sbatch", "--parsable",
                   f"--dependency=afterany:{':'.join(all_submitted)}",
                   errscan_script]
            if dry_run:
                print(f"  [DRY-RUN] {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    scan_id = result.stdout.strip().split(";")[0]
                    print(f"  [ERRSCAN] Resubmitted error scan: {scan_id} (afterany)")
                else:
                    print(f"  WARNING: Failed to resubmit error_scan: {result.stderr.strip()}")
        else:
            print(f"  WARNING: error_scan.sh not found at {errscan_script}")

    return all_submitted


def main():
    args = parse_args()
    experiment = args.experiment

    print(f"OOM retry check for experiment: {experiment}")

    # Load errors
    oom_entries, all_entries = load_errors(experiment)
    if not oom_entries:
        print("No OOM failures detected. Nothing to retry.")
        return

    print(f"Found {len(oom_entries)} OOM failure(s):")
    for e in oom_entries:
        chunk_str = f" chunk={e.get('chunk')}" if e.get("chunk") is not None else ""
        print(f"  Step {e['step']}{chunk_str} (job {e['job_id']})")

    # Check retry count
    check_retry_count(experiment, args.max_retries)

    # Compute retry set
    oom_pairs, downstream_pairs, affected_steps = get_retry_set(oom_entries, all_entries)
    print(f"Affected steps: {sorted(affected_steps)}")
    if downstream_pairs:
        print(f"Downstream resubmissions: {sorted(downstream_pairs)}")

    # Submit retry chain
    submitted = submit_retry_chain(
        experiment, oom_pairs, downstream_pairs, affected_steps,
        args.mem_cap, args.test, args.dry_run,
    )

    if submitted:
        print(f"Resubmitted {len(submitted)} job(s).")
    else:
        print("No jobs were submitted.")


if __name__ == "__main__":
    main()
