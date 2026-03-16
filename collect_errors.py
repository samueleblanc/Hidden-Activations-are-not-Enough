"""
Collect pipeline errors into experiments/{experiment}/overall_errors.json.

Replaces the inline Python previously embedded in run_experiment.sh.
Can run as a Slurm job or directly on the login node.

Usage:
    python collect_errors.py --experiment alexnet_cifar10
    python collect_errors.py --experiment alexnet_cifar10 --test
    python collect_errors.py --experiment alexnet_cifar10 --include-audit-report
"""

import os
import re
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime

from utils.error_classification import (
    LOG_PATTERN, STEP_LABELS, STEP_ORDER,
    ERROR_CATEGORIES,
    classify_error, get_error_category, extract_traceback, read_tail,
    parse_log_filename,
)


SCHEMA_VERSION = "2.0"

# Patterns that indicate runtime errors in .out files of COMPLETED jobs
RUNTIME_ERROR_RE = re.compile(
    r'ERROR:|FAILED:|Traceback|RuntimeError|AttributeError|urllib\.error|CUDA error'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect pipeline errors into overall_errors.json"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False,
                        help="Use test-mode log directories")
    parser.add_argument("--include-audit-report", action="store_true", default=False,
                        help="Merge integrity summary from audit_report.json")
    return parser.parse_args()


def discover_jobs(experiment, slurm_out_dir, slurm_err_dir):
    """Discover all pipeline jobs for this experiment from log filenames.

    Scans both .out and .err directories to find all job IDs.
    Returns a list of job dicts with associated file paths.
    """
    jobs = {}  # keyed by job_id to deduplicate

    for directory, ext in [(slurm_err_dir, "err"), (slurm_out_dir, "out")]:
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            parsed = parse_log_filename(fname)
            if not parsed:
                continue
            if parsed["exp"] != experiment:
                continue
            if parsed["step"] == "ERRSCAN":
                continue  # skip our own logs

            job_id = parsed["job_id"]
            if job_id not in jobs:
                jobs[job_id] = {
                    "job_id": job_id,
                    "prefix": parsed["prefix"],
                    "step": parsed["step"],
                    "chunk": int(parsed["chunk"]) if parsed["chunk"] else None,
                    "err_file": None,
                    "out_file": None,
                }

            fpath = os.path.join(directory, fname)
            if parsed["ext"] == "err":
                jobs[job_id]["err_file"] = fpath
            elif parsed["ext"] == "out":
                jobs[job_id]["out_file"] = fpath

    return list(jobs.values())


def query_sacct(job_ids):
    """Query sacct for job status. Returns dict keyed by job_id.

    Handles sacct being unavailable (e.g., on login nodes without Slurm).
    """
    if not job_ids:
        return {}
    try:
        ids_str = ",".join(job_ids)
        result = subprocess.run(
            ["sacct", "--jobs=" + ids_str, "--parsable2", "--noheader",
             "--format=JobID,State,ExitCode,Elapsed"],
            capture_output=True, text=True, timeout=30,
        )
        data = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                jid = parts[0].split(".")[0]  # strip .batch suffix
                if jid in job_ids and jid not in data:
                    data[jid] = {
                        "state": parts[1],
                        "exit_code": parts[2],
                        "elapsed": parts[3],
                    }
        return data
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def detect_error(job, sacct_info):
    """Determine if a job has an error and classify it.

    Returns (has_error, error_type, error_category, traceback, err_tail, source_file).
    """
    slurm_state = sacct_info.get("state", "UNKNOWN")
    exit_code = sacct_info.get("exit_code", "?")

    has_error = False
    source_file = None

    # Check Slurm state
    if slurm_state in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"):
        has_error = True

    # Check non-zero exit code
    if not has_error and exit_code not in ("0:0", "?") and ":" in exit_code:
        try:
            main_code = int(exit_code.split(":")[0])
            if main_code != 0:
                has_error = True
        except ValueError:
            pass

    # For failed jobs: scan .err file
    if has_error:
        tail_text = ""
        if job.get("err_file"):
            tail_text = read_tail(job["err_file"])
            source_file = job["err_file"]
        error_type = classify_error(tail_text)
        traceback = extract_traceback(tail_text)
        if error_type is None:
            error_type = "code" if traceback else "unknown"
        return True, error_type, get_error_category(error_type), traceback, tail_text, source_file

    # For COMPLETED/UNKNOWN jobs: scan both .err and .out for runtime errors
    for file_key in ["err_file", "out_file"]:
        fpath = job.get(file_key)
        if not fpath or not os.path.isfile(fpath):
            continue
        content = read_tail(fpath, 50)
        if RUNTIME_ERROR_RE.search(content):
            error_type = classify_error(content)
            traceback = extract_traceback(content)
            if error_type is None:
                error_type = "code" if traceback else "unknown"
            return True, error_type, get_error_category(error_type), traceback, content, fpath

    return False, None, None, None, None, None


def load_audit_report(experiment):
    """Load audit_report.json if it exists."""
    path = os.path.join("experiments", experiment, "audit_report.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_integrity_summary(audit_report):
    """Extract integrity summary from audit_report.json."""
    if not audit_report:
        return None
    summary = audit_report.get("summary", {})
    return {
        "total_checks": summary.get("total", 0),
        "ok": summary.get("ok", 0),
        "missing": summary.get("missing", 0),
        "corrupt": summary.get("corrupt", 0),
    }


def main():
    args = parse_args()
    experiment = args.experiment
    test_mode = args.test

    slurm_out_dir = "slurm_out_test" if test_mode else "slurm_out"
    slurm_err_dir = "slurm_err_test" if test_mode else "slurm_err"
    output_dir = os.path.join("experiments", experiment)

    # Discover jobs
    jobs = discover_jobs(experiment, slurm_out_dir, slurm_err_dir)

    # Query sacct
    job_ids = {j["job_id"] for j in jobs}
    sacct_data = query_sacct(job_ids)

    # Analyze each job
    steps_output = []
    error_types = {}
    error_category_summary = {cat: 0 for cat in ERROR_CATEGORIES}
    jobs_with_errors = 0
    jobs_succeeded = 0

    for job in jobs:
        jid = job["job_id"]
        info = sacct_data.get(jid, {})
        slurm_state = info.get("state", "UNKNOWN")
        exit_code = info.get("exit_code", "?")
        elapsed = info.get("elapsed", "?")
        step = job["step"]
        step_label = STEP_LABELS.get(step, step)

        has_error, error_type, error_category, traceback, err_tail, source_file = \
            detect_error(job, info)

        entry = {
            "step": step,
            "step_label": step_label,
            "chunk": job["chunk"],
            "job_id": jid,
            "slurm_state": slurm_state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "error_detected": has_error,
        }

        if has_error:
            jobs_with_errors += 1
            entry["error_type"] = error_type
            entry["error_category"] = error_category
            if traceback:
                entry["traceback"] = traceback
            if err_tail:
                # Truncate to last 40 lines to keep JSON manageable
                tail_lines = err_tail.strip().split("\n")
                entry["err_tail"] = "\n".join(tail_lines[-40:])
            if source_file:
                entry["err_file"] = os.path.relpath(
                    source_file, os.environ.get("SLURM_SUBMIT_DIR", ".")
                )
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_category_summary[error_category] = \
                error_category_summary.get(error_category, 0) + 1
        else:
            if slurm_state == "COMPLETED":
                jobs_succeeded += 1

        steps_output.append(entry)

    # Sort: errors first, then by step order
    def sort_key(e):
        idx = STEP_ORDER.index(e["step"]) if e["step"] in STEP_ORDER else 99
        return (0 if e["error_detected"] else 1, idx, e.get("chunk") or 0)

    steps_output.sort(key=sort_key)

    pipeline_success = jobs_with_errors == 0

    # Build report
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "test" if test_mode else "normal",
        "pipeline_success": pipeline_success,
        "total_jobs": len(jobs),
        "jobs_succeeded": jobs_succeeded,
        "jobs_with_errors": jobs_with_errors,
        "error_category_summary": error_category_summary,
        "error_types_summary": error_types,
        "steps": steps_output,
    }

    # Optionally include integrity summary from audit_report.json
    if args.include_audit_report:
        audit_report = load_audit_report(experiment)
        integrity = build_integrity_summary(audit_report)
        if integrity:
            report["integrity"] = integrity

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "overall_errors.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    status_str = "SUCCESS" if pipeline_success else "FAILURE"
    print(f"Error scan complete: {status_str}")
    print(f"  Total jobs scanned: {len(jobs)}")
    print(f"  Jobs succeeded:     {jobs_succeeded}")
    print(f"  Jobs with errors:   {jobs_with_errors}")
    if error_types:
        print(f"  Error types:        {error_types}")
    if any(v > 0 for v in error_category_summary.values()):
        nonzero = {k: v for k, v in error_category_summary.items() if v > 0}
        print(f"  Error categories:   {nonzero}")
    print(f"  Report written to:  {out_path}")


if __name__ == "__main__":
    main()
