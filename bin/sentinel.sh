#!/bin/bash
# bin/sentinel.sh -- post-job hook for OOM/timeout/CUDA-OOM handling.
#
# Usage (called by run_pipeline.sh after sbatch returns the job id):
#   bash bin/sentinel.sh <job_id> <job_script> [calibration_file]
#
# Behavior:
#   - System OOM (exit 137 / oom-killer): resubmit with --mem (or
#     --mem-per-cpu) doubled. Fails loudly if neither directive is present.
#   - Timeout (exit 124 / "DUE TO TIME LIMIT"): resubmit with --time doubled.
#     Time strings can be "HH:MM:SS" or "D-HH:MM:SS" (SLURM day form).
#   - CUDA OOM (stderr contains "CUDA out of memory"):
#       step calibration tier down (93->90->85); fail if already 85
#       resubmit with same --mem/--time
#   - Other failure: retry once with same params; fail loudly on second attempt
#   - Success (exit 0): exit 0 silently
#
# All errors are logged to overall_errors.json (atomic append).
#
# The script is sourceable: helpers (double_slurm_time, double_mem,
# get_exit_code, get_state, retry_marker, log_error) are defined as
# functions and only invoked from main(); main() is gated on the script
# being executed directly so tests can `source bin/sentinel.sh` to expose
# helpers without triggering execution.

set -uo pipefail

LOG_DIR="${LOG_DIR:-slurm_err}"
ERR_LOG="${ERR_LOG:-overall_errors.json}"

# --- helpers (always defined; safe to source) ---

log_error() {
    local kind="$1"
    local detail="$2"
    local job_id="${JOB_ID:-unknown}"
    local script="${JOB_SCRIPT:-unknown}"
    python3 - "$ERR_LOG" "$kind" "$detail" "$job_id" "$script" <<'PYEOF'
import json, os, sys, time, fcntl
err_log, kind, detail, job_id, script = sys.argv[1:6]
with open(err_log, 'a+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    try:
        data = json.loads(f.read() or '[]')
    except json.JSONDecodeError:
        data = []
    data.append({
        'ts': time.time(),
        'kind': kind,
        'detail': detail,
        'job_id': job_id,
        'script': script,
    })
    f.seek(0); f.truncate()
    f.write(json.dumps(data, indent=2))
PYEOF
}

get_exit_code() {
    sacct -j "$1" --format=ExitCode --noheader 2>/dev/null | head -1 | awk -F: '{print $1}' | tr -d ' '
}

get_state() {
    sacct -j "$1" --format=State --noheader 2>/dev/null | head -1 | tr -d ' '
}

double_slurm_time() {
    local t="$1"
    python3 - "$t" <<'PYEOF'
import sys
arg = sys.argv[1]

# SLURM accepts D-HH:MM:SS in addition to HH:MM:SS, MM:SS, MM.
days = 0
if '-' in arg:
    day_part, _, rest = arg.partition('-')
    days = int(day_part)
    arg = rest

parts = arg.split(':')
if len(parts) == 3:
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
elif len(parts) == 2:
    h, m, s = 0, int(parts[0]), int(parts[1])
else:
    h, m, s = 0, 0, int(parts[0])

total = (days*86400 + h*3600 + m*60 + s) * 2
out_days, rem = divmod(total, 86400)
out_h, rem = divmod(rem, 3600)
out_m, out_s = divmod(rem, 60)

# Use D-HH:MM:SS when total >= 24h, else HH:MM:SS for backward compat.
if out_days > 0:
    print(f'{out_days}-{out_h:02d}:{out_m:02d}:{out_s:02d}')
else:
    print(f'{out_h:02d}:{out_m:02d}:{out_s:02d}')
PYEOF
}

double_mem() {
    # mem like '96G'; double the integer prefix while preserving the unit.
    python3 - "$1" <<'PYEOF'
import sys, re
m = re.match(r'(\d+)([A-Za-z])', sys.argv[1])
if m:
    print(f'{int(m.group(1))*2}{m.group(2)}')
else:
    print(sys.argv[1])
PYEOF
}

retry_marker() {
    # Prefer JOB_SCRIPT; fall back to a generic name when called outside main.
    local script="${JOB_SCRIPT:-unknown_script}"
    echo ".retried_$(echo "$script" | tr '/' '_').marker"
}

# --- main entry point ---

main() {
    JOB_ID="$1"
    JOB_SCRIPT="$2"
    CALIB_FILE="${3:-}"

    local exit_code state stderr_file stderr_text current_mem new_mem
    local current_time new_time next_tier

    exit_code=$(get_exit_code "$JOB_ID")
    state=$(get_state "$JOB_ID")
    stderr_file=$(ls "$LOG_DIR"/*"${JOB_ID}"* 2>/dev/null | head -1)
    stderr_text=""
    [ -n "$stderr_file" ] && stderr_text=$(cat "$stderr_file" 2>/dev/null || echo "")

    if [ "$exit_code" = "0" ] && [ "$state" = "COMPLETED" ]; then
        rm -f "$(retry_marker)"   # success -- reset retry marker
        exit 0
    fi

    # CUDA OOM
    if echo "$stderr_text" | grep -qE "CUDA (out of memory|runtime error.*out of memory)"; then
        log_error "cuda_oom" "$(echo "$stderr_text" | grep -i 'out of memory' | head -3 | tr '\n' ' ')"
        if [ -z "$CALIB_FILE" ] || [ ! -f "$CALIB_FILE" ]; then
            echo "FAIL: CUDA OOM but no calibration file provided to sentinel" >&2
            exit 1
        fi
        next_tier=$(python3 -c "from bin.calibrate import step_down_tier; print(step_down_tier('$CALIB_FILE'))")
        if [ -z "$next_tier" ]; then
            echo "FAIL: CUDA OOM at lowest tier; calibration is broken" >&2
            exit 1
        fi
        echo "CUDA OOM: stepped active tier to $next_tier for $CALIB_FILE; resubmitting" >&2
        sbatch "$JOB_SCRIPT"
        exit 0
    fi

    # System OOM
    if [ "$exit_code" = "137" ] || echo "$stderr_text" | grep -qE "out-of-memory|oom-killer|MemoryError"; then
        log_error "system_oom" "exit_code=$exit_code state=$state"
        # Look for --mem= first; fall back to --mem-per-cpu=. SLURM accepts
        # either, and the sentinel must double whichever the script declares.
        # Failing loudly when neither is present beats sbatch'ing with an
        # empty --mem= arg (which silently picks the partition default).
        current_mem=$(grep -E "^#SBATCH --mem=" "$JOB_SCRIPT" | head -1 | sed 's/.*--mem=//')
        if [ -n "$current_mem" ]; then
            new_mem=$(double_mem "$current_mem")
            echo "System OOM: doubling --mem from $current_mem to $new_mem" >&2
            sbatch --mem="$new_mem" "$JOB_SCRIPT"
            exit 0
        fi
        current_mem=$(grep -E "^#SBATCH --mem-per-cpu=" "$JOB_SCRIPT" | head -1 | sed 's/.*--mem-per-cpu=//')
        if [ -n "$current_mem" ]; then
            new_mem=$(double_mem "$current_mem")
            echo "System OOM: doubling --mem-per-cpu from $current_mem to $new_mem" >&2
            sbatch --mem-per-cpu="$new_mem" "$JOB_SCRIPT"
            exit 0
        fi
        echo "FAIL: no mem directive found in $JOB_SCRIPT -- cannot escalate memory" >&2
        log_error "system_oom_no_mem_directive" "$JOB_SCRIPT"
        exit 1
    fi

    # Timeout
    if [ "$state" = "TIMEOUT" ] || [ "$exit_code" = "124" ] || echo "$stderr_text" | grep -q "DUE TO TIME LIMIT"; then
        log_error "timeout" "state=$state"
        current_time=$(grep -E "^#SBATCH --time=" "$JOB_SCRIPT" | head -1 | sed 's/.*--time=//')
        new_time=$(double_slurm_time "$current_time")
        echo "Timeout: doubling --time from $current_time to $new_time" >&2
        sbatch --time="$new_time" "$JOB_SCRIPT"
        exit 0
    fi

    # Other failure
    if [ "$exit_code" != "0" ]; then
        log_error "other_failure" "exit_code=$exit_code state=$state stderr_head=$(echo "$stderr_text" | head -3 | tr '\n' ' ')"
        if [ -f "$(retry_marker)" ]; then
            echo "FAIL: $JOB_SCRIPT failed twice; aborting" >&2
            exit 1
        fi
        touch "$(retry_marker)"
        sbatch "$JOB_SCRIPT"
        exit 0
    fi

    exit 0
}

# Only execute main when this script is run directly (not sourced).
if [ "${BASH_SOURCE[0]:-$0}" = "${0}" ]; then
    main "$@"
fi
