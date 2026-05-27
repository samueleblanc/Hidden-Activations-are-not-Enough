#!/bin/bash
# bin/sentinel.sh -- post-job hook for OOM/timeout/CUDA-OOM handling.
#
# Usage (called by run_pipeline.sh after sbatch returns the job id):
#   bash bin/sentinel.sh [--account=<acct> | --account-gpu=<acct> --account-cpu=<acct>] \
#                        <job_id> <job_script> [calibration_path]
#
# --account=<acct>      Optional, back-compat. Same account for the GPU
#                       resubmit AND the CPU recursive sentinel wrap.
# --account-gpu=<acct>  Account used when resubmitting a GPU job (script
#                       has #SBATCH --gpus). Required on clusters where the
#                       principal alias does not auto-route to _gpu / _cpu
#                       at submission time (every Nibi account except
#                       def-assem). May be combined with --account-cpu.
# --account-cpu=<acct>  Account used when resubmitting a CPU job AND for
#                       the recursive sentinel-wrap launcher.
#
# calibration_path  May be a single calibration.json file OR a directory
#                   containing {arch}_imagenet/calibration.json files.
#                   On CUDA OOM the sentinel steps down EVERY calibration
#                   found under the directory (all archs). This is safe
#                   because Phase-1 workers iterate all archs per chunk,
#                   so a CUDA OOM cannot be attributed to a single arch.
#
# Behavior:
#   - System OOM (exit 137, state=OUT_OF_MEMORY, or oom-killer in stderr):
#     resubmit with --mem (or --mem-per-cpu) doubled. Fails loudly if
#     neither directive is present.
#   - Timeout (exit 124 / "DUE TO TIME LIMIT"): resubmit at the SAME wall
#     (inherits #SBATCH --time from the job script — 8h cap per repo policy,
#     see feedback_max_8h_walls). Relies on per-checkpoint resume in the
#     producing workers; slow workloads finish via 2-3 chained 8h slots.
#     NEVER bumps --time — that's the explicit trade-off in d73bcc6.
#   - CUDA OOM (stderr contains "CUDA out of memory"):
#       step calibration tier down (93->90->85) for every discovered
#       calibration.json; fail if every calibration is already at 85.
#       resubmit with same --mem/--time
#   - Other failure: retry once with same params; fail loudly on second attempt
#   - Success (exit 0): exit 0 silently
#
# All errors are logged to overall_errors.json (atomic append).
#
# The script is sourceable: helpers (double_slurm_time, double_mem,
# get_exit_code, get_state, retry_marker, log_error, script_needs_gpu,
# sbatch_with_account) are defined as functions and only invoked from
# main(); main() is gated on the script being executed directly so tests
# can `source bin/sentinel.sh` to expose helpers without triggering
# execution.

set -euo pipefail

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
    # For array jobs, scan ALL task rows and return the maximum exit code.
    # head -1 on the first task is wrong: it can return 0 (n_done-skip task)
    # while other tasks failed. Filter out .batch/.extern step rows since
    # they duplicate the parent task's exit code.
    sacct -j "$1" --format=JobID,ExitCode --noheader -P 2>/dev/null \
        | awk -F'|' '$1 !~ /\.(batch|extern)$/ {split($2,a,":"); if (a[1]+0 > max) max=a[1]+0} END {print max+0}'
}

get_state() {
    # Worst-case state across all array tasks:
    # TIMEOUT > OUT_OF_MEMORY > FAILED > CANCELLED > COMPLETED.
    # OUT_OF_MEMORY must be matched here because cgroup-v2 OOM-kills report
    # exit "0:125" (parsed to 0 by get_exit_code), so the state string is
    # the only signal the System-OOM branch in main() can rely on.
    sacct -j "$1" --format=JobID,State --noheader -P 2>/dev/null \
        | awk -F'|' '$1 !~ /\.(batch|extern)$/ {gsub(/^ +| +$/,"",$2); if ($2=="TIMEOUT") to=1; else if ($2=="OUT_OF_MEMORY") oo=1; else if ($2=="FAILED") fa=1; else if ($2=="CANCELLED") ca=1; else if ($2=="COMPLETED") co=1} END {if (to) print "TIMEOUT"; else if (oo) print "OUT_OF_MEMORY"; else if (fa) print "FAILED"; else if (ca) print "CANCELLED"; else if (co) print "COMPLETED"; else print "UNKNOWN"}'
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

script_needs_gpu() {
    # Returns 0 (true) if the job script requests a GPU via #SBATCH --gpus
    # or #SBATCH --gres=gpu — so the sentinel knows whether to bill the
    # resubmit on the GPU or CPU account variant.
    grep -qE "^#SBATCH (--gpus|--gres=gpu)" "$1" 2>/dev/null
}

sbatch_with_account() {
    # Wraps sbatch to inject --account based on JOB_SCRIPT's resource type.
    # SENTINEL_ACCOUNT_GPU is used for GPU job scripts, SENTINEL_ACCOUNT_CPU
    # for CPU ones. If only the legacy single SENTINEL_ACCOUNT is set, it
    # applies to both.
    local acct
    if script_needs_gpu "$JOB_SCRIPT"; then
        acct="${SENTINEL_ACCOUNT_GPU:-${SENTINEL_ACCOUNT:-}}"
    else
        acct="${SENTINEL_ACCOUNT_CPU:-${SENTINEL_ACCOUNT:-}}"
    fi
    if [ -n "$acct" ]; then
        sbatch --account="$acct" "$@"
    else
        sbatch "$@"
    fi
}

resubmit_with_sentinel() {
    # Resubmit the failed job AND attach a fresh sentinel to the resubmitted
    # job, so subsequent timeouts/OOMs/CUDA-OOMs are also auto-handled.
    # Without this, a job that times out would be resubmitted ONCE; if the
    # resubmit also times out, no further auto-handling occurs.
    local NEW_JOB_ID
    NEW_JOB_ID=$(sbatch_with_account --parsable "$@" "$JOB_SCRIPT")
    if [ -z "$NEW_JOB_ID" ]; then
        echo "FAIL: sbatch resubmit returned empty job id" >&2
        log_error "resubmit_failed" "args=$*"
        exit 1
    fi
    echo "Resubmitted as $NEW_JOB_ID; attaching recursive sentinel" >&2
    # Recursive wrap is always CPU (4G, 5min, no --gpus); bill it on
    # ACCOUNT_CPU. Propagate both GPU and CPU accounts to the inner call
    # so it can route the next resubmit correctly.
    local wrap_acct="${SENTINEL_ACCOUNT_CPU:-${SENTINEL_ACCOUNT:-}}"
    local inner_flags=""
    [ -n "${SENTINEL_ACCOUNT_GPU:-}" ] && inner_flags="$inner_flags --account-gpu=$SENTINEL_ACCOUNT_GPU"
    [ -n "${SENTINEL_ACCOUNT_CPU:-}" ] && inner_flags="$inner_flags --account-cpu=$SENTINEL_ACCOUNT_CPU"
    [ -z "$inner_flags" ] && [ -n "${SENTINEL_ACCOUNT:-}" ] && inner_flags="--account=$SENTINEL_ACCOUNT"
    sbatch ${wrap_acct:+--account=$wrap_acct} \
        --parsable --time=00:05:00 --mem=4G \
        --dependency=afterany:"$NEW_JOB_ID" \
        --wrap="bash bin/sentinel.sh $inner_flags $NEW_JOB_ID $JOB_SCRIPT ${CALIB_FILE:-}" \
        > /dev/null 2>&1 || echo "Warning: failed to attach recursive sentinel to $NEW_JOB_ID" >&2
}

# --- main entry point ---

main() {
    # Parse optional --account / --account-gpu / --account-cpu flags.
    # --account=X is back-compat (applies to both); --account-gpu=X and
    # --account-cpu=X let the caller route GPU resubmits and CPU recursive
    # wraps to different accounts (needed on clusters where the principal
    # alias does not auto-route by partition, e.g. def-amorales on Nibi).
    SENTINEL_ACCOUNT=""
    SENTINEL_ACCOUNT_GPU=""
    SENTINEL_ACCOUNT_CPU=""
    while [[ "${1:-}" == --account* ]]; do
        case "$1" in
            --account=*)      SENTINEL_ACCOUNT="${1#--account=}" ;;
            --account-gpu=*)  SENTINEL_ACCOUNT_GPU="${1#--account-gpu=}" ;;
            --account-cpu=*)  SENTINEL_ACCOUNT_CPU="${1#--account-cpu=}" ;;
            *) echo "FAIL: unknown account flag: $1" >&2; exit 1 ;;
        esac
        shift
    done

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
        if [ -z "$CALIB_FILE" ]; then
            echo "FAIL: CUDA OOM but no calibration path provided to sentinel" >&2
            exit 1
        fi
        # Build the list of calibrations to step down.
        local -a CALIB_LIST
        if [ -d "$CALIB_FILE" ]; then
            # Directory mode: step down every */calibration.json under it.
            mapfile -t CALIB_LIST < <(find "$CALIB_FILE" -mindepth 2 -maxdepth 2 -name 'calibration.json' 2>/dev/null)
        elif [ -f "$CALIB_FILE" ]; then
            CALIB_LIST=("$CALIB_FILE")
        else
            echo "FAIL: CUDA OOM but $CALIB_FILE is neither a file nor a directory" >&2
            exit 1
        fi
        if [ ${#CALIB_LIST[@]} -eq 0 ]; then
            echo "FAIL: CUDA OOM but no calibration files found under $CALIB_FILE" >&2
            exit 1
        fi
        local any_stepped=false
        for cal in "${CALIB_LIST[@]}"; do
            next_tier=$(python3 -c "from bin.calibrate import step_down_tier; print(step_down_tier('$cal'))")
            if [ -n "$next_tier" ]; then
                echo "CUDA OOM: stepped active tier to $next_tier for $cal" >&2
                any_stepped=true
            else
                echo "CUDA OOM: $cal is at lowest tier; cannot step down" >&2
            fi
        done
        if [ "$any_stepped" != true ]; then
            echo "FAIL: CUDA OOM but every calibration is already at the lowest tier" >&2
            exit 1
        fi
        resubmit_with_sentinel
        exit 0
    fi

    # System OOM
    # state=OUT_OF_MEMORY catches cgroup-v2 OOM-kills, which report
    # exit "0:125" (not 137) and whose stderr is empty after SIGKILL.
    if [ "$exit_code" = "137" ] || [ "$state" = "OUT_OF_MEMORY" ] \
        || echo "$stderr_text" | grep -qE "out-of-memory|oom-killer|MemoryError"; then
        log_error "system_oom" "exit_code=$exit_code state=$state"
        # Look for --mem= first; fall back to --mem-per-cpu=. SLURM accepts
        # either, and the sentinel must double whichever the script declares.
        # Failing loudly when neither is present beats sbatch'ing with an
        # empty --mem= arg (which silently picks the partition default).
        current_mem=$(grep -E "^#SBATCH --mem=" "$JOB_SCRIPT" | head -1 | sed 's/.*--mem=//')
        if [ -n "$current_mem" ]; then
            new_mem=$(double_mem "$current_mem")
            echo "System OOM: doubling --mem from $current_mem to $new_mem" >&2
            resubmit_with_sentinel --mem="$new_mem"
            exit 0
        fi
        current_mem=$(grep -E "^#SBATCH --mem-per-cpu=" "$JOB_SCRIPT" | head -1 | sed 's/.*--mem-per-cpu=//')
        if [ -n "$current_mem" ]; then
            new_mem=$(double_mem "$current_mem")
            echo "System OOM: doubling --mem-per-cpu from $current_mem to $new_mem" >&2
            resubmit_with_sentinel --mem-per-cpu="$new_mem"
            exit 0
        fi
        echo "FAIL: no mem directive found in $JOB_SCRIPT -- cannot escalate memory" >&2
        log_error "system_oom_no_mem_directive" "$JOB_SCRIPT"
        exit 1
    fi

    # Timeout
    # Per repo policy (see feedback_max_8h_walls.md): NEVER bump --time past
    # the 8h ceiling. Resubmit at the SAME wall and rely on per-checkpoint
    # resume (validate_theorem45 per-pair flush every 20; cross_model per-KM
    # flush every 50; S2/S3 per-chunk JSONs; A2 pairs.pth periodic atomic
    # save). The resubmit inherits --time from the job script's #SBATCH
    # directive, so omitting --time here pins it at the 8h ceiling.
    # Slow workloads finish via 2-3 sentinel-chained 8h slots, not via
    # walltime escalation.
    if [ "$state" = "TIMEOUT" ] || [ "$exit_code" = "124" ] || echo "$stderr_text" | grep -q "DUE TO TIME LIMIT"; then
        log_error "timeout" "state=$state"
        echo "Timeout: resubmitting at the same 8h wall (relying on per-checkpoint resume)" >&2
        resubmit_with_sentinel
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
        resubmit_with_sentinel
        exit 0
    fi

    exit 0
}

# Only execute main when this script is run directly (not sourced).
if [ "${BASH_SOURCE[0]:-$0}" = "${0}" ]; then
    main "$@"
fi
