#!/bin/bash
# Unit tests for bin/sentinel.sh — focuses on state detection so the
# silent-OOM regression on Phase-1 S3 (where all 64 cgroup-v2 OOM-killed
# tasks reported "0:125" and the sentinel exited 0 without resubmitting)
# cannot recur.
#
# Usage:  bash unit_test/test_sentinel.sh
# Exit:   0 if all pass, non-zero on first failure.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR_TEST=$(mktemp -d)
trap 'rm -rf "$TMPDIR_TEST"' EXIT

FAILS=0
PASSES=0

assert_eq() {
    local expected="$1" actual="$2" label="$3"
    if [ "$expected" = "$actual" ]; then
        PASSES=$((PASSES + 1))
        echo "PASS: $label"
    else
        FAILS=$((FAILS + 1))
        echo "FAIL: $label"
        echo "  expected: $expected"
        echo "  actual:   $actual"
    fi
}

# Build a mock `sacct` that emits whatever the test setup writes to
# $TMPDIR_TEST/sacct_output. Prepend TMPDIR_TEST to PATH so this mock
# shadows the real sacct (which isn't installed locally anyway).
cat > "$TMPDIR_TEST/sacct" <<'SACCT_MOCK'
#!/bin/bash
cat "${SACCT_MOCK_OUTPUT:-/dev/null}"
SACCT_MOCK
chmod +x "$TMPDIR_TEST/sacct"
export PATH="$TMPDIR_TEST:$PATH"

# Source sentinel.sh — its main() is gated on direct execution, so sourcing
# only loads the helpers (get_state, get_exit_code, ...).
# shellcheck source=../bin/sentinel.sh
source "$REPO_ROOT/bin/sentinel.sh"

# --- Test 1: all-OUT_OF_MEMORY array (the silent-failure regression) ---
SACCT_MOCK_OUTPUT="$TMPDIR_TEST/case_oom.txt"
{
    for i in $(seq 0 3); do
        echo "14804157_${i}|OUT_OF_MEMORY"
        echo "14804157_${i}.batch|OUT_OF_MEMORY"
        echo "14804157_${i}.extern|COMPLETED"
    done
} > "$SACCT_MOCK_OUTPUT"
export SACCT_MOCK_OUTPUT
assert_eq "OUT_OF_MEMORY" "$(get_state 14804157)" "get_state: all-OOM array returns OUT_OF_MEMORY"

# get_exit_code reads ExitCode (col 2), which for OUT_OF_MEMORY is "0:125";
# the awk in get_exit_code parses the integer before the colon, so this
# returns 0. The state-based match is what saves the System-OOM branch.
SACCT_MOCK_OUTPUT="$TMPDIR_TEST/case_oom_exit.txt"
{
    for i in $(seq 0 3); do
        echo "14804157_${i}|0:125"
        echo "14804157_${i}.batch|0:125"
        echo "14804157_${i}.extern|0:0"
    done
} > "$SACCT_MOCK_OUTPUT"
assert_eq "0" "$(get_exit_code 14804157)" "get_exit_code: '0:125' parses to 0 (motivates state-based detection)"

# --- Test 2: COMPLETED array still returns COMPLETED (regression guard) ---
SACCT_MOCK_OUTPUT="$TMPDIR_TEST/case_done.txt"
{
    for i in $(seq 0 3); do
        echo "9999999_${i}|COMPLETED"
        echo "9999999_${i}.batch|COMPLETED"
        echo "9999999_${i}.extern|COMPLETED"
    done
} > "$SACCT_MOCK_OUTPUT"
assert_eq "COMPLETED" "$(get_state 9999999)" "get_state: all-COMPLETED array returns COMPLETED"

# --- Test 3: TIMEOUT dominates OUT_OF_MEMORY in the ladder ---
SACCT_MOCK_OUTPUT="$TMPDIR_TEST/case_mixed.txt"
{
    echo "8888888_0|TIMEOUT"
    echo "8888888_0.batch|TIMEOUT"
    echo "8888888_1|OUT_OF_MEMORY"
    echo "8888888_1.batch|OUT_OF_MEMORY"
    echo "8888888_2|COMPLETED"
    echo "8888888_2.batch|COMPLETED"
} > "$SACCT_MOCK_OUTPUT"
assert_eq "TIMEOUT" "$(get_state 8888888)" "get_state: TIMEOUT outranks OUT_OF_MEMORY"

# --- Test 4: OUT_OF_MEMORY outranks FAILED ---
SACCT_MOCK_OUTPUT="$TMPDIR_TEST/case_oom_vs_failed.txt"
{
    echo "7777777_0|OUT_OF_MEMORY"
    echo "7777777_0.batch|OUT_OF_MEMORY"
    echo "7777777_1|FAILED"
    echo "7777777_1.batch|FAILED"
} > "$SACCT_MOCK_OUTPUT"
assert_eq "OUT_OF_MEMORY" "$(get_state 7777777)" "get_state: OUT_OF_MEMORY outranks FAILED"

# --- Test 5: script_needs_gpu detects GPU and CPU job scripts ---
GPU_SCRIPT="$TMPDIR_TEST/fake_gpu_job.sh"
CPU_SCRIPT="$TMPDIR_TEST/fake_cpu_job.sh"
cat > "$GPU_SCRIPT" <<'GPUEOF'
#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --mem=128G
echo hi
GPUEOF
cat > "$CPU_SCRIPT" <<'CPUEOF'
#!/bin/bash
#SBATCH --mem=4G
echo hi
CPUEOF
if script_needs_gpu "$GPU_SCRIPT"; then
    PASSES=$((PASSES + 1)); echo "PASS: script_needs_gpu: detects --gpus directive"
else
    FAILS=$((FAILS + 1)); echo "FAIL: script_needs_gpu: missed --gpus directive in $GPU_SCRIPT"
fi
if script_needs_gpu "$CPU_SCRIPT"; then
    FAILS=$((FAILS + 1)); echo "FAIL: script_needs_gpu: false positive on CPU-only $CPU_SCRIPT"
else
    PASSES=$((PASSES + 1)); echo "PASS: script_needs_gpu: returns false on CPU-only script"
fi

echo
echo "=== $PASSES passed, $FAILS failed ==="
exit "$FAILS"
