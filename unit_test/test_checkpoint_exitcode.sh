#!/bin/bash
# Test that checkpoint JSON includes exit_code and handles failed status correctly
set -euo pipefail

PASS=0
FAIL=0

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc (expected='$expected', got='$actual')"
        FAIL=$((FAIL + 1))
    fi
}

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Source the functions we're testing (suppress side effects from module load etc.)
source experiment_config.sh 2>/dev/null || true

echo "=== Test: read_checkpoint_status with failed status ==="
echo '{"status":"failed","exit_code":137,"mem":"16G","timestamp":"2026-03-25T00:00:00"}' > "$TMPDIR/test_failed.json"
STATUS=$(read_checkpoint_status "$TMPDIR/test_failed.json")
assert_eq "failed checkpoint returns 'failed'" "failed" "$STATUS"

echo "=== Test: read_checkpoint_status with complete status ==="
echo '{"status":"complete","exit_code":0,"timestamp":"2026-03-25T00:00:00"}' > "$TMPDIR/test_complete.json"
STATUS=$(read_checkpoint_status "$TMPDIR/test_complete.json")
assert_eq "complete checkpoint returns 'complete'" "complete" "$STATUS"

echo "=== Test: read_checkpoint_status with missing file ==="
STATUS=$(read_checkpoint_status "$TMPDIR/nonexistent.json")
assert_eq "missing file returns 'missing'" "missing" "$STATUS"

echo "=== Test: checkpoint JSON has exit_code field ==="
echo '{"status":"failed","exit_code":137,"mem":"16G","timestamp":"2026-03-25T00:00:00"}' > "$TMPDIR/test_exitcode.json"
EXIT_CODE=$(python3 -c "import json; print(json.load(open('$TMPDIR/test_exitcode.json')).get('exit_code', 'MISSING'))")
assert_eq "exit_code field present in failed checkpoint" "137" "$EXIT_CODE"

echo '{"status":"complete","exit_code":0,"timestamp":"2026-03-25T00:00:00"}' > "$TMPDIR/test_exitcode_ok.json"
EXIT_CODE=$(python3 -c "import json; print(json.load(open('$TMPDIR/test_exitcode_ok.json')).get('exit_code', 'MISSING'))")
assert_eq "exit_code field present in complete checkpoint" "0" "$EXIT_CODE"

echo "=== Test: enforce_min_mem works ==="
RESULT=$(enforce_min_mem "2G" "16G")
assert_eq "2G enforced to 16G" "16G" "$RESULT"

RESULT=$(enforce_min_mem "32G" "16G")
assert_eq "32G stays at 32G" "32G" "$RESULT"

echo "=== Test: double_mem works ==="
RESULT=$(double_mem "16G")
assert_eq "16G doubled to 32G" "32G" "$RESULT"

RESULT=$(double_mem "2G")
assert_eq "2G doubled to 4G" "4G" "$RESULT"

RESULT=$(double_mem "128G")
assert_eq "128G doubled to 256G" "256G" "$RESULT"

echo "=== Test: read_checkpoint_field works ==="
echo '{"status":"failed","exit_code":137,"mem":"16G","timestamp":"2026-03-25T00:00:00"}' > "$TMPDIR/test_oom.json"
EXIT_CODE=$(read_checkpoint_field "$TMPDIR/test_oom.json" "exit_code")
assert_eq "read exit_code from OOM checkpoint" "137" "$EXIT_CODE"
MEM=$(read_checkpoint_field "$TMPDIR/test_oom.json" "mem")
assert_eq "read mem from OOM checkpoint" "16G" "$MEM"
DOUBLED=$(double_mem "$MEM")
assert_eq "OOM mem doubled correctly" "32G" "$DOUBLED"

echo "=== Test: read_checkpoint_field on missing file ==="
RESULT=$(read_checkpoint_field "$TMPDIR/nonexistent.json" "mem")
assert_eq "missing file returns empty" "" "$RESULT"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
