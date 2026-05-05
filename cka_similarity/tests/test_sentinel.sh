#!/bin/bash
# Minimal smoke test for double_slurm_time and double_mem helpers.
set -e
source bin/sentinel.sh 2>/dev/null || true   # source to expose functions; ignore main exec

# Test double_slurm_time -- HH:MM:SS form
result=$(double_slurm_time "01:30:00")
[ "$result" = "03:00:00" ] || { echo "FAIL: double_slurm_time 01:30:00 -> $result"; exit 1; }
result=$(double_slurm_time "00:25:00")
[ "$result" = "00:50:00" ] || { echo "FAIL: double_slurm_time 00:25:00 -> $result"; exit 1; }

# Test double_slurm_time -- D-HH:MM:SS form (SLURM day format)
result=$(double_slurm_time "1-00:00:00")
[ "$result" = "2-00:00:00" ] || { echo "FAIL: double_slurm_time 1-00:00:00 -> $result"; exit 1; }

# Test double_slurm_time -- 12h*2 = 24h, output in D-HH:MM:SS form
result=$(double_slurm_time "12:00:00")
[ "$result" = "1-00:00:00" ] || { echo "FAIL: double_slurm_time 12:00:00 -> $result"; exit 1; }

# Test double_mem
result=$(double_mem "96G")
[ "$result" = "192G" ] || { echo "FAIL: double_mem 96G -> $result"; exit 1; }
result=$(double_mem "16G")
[ "$result" = "32G" ] || { echo "FAIL: double_mem 16G -> $result"; exit 1; }

echo "PASS: sentinel helpers"
