"""Tests for slurm.hp_error_collector error classification and sacct parsing."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from slurm.hp_error_collector import classify_job, parse_sacct_line


# ---------------------------------------------------------------------------
# classify_job
# ---------------------------------------------------------------------------


class TestClassifyExitCodeCudaOom:
    def test_classify_exit_code_cuda_oom(self):
        """Exit code 42 (main) with FAILED state -> cuda_oom."""
        result = classify_job(state="FAILED", exit_code="42:0", stderr_tail="")
        assert result == "cuda_oom"


class TestClassifyExitCodeSystemOom:
    def test_classify_exit_code_system_oom(self):
        """OUT_OF_MEMORY state -> system_oom."""
        result = classify_job(
            state="OUT_OF_MEMORY", exit_code="0:137", stderr_tail=""
        )
        assert result == "system_oom"


class TestClassifyExitCodeTimeout:
    def test_classify_exit_code_timeout(self):
        """TIMEOUT state -> timeout."""
        result = classify_job(state="TIMEOUT", exit_code="0:140", stderr_tail="")
        assert result == "timeout"


class TestClassifyCompleted:
    def test_classify_completed(self):
        """COMPLETED + 0:0 -> completed."""
        result = classify_job(state="COMPLETED", exit_code="0:0", stderr_tail="")
        assert result == "completed"


class TestClassifyStderrCudaOom:
    def test_classify_stderr_cuda_oom(self):
        """CUDA out of memory in stderr -> cuda_oom."""
        result = classify_job(
            state="FAILED",
            exit_code="1:0",
            stderr_tail="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        )
        assert result == "cuda_oom"

    def test_classify_stderr_torch_oom(self):
        """torch.cuda.OutOfMemoryError in stderr -> cuda_oom."""
        result = classify_job(
            state="FAILED",
            exit_code="1:0",
            stderr_tail="torch.cuda.OutOfMemoryError: CUDA out of memory.",
        )
        assert result == "cuda_oom"


class TestClassifyUnknownFailure:
    def test_classify_unknown_failure(self):
        """Generic failure with unrecognized error -> other."""
        result = classify_job(
            state="FAILED", exit_code="1:0", stderr_tail="KeyError: 'foo'"
        )
        assert result == "other"


class TestClassifyStderrSystemOom:
    def test_classify_stderr_oom_kill(self):
        """oom-kill in stderr -> system_oom."""
        result = classify_job(
            state="FAILED",
            exit_code="1:0",
            stderr_tail="slurmstepd: oom-kill event in step 12345.0",
        )
        assert result == "system_oom"

    def test_classify_stderr_out_of_memory(self):
        """Out of memory in stderr -> system_oom."""
        result = classify_job(
            state="FAILED",
            exit_code="1:0",
            stderr_tail="Out of memory: Killed process 12345",
        )
        assert result == "system_oom"


class TestClassifyExitCode137:
    def test_classify_exit_code_137_main(self):
        """Exit code 137 in main position -> system_oom."""
        result = classify_job(state="FAILED", exit_code="137:0", stderr_tail="")
        assert result == "system_oom"

    def test_classify_exit_code_137_signal(self):
        """Exit code 137 in signal position -> system_oom."""
        result = classify_job(state="FAILED", exit_code="0:137", stderr_tail="")
        assert result == "system_oom"


class TestClassifyExitCode140:
    def test_classify_exit_code_140_main(self):
        """Exit code 140 in main position -> timeout."""
        result = classify_job(state="FAILED", exit_code="140:0", stderr_tail="")
        assert result == "timeout"

    def test_classify_exit_code_140_signal(self):
        """Exit code 140 in signal position -> timeout."""
        result = classify_job(state="FAILED", exit_code="0:140", stderr_tail="")
        assert result == "timeout"


# ---------------------------------------------------------------------------
# parse_sacct_line
# ---------------------------------------------------------------------------


class TestParseSacctLine:
    def test_parse_sacct_line(self):
        """Pipe-delimited sacct line parsed correctly."""
        job_id, state, exit_code = parse_sacct_line("12345|COMPLETED|0:0")
        assert job_id == "12345"
        assert state == "COMPLETED"
        assert exit_code == "0:0"

    def test_parse_sacct_line_failed(self):
        """Failed job line parsed correctly."""
        job_id, state, exit_code = parse_sacct_line("67890|FAILED|42:0")
        assert job_id == "67890"
        assert state == "FAILED"
        assert exit_code == "42:0"

    def test_parse_sacct_line_timeout(self):
        """Timeout job line parsed correctly."""
        job_id, state, exit_code = parse_sacct_line("11111|TIMEOUT|0:140")
        assert job_id == "11111"
        assert state == "TIMEOUT"
        assert exit_code == "0:140"
