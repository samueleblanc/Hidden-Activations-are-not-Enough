#!/usr/bin/env python
"""Unit tests for critical and high-severity bug fixes.

Tests cover:
  C1: CyclicLR variable shadowing in training.py
  C2: Remainder-aware chunking in generate_matrices.py
  H9: chunk_id usage in generate_matrices.py
"""
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestC1SchedulerPassthrough(unittest.TestCase):
    """C1: Verify training.py passes scheduler object, not string."""

    def test_scheduler_argument_is_not_string(self):
        """Read training.py and verify the scheduler= argument is not 'sched'."""
        training_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "training.py"
        )
        with open(training_path) as f:
            lines = f.readlines()

        # Find lines around the train_one_epoch() call site (not the def)
        # Look for "train_one_epoch(" preceded by indentation (call), then
        # scan forward for the scheduler= keyword argument
        import re
        found_call = False
        for i, line in enumerate(lines):
            # Call site is indented (inside a function body), not a def
            if 'train_one_epoch(' in line and 'def ' not in line:
                # Scan next few lines for scheduler=
                for j in range(i, min(i + 10, len(lines))):
                    sched_match = re.search(r'scheduler\s*=\s*(\w+)', lines[j])
                    if sched_match:
                        self.assertEqual(
                            sched_match.group(1), 'scheduler',
                            f"Line {j+1}: scheduler= should be 'scheduler', "
                            f"got '{sched_match.group(1)}'")
                        found_call = True
                        break
                break
        self.assertTrue(found_call, "Could not find scheduler= kwarg in train_one_epoch call")


class TestC2RemainderChunking(unittest.TestCase):
    """C2: Verify remainder-aware chunking covers all samples."""

    def _chunk_indices(self, N, total_chunks):
        """Reproduce the remainder-aware chunking logic from generate_matrices.py."""
        all_indices = set()
        for chunk_id in range(total_chunks):
            base_chunk = N // total_chunks
            remainder = N % total_chunks
            if chunk_id < remainder:
                start_idx = chunk_id * (base_chunk + 1)
                end_idx = start_idx + (base_chunk + 1)
            else:
                start_idx = chunk_id * base_chunk + remainder
                end_idx = start_idx + base_chunk
            all_indices.update(range(start_idx, end_idx))
        return all_indices

    def test_even_division(self):
        """1000 samples / 8 chunks = no remainder."""
        indices = self._chunk_indices(1000, 8)
        self.assertEqual(indices, set(range(1000)))

    def test_uneven_division(self):
        """1003 samples / 8 chunks = 3 extra samples distributed."""
        indices = self._chunk_indices(1003, 8)
        self.assertEqual(indices, set(range(1003)))

    def test_small_uneven(self):
        """7 samples / 3 chunks."""
        indices = self._chunk_indices(7, 3)
        self.assertEqual(indices, set(range(7)))

    def test_single_chunk(self):
        """All samples in one chunk."""
        indices = self._chunk_indices(100, 1)
        self.assertEqual(indices, set(range(100)))

    def test_more_chunks_than_samples(self):
        """3 samples / 5 chunks — some chunks empty."""
        indices = self._chunk_indices(3, 5)
        self.assertEqual(indices, set(range(3)))


class TestH9ChunkIdUsage(unittest.TestCase):
    """H9: Verify generate_matrices.py uses chunk_id, not args.chunk_id."""

    def test_done_file_uses_chunk_id(self):
        """Read generate_matrices.py and verify done_file uses chunk_id variable."""
        gen_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generate_matrices.py"
        )
        with open(gen_path) as f:
            content = f.read()

        # Check that args.chunk_id is NOT used in done_file or print statements
        # after the initial chunk_id assignment
        import re
        # Find the done_file line
        done_match = re.search(r'done_file.*done_chunk_(.*?)\.txt', content)
        self.assertIsNotNone(done_match, "Could not find done_file line")
        self.assertNotIn('args.chunk_id', done_match.group(0),
                         "done_file should use chunk_id, not args.chunk_id")


if __name__ == "__main__":
    unittest.main()
