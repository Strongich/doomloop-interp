#!/usr/bin/env python3
"""CPU regression checks: uv run python scripts/test_vllm_steering.py."""
import unittest
import tempfile
from pathlib import Path

from branch_continue_vllm import read_journal, request_seed

import numpy as np
import torch

from reasoning_attention.serving.vllm_steering import boundary_positions, edit_decoder_output


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.boundaries = np.zeros(12, dtype=bool)
        self.boundaries[2] = True
        # Prompt includes a boundary and a close; neither affects generated scope.
        self.tokens = np.array([2, 9, 1, 2, 1, 2, 9, 2, 1])

    def sites(self, start, count, thinking=True):
        return boundary_positions(self.tokens, 3, start, count, self.boundaries, 9, thinking)

    def test_prompt_is_never_steered(self):
        self.assertEqual(self.sites(0, 3), [])

    def test_decode_and_close_scope(self):
        self.assertEqual([self.sites(i, 1) for i in range(3, 9)], [[3], [], [5], [], [], []])

    def test_recomputed_and_chunked_tokens_match_decode(self):
        self.assertEqual(self.sites(0, 9), [3, 5])
        self.assertEqual(self.sites(0, 4) + self.sites(4, 5), [3, 5])

    def test_exit_never_steers(self):
        self.assertEqual(self.sites(0, 9, thinking=False), [])

    def test_request_order_does_not_enter_site_selection(self):
        other = np.array([1, 1, 1, 9, 2, 2])
        cases = [self.tokens, other]
        def evaluate(rows):
            return [boundary_positions(t, 3, 3, 3, self.boundaries, 9) for t in rows]
        self.assertEqual(evaluate(cases), list(reversed(evaluate(list(reversed(cases))))))
        self.assertEqual(evaluate(cases), [[3, 5], []])

    def test_exact_residual_edit_and_untouched_rows(self):
        torch.manual_seed(2)
        for dtype in (torch.float32, torch.bfloat16):
            hidden = torch.randn(5, 64).to(dtype)
            residual = torch.randn(5, 64).to(dtype)
            unit = torch.randn(64)
            unit /= unit.norm()
            indices = torch.tensor([1, 3])
            h = hidden[indices] + residual[indices]
            expected = h + (0.25 * h.float().norm(dim=-1, keepdim=True) * unit).to(dtype)
            actual_h, actual_r = edit_decoder_output(hidden, residual, indices, unit, 0.25)
            self.assertTrue(torch.equal((actual_h + actual_r)[indices], expected))
            self.assertTrue(torch.equal(actual_h[[0, 2, 4]], hidden[[0, 2, 4]]))
            self.assertTrue(torch.equal(actual_r[[0, 2, 4]], residual[[0, 2, 4]]))
            self.assertTrue(torch.equal(actual_r[indices], torch.zeros_like(h)))

    def test_zero_alpha_preserves_exact_tensor_pair(self):
        hidden, residual = torch.randn(2, 4), torch.randn(2, 4)
        out = edit_decoder_output(hidden, residual, torch.tensor([1]), torch.ones(4), 0.)
        self.assertIs(out[0], hidden)
        self.assertIs(out[1], residual)


class JournalTests(unittest.TestCase):
    def test_resume_recovers_only_incomplete_last_write(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "journal.jsonl"
            p.write_bytes(b'{"id": 1}\n{"id":')
            self.assertEqual(read_journal(p), [{"id": 1}])
            self.assertEqual(p.read_bytes(), b'{"id": 1}\n')
            p.write_bytes(b'{"id": 1}\nBROKEN\n')
            with self.assertRaises(ValueError):
                read_journal(p)

    def test_resume_accepts_complete_unterminated_record(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "journal.jsonl"
            p.write_bytes(b'{"id": 1}')
            self.assertEqual(read_journal(p), [{"id": 1}])
            self.assertTrue(p.read_bytes().endswith(b"\n"))

    def test_seed_does_not_depend_on_batch_position(self):
        qs = ["gsm8k:1", "gsm8k:2"]
        self.assertEqual({q: request_seed(q, 0) for q in qs},
                         {q: request_seed(q, 0) for q in reversed(qs)})
        self.assertNotEqual(request_seed(qs[0], 0), request_seed(qs[0], 1))


if __name__ == '__main__':
    unittest.main()
