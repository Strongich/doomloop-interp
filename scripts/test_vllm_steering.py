#!/usr/bin/env python3
"""CPU regression checks: uv run python scripts/test_vllm_steering.py."""
import random
import unittest
import tempfile
from pathlib import Path

from branch_continue_vllm import read_journal, request_seed
from reasoning_policy_vllm import expected_sites, parse_policy

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

class DelayTests(unittest.TestCase):
    """start_delay: g = p - prompt_length + 1 must be >= delay.

    Prompt length 3, so generated indices g=1..6 sit at absolute p=3..8.
    Boundaries (token id 2) are at p=3,5,7; p=7 is unreachable because the close
    token (id 9) lands at p=6 and ends the thinking scope.
    """

    def setUp(self):
        self.boundaries = np.zeros(12, dtype=bool)
        self.boundaries[2] = True
        #        p:  0  1  2 | 3  4  5  6  7  8
        #        g:            1  2  3  4  5  6
        self.tokens = np.array([2, 9, 1, 2, 1, 2, 9, 2, 1])

    def sites(self, delay, start=0, count=9):
        return boundary_positions(self.tokens, 3, start, count, self.boundaries, 9,
                                  True, delay)

    def test_delay_zero_and_one_are_the_same_policy(self):
        # g >= 0 and g >= 1 both admit every generated boundary.
        self.assertEqual(self.sites(0), [3, 5])
        self.assertEqual(self.sites(1), [3, 5])

    def test_boundary_immediately_before_at_and_after_threshold(self):
        # Each site is admitted exactly while delay <= its own g.
        # p=3 -> g=1, p=5 -> g=3.
        self.assertEqual(self.sites(1), [3, 5])  # both at their thresholds
        self.assertEqual(self.sites(2), [5])     # one past p=3's g, still under p=5's
        self.assertEqual(self.sites(3), [5])     # exactly at p=5's g
        self.assertEqual(self.sites(4), [])      # one past it

    def test_delay_never_reaches_past_the_close(self):
        # p=7 is a boundary but follows the close at p=6, so no delay revives it.
        self.assertEqual(self.sites(5), [])
        self.assertEqual(self.sites(99), [])

    def test_delay_counts_generated_tokens_not_absolute_positions(self):
        # A longer prompt shifts the admitted site by exactly the prompt length.
        long_prompt = np.array([1] * 10 + [2, 1, 2])
        got = boundary_positions(long_prompt, 10, 0, 13, self.boundaries, 9, True, 3)
        self.assertEqual(got, [12])  # p=12 -> g=3
        self.assertEqual(
            boundary_positions(long_prompt, 10, 0, 13, self.boundaries, 9, True, 4), []
        )

    def test_chunked_and_recomputed_prefill_agree_with_decode(self):
        for delay in (0, 2, 3, 4):
            whole = self.sites(delay)
            split = self.sites(delay, 0, 4) + self.sites(delay, 4, 5)
            steps = [x for i in range(9) for x in self.sites(delay, i, 1)]
            self.assertEqual(whole, split, f"chunked mismatch at delay={delay}")
            self.assertEqual(whole, steps, f"stepwise mismatch at delay={delay}")

    def test_exit_arm_ignores_delay_entirely(self):
        for delay in (0, 3, 99):
            self.assertEqual(
                boundary_positions(self.tokens, 3, 0, 9, self.boundaries, 9, False, delay), []
            )


class DriverAuditAgreementTests(unittest.TestCase):
    """The driver's audit must agree with the worker on randomized inputs.

    `reasoning_policy_vllm.expected_sites` reconstructs eligible sites from
    output token IDs to check the worker. That check is only meaningful if the
    two implementations agree wherever they should -- otherwise the audit either
    never fires or fires constantly. Note the two modelling facts the test
    encodes: a finished request never consumes its FINAL sampled token, and a
    forward pass may be split anywhere by chunked prefill.
    """

    def test_randomized_agreement_including_chunk_splits(self):
        rng = random.Random(0)
        vocab, close, barr = 8, 9, np.zeros(10, dtype=bool)
        barr[2] = True
        nontrivial = 0
        for _ in range(3000):
            plen = rng.randint(1, 6)
            ids = [rng.randrange(vocab) for _ in range(rng.randint(1, 14))]
            delay = rng.randint(0, 8)
            tokens = np.array([rng.randrange(vocab) for _ in range(plen)] + ids)
            consumed = len(tokens) - 1
            cut = rng.randint(0, consumed)
            worker = (
                boundary_positions(tokens, plen, 0, cut, barr, close, True, delay)
                + boundary_positions(tokens, plen, cut, consumed - cut, barr, close, True, delay)
            )
            driver = expected_sites(ids, plen, {2}, close, delay)
            nontrivial += bool(driver)
            self.assertEqual(worker, driver, f"plen={plen} ids={ids} delay={delay} cut={cut}")
        self.assertGreater(nontrivial, 500, "test data degenerate: too few real sites")


class SteeringTensorTests(unittest.TestCase):
    def test_zero_alpha_preserves_exact_tensor_pair(self):
        hidden, residual = torch.randn(2, 4), torch.randn(2, 4)
        out = edit_decoder_output(hidden, residual, torch.tensor([1]), torch.ones(4), 0.)
        self.assertIs(out[0], hidden)
        self.assertIs(out[1], residual)


class PolicyParsingTests(unittest.TestCase):
    def test_policy_round_trip(self):
        p = parse_policy("N@a0.25d512")
        self.assertEqual(
            (p["direction"], p["alpha"], p["delay"], p["brevity"]), ("N", 0.25, 512, None)
        )
        self.assertEqual(parse_policy("base")["direction"], None)
        self.assertEqual(parse_policy("brevityB")["brevity"], "B")

    def test_unknown_policies_are_rejected(self):
        for bad in ("N@a1.0", "Z@a1.0d0", "brevityZ", "N@a1.0d-1", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_policy(bad)


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
