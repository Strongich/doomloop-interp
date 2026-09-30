#!/usr/bin/env python3
"""Grader regression checks (D66/D69): uv run python scripts/test_grading.py."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_attention.grading import canonicalize_gold, grade  # noqa: E402


class GoldCanonicalization(unittest.TestCase):
    def test_shorthand_is_braced(self):
        for raw, want in [("\\frac83", "\\frac{8}{3}"), ("\\frac 1{72}", "\\frac{1}{72}"),
                          ("\\frac{1}2", "\\frac{1}{2}"), ("\\sqrt2", "\\sqrt{2}"),
                          ("11\\sqrt2", "11\\sqrt{2}"), ("\\frac\\pi2", "\\frac{\\pi}{2}"),
                          ("\\tfrac{8\\pi}5", "\\tfrac{8\\pi}{5}"), ("\\dfrac38", "\\dfrac{3}{8}")]:
            self.assertEqual(canonicalize_gold(raw), want)

    def test_choice_list_unwrapped(self):
        self.assertEqual(canonicalize_gold("\\text{C,E}"), "C,E")
        self.assertEqual(canonicalize_gold("\\text{B}"), "B")
        self.assertEqual(canonicalize_gold("\\text{Evelyn}"), "\\text{Evelyn}")  # a word, kept

    def test_canonical_golds_unchanged(self):
        for g in ["\\frac{3\\sqrt{13}}{2}", "\\sqrt[3]{2}", "\\dfrac{8}{3}", "45^\\circ, 135^\\circ",
                  "\\left(\\frac{1}{2},3\\right)", "6-5i", "\\pi", "p - q", "18"]:
            self.assertEqual(canonicalize_gold(g), g)

    def test_cohort_golds_only_shorthand_changes(self):
        """On every cohort, canonicalization may only touch golds that contain shorthand
        (a macro followed by a non-brace argument) or a \\text{} choice list."""
        import re
        pat = re.compile(r"\\(?:d|t)?frac\s*[^{\s]|\\(?:d|t)?frac\{[^{}]*\}\s*[^{\s]|\\sqrt\s*[^{\[\s]|^\\text\{[A-Za-z](,\s*[A-Za-z])*\}$")
        for f in ["data/policy/math500.jsonl", "data/xfer8b/confirm_mathtest400.jsonl",
                  "data/xfer8b/dev_math250.jsonl", "data/xfer8b/fit_math400.jsonl", "data/policy/dev400.jsonl"]:
            for line in open(ROOT / f):
                r = json.loads(line)
                g = str(r.get("gold", r.get("answer")))
                if canonicalize_gold(g) != g:
                    self.assertTrue(pat.search(g), f"{f}: unexpected change of {g!r}")


class Grading(unittest.TestCase):
    def test_fixed_false_negatives(self):
        for gold, resp in [("\\frac83", "\\boxed{\\dfrac{8}{3}}"), ("\\frac 1{72}", "\\boxed{\\dfrac{1}{72}}"),
                           ("\\text{C,E}", "\\boxed{C,E}"), ("11\\sqrt2", "\\boxed{11\\sqrt{2}}"),
                           ("\\frac9{19}", "\\boxed{\\frac{9}{19}}")]:
            self.assertTrue(grade(resp, gold).is_correct, (gold, resp))

    def test_wrong_stays_wrong(self):
        for gold, resp in [("\\frac83", "\\boxed{3}"), ("\\frac83", "\\boxed{\\frac{3}{8}}"),
                           ("\\text{C,E}", "\\boxed{C}"), ("11\\sqrt2", "\\boxed{11}"),
                           ("45^\\circ, 135^\\circ", "\\boxed{135}")]:
            self.assertFalse(grade(resp, gold).is_correct, (gold, resp))

    def test_three_outcomes_kept(self):
        self.assertEqual(grade("no box here", "\\frac83").has_answer, False)
        g = grade("\\boxed{7}", "\\frac83")
        self.assertEqual((g.has_answer, g.is_correct), (True, False))


if __name__ == "__main__":
    unittest.main()
