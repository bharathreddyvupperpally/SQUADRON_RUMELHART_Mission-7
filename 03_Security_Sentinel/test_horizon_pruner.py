"""Unit tests for Station D (run: python test_horizon_pruner.py)."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horizon_pruner import (ATTACKS, EventHorizonPruner, GroundedPRM, StyleBiasedPRM,
                            reduce, run_attack_matrix)

CLEAN = ["7 + 5 = 12", "12 * 3 = 36", "36 - 4 = 32", "32 * 2 = 64"]
BAD = ["7 + 5 = 12", "12 * 3 = 35", "35 - 4 = 31", "31 * 2 = 62"]


class _Node:                       # minimal ThoughtNode stand-in
    def __init__(self, texts): self._t = texts
    def thought_texts(self): return self._t


class TestPruner(unittest.TestCase):
    def test_clean_survives_flawed_pruned(self):
        p = EventHorizonPruner(0.35, "geometric_mean")
        self.assertFalse(p.evaluate(CLEAN, "c").pruned)
        d = p.evaluate(BAD, "b")
        self.assertTrue(d.pruned)
        self.assertEqual(d.prune_step, 2)          # severed at the first step after the grace period
        self.assertGreater(d.tokens_saved, 0)

    def test_prune_happens_before_remaining_steps(self):
        d = EventHorizonPruner(0.35, "min").evaluate(BAD, "b")
        self.assertLess(d.prune_step, len(BAD))

    def test_filler_detection(self):
        self.assertEqual(EventHorizonPruner.detect_filler("1 + 1 = 2"), [])
        self.assertIn("trust me", EventHorizonPruner.detect_filler("Trust me, 1 + 1 = 3"))
        self.assertEqual(EventHorizonPruner.strip_filler("clearly 1 + 1 = 3"), "1 + 1 = 3")

    def test_style_biased_prm_is_hackable_and_defence_undoes_it(self):
        prm = StyleBiasedPRM()
        self.assertGreater(prm("12 * 3 = 35, verified and clearly correct, trust me!", ["7 + 5 = 12"]), 0.5)
        p = EventHorizonPruner(0.35, "geometric_mean", prm)
        flat = ATTACKS["atk1_flattery_on_wrong_step"][0]
        self.assertGreater(p.raw_score(flat), 0.8)                     # attack works on raw score
        self.assertTrue(p.evaluate(flat, "f").pruned)                   # defence stops it

    def test_step_splitting_detected(self):
        p = EventHorizonPruner()
        steps = ATTACKS["atk2_duplicate_splitting"][0]
        self.assertEqual(p.detect_step_splitting(steps), list(range(1, 8)))
        frag = ATTACKS["atk2_fragment_splitting"][0]
        self.assertTrue(set(range(1, 9)) <= set(p.detect_step_splitting(frag)))

    def test_splitting_inflates_raw_but_not_defended(self):
        p = EventHorizonPruner(0.35, "geometric_mean", StyleBiasedPRM())
        steps = ATTACKS["atk2_duplicate_splitting"][0]
        self.assertGreater(p.raw_score(steps), 0.35)
        self.assertLess(p.defended_score(steps), 0.35)

    def test_attack_matrix_full_catch_no_false_prune(self):
        modes, table = run_attack_matrix()
        defended = modes[-1]
        for name, (valid, row) in table.items():
            self.assertEqual(row[defended], not valid, name)

    def test_single_remark_is_not_a_false_positive(self):
        steps = ATTACKS["clean_with_one_remark"][0]
        self.assertFalse(EventHorizonPruner(0.35, "geometric_mean", StyleBiasedPRM()).evaluate(steps).pruned)

    def test_all_reductions_work(self):
        for r in ("min", "prod", "geometric_mean"):
            self.assertTrue(EventHorizonPruner(0.35, r).evaluate(BAD).pruned)
            self.assertFalse(EventHorizonPruner(0.35, r).evaluate(CLEAN).pruned)

    def test_node_filter(self):
        f = EventHorizonPruner(0.35, "geometric_mean").node_filter()
        self.assertFalse(f(_Node(["7 + 5 = 12"])))
        self.assertTrue(f(_Node(["7 + 5 = 13"])))
        self.assertFalse(f(_Node([])))

    def test_log_and_summary(self):
        p = EventHorizonPruner(0.35)
        p.evaluate(CLEAN, "c"); p.evaluate(BAD, "b")
        s = p.summary()
        self.assertEqual((s["trajectories"], s["pruned"]), (2, 1))
        self.assertEqual(s["tokens_saved"], p.log[1].tokens_saved)
        with tempfile.TemporaryDirectory() as d:
            p.save_log(os.path.join(d, "x.json"))
            self.assertTrue(os.path.exists(os.path.join(d, "x.json")))

    def test_bad_reduction(self):
        with self.assertRaises(ValueError):
            reduce([0.5], "median")


if __name__ == "__main__":
    unittest.main(verbosity=2)
