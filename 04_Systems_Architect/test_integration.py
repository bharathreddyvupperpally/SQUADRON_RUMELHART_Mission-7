"""Integration tests (run: python test_integration.py) - needs all four station folders side by side."""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mission as R  # noqa: E402  (adds the sibling station folders to sys.path)
from nav_domain import NavigationProblem  # noqa: E402
from tot_solver import TreeOfThoughtsSolver  # noqa: E402
from horizon_pruner import EventHorizonPruner  # noqa: E402


class TestIntegration(unittest.TestCase):
    def test_honest_proposer_pruner_changes_nothing(self):
        for seed in range(5):
            a = R.tot_trial(0.0, seed, False)
            b = R.tot_trial(0.0, seed, True)
            self.assertEqual((a[0], a[1]), (True, True))
            self.assertEqual((b[0], b[1]), (True, True))

    def test_false_claim_is_caught_and_repaired_by_pruner(self):
        claimed, ok, _ = R.tot_trial(0.2, 30, False)
        self.assertTrue(claimed)
        self.assertFalse(ok)                      # reached 61 through a hallucinated step
        claimed, ok, _ = R.tot_trial(0.2, 30, True)
        self.assertTrue(claimed and ok)           # pruner forces a genuine route

    def test_verified_success_improves_with_pruner(self):
        seeds = range(20)
        plain = sum(R.tot_trial(0.2, s, False)[1] for s in seeds)
        guarded = sum(R.tot_trial(0.2, s, True)[1] for s in seeds)
        self.assertGreater(guarded, plain)

    def test_prm_guided_mcts_beats_plain_on_verified(self):
        seeds = range(12)
        plain = sum(R.mcts_trial(0.2, s, False)[1] for s in seeds)
        guided = sum(R.mcts_trial(0.2, s, True)[1] for s in seeds)
        self.assertGreater(guided, plain)

    def test_budget_never_exceeded_with_pruner(self):
        p = NavigationProblem(2, 61, 6, hallucination_rate=0.3, seed=4)
        pf = EventHorizonPruner(0.35).node_filter()
        r = TreeOfThoughtsSolver(p, 3, 15, prune_fn=pf).bfs_beam_search()
        self.assertLessEqual(r.expansions, 15)

    def test_run_mission_is_reproducible(self):
        outs = []
        for _ in range(2):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.main()
            outs.append(buf.getvalue())
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
