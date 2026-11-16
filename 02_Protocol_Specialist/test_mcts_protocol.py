"""Unit tests for Station B (run: python test_mcts_protocol.py)."""
import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_domain import NavigationProblem
from mcts_protocol import (MCTSEngine, MCTSNode, puct_score, select_best_child,
                           puct_selection_trace)


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.p = NavigationProblem(2, 61, 6)

    def test_puct_exact_formula(self):
        got = puct_score(0.5, 0.6, 13, 10, 1.0)
        self.assertAlmostEqual(got, 0.5 + 1.0 * 0.6 * math.sqrt(13) / 11)

    def test_zero_visit_clean(self):
        parent = MCTSNode((0, 0))
        for name, pr in (("a", 0.2), ("b", 0.7), ("c", 0.1)):
            parent.children.append(MCTSNode((0, 1), parent, name, pr))
        parent.expanded = True
        self.assertEqual(parent.children[0].Q, 0.0)
        self.assertEqual(select_best_child(parent, 1.4).action, "b")  # tie -> highest prior

    def test_unvisited_child_gets_exploration_bonus(self):
        parent = MCTSNode((0, 0)); parent.N = 20
        seen = MCTSNode((0, 1), parent, "seen", 0.5); seen.N, seen.W = 19, 9.5
        new = MCTSNode((0, 1), parent, "new", 0.5)
        parent.children = [seen, new]; parent.expanded = True
        self.assertEqual(select_best_child(parent, 2.0).action, "new")

    def test_c_zero_is_pure_exploitation(self):
        _, rows = puct_selection_trace((0.0,))
        self.assertTrue(rows[0][2].startswith("A"))

    def test_priors_normalised(self):
        eng = MCTSEngine(self.p)
        th = self.p.propose(self.p.initial_state())
        self.assertAlmostEqual(sum(eng.priors(th)), 1.0)

    def test_backprop_updates_whole_path(self):
        a = MCTSNode((0, 0)); b = MCTSNode((0, 1), a); c = MCTSNode((0, 2), b)
        MCTSEngine.backpropagate(c, 0.8)
        for n in (a, b, c):
            self.assertEqual(n.N, 1); self.assertAlmostEqual(n.W, 0.8)

    def test_root_visits_equal_simulations(self):
        out = MCTSEngine(self.p, n_simulations=60, seed=1).search(self.p.initial_state())
        self.assertEqual(out.root_visits, 60)
        self.assertEqual(sum(out.visits.values()), 60)  # every simulation passes through a root child

    def test_solves_and_is_deterministic(self):
        r1 = MCTSEngine(self.p, 1.4, 150, seed=3).solve()
        r2 = MCTSEngine(self.p, 1.4, 150, seed=3).solve()
        self.assertEqual(r1, r2)
        self.assertTrue(r1[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
