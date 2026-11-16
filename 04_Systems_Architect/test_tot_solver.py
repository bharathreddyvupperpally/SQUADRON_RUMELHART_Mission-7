"""Unit tests for Station A (run: python test_tot_solver.py)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_domain import NavigationProblem
from tot_solver import TreeOfThoughtsSolver, ThoughtNode, compare_bfs_dfs


class TestToT(unittest.TestCase):
    def setUp(self):
        self.p = NavigationProblem(2, 61, 6)

    def test_budget_is_hard_cap(self):
        for b in (0, 1, 3, 7, 25):
            for m in ("bfs_beam_search", "dfs_backtrack_search"):
                r = getattr(TreeOfThoughtsSolver(self.p, 3, b), m)()
                self.assertLessEqual(r.expansions, b)

    def test_zero_budget_returns_root(self):
        r = TreeOfThoughtsSolver(self.p, 3, 0).bfs_beam_search()
        self.assertFalse(r.success)
        self.assertEqual(r.expansions, 0)

    def test_bfs_solves_with_modest_budget(self):
        r = TreeOfThoughtsSolver(self.p, 3, 20).bfs_beam_search()
        self.assertTrue(r.success)
        self.assertEqual(r.goal_node.state[0], 61)

    def test_dfs_needs_larger_budget_than_bfs(self):
        bfs = TreeOfThoughtsSolver(self.p, 3, 20).bfs_beam_search()
        dfs = TreeOfThoughtsSolver(self.p, 3, 20).dfs_backtrack_search()
        self.assertTrue(bfs.success)
        self.assertFalse(dfs.success)
        self.assertTrue(TreeOfThoughtsSolver(self.p, 3, 80).dfs_backtrack_search().success)

    def test_beam_width_limits_frontier(self):
        s = TreeOfThoughtsSolver(self.p, 2, 100)
        r = s.bfs_beam_search()
        per_depth = {}
        for tag in r.visit_order:
            d = int(tag.split("@d")[1]); per_depth[d] = per_depth.get(d, 0) + 1
        self.assertTrue(all(c <= 2 for c in per_depth.values()))

    def test_solution_path_is_valid_arithmetic(self):
        r = TreeOfThoughtsSolver(self.p, 3, 40).bfs_beam_search()
        for t in r.solution:
            lhs, rhs = t.split(" = ")
            self.assertEqual(eval(lhs), int(rhs))  # noqa: S307 (trusted test strings)

    def test_prune_hook_severs_branches(self):
        s = TreeOfThoughtsSolver(self.p, 3, 40, prune_fn=lambda n: n.state[0] % 2 == 1)
        r = s.bfs_beam_search()
        self.assertGreater(r.nodes_pruned, 0)

    def test_compare_table_shape(self):
        rows = compare_bfs_dfs(self.p, (10, 20))
        self.assertEqual(len(rows), 4)

    def test_node_path(self):
        root = ThoughtNode((1, 0)); c = ThoughtNode((2, 1), "1 + 1 = 2", root)
        self.assertEqual(c.depth, 1); self.assertEqual(c.thought_texts(), ["1 + 1 = 2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
