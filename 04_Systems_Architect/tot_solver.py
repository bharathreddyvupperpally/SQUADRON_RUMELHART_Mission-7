"""tot_solver.py - Station A (Systems Architect)
Tree-of-Thoughts search topologies under a strict compute budget.

* ThoughtNode              - node of the thought graph (state, thought, score, links)
* TreeOfThoughtsSolver     - bfs_beam_search()  : breadth-first, beam width B per depth
                             dfs_backtrack_search(): depth-first with backtracking
* Compute budget           - `max_expansions` is a HARD cap: one expansion = one call
                             to the proposer ("LLM call") on one node.  The solver never
                             exceeds it (checked by assertions and unit tests).

Problem interface (duck typed, see nav_domain.NavigationProblem):
    initial_state(), propose(state)->[obj with .text/.state], value(state)->float,
    is_goal(state), is_terminal(state)

Run `python tot_solver.py` for the BFS-vs-DFS comparison.  Standard library only.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_domain import NavigationProblem  # noqa: E402


class BudgetExhausted(Exception):
    """Raised internally when max_expansions has been used up."""


class ThoughtNode:
    """One node in the thought graph."""

    _ids = itertools.count()

    def __init__(self, state, thought: str = "", parent: Optional["ThoughtNode"] = None,
                 score: float = 0.0):
        self.id = next(ThoughtNode._ids)
        self.state = state
        self.thought = thought
        self.parent = parent
        self.score = score
        self.children: List["ThoughtNode"] = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.expanded = False
        self.pruned = False

    def path(self) -> List["ThoughtNode"]:
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]

    def thought_texts(self) -> List[str]:
        """Thoughts from the root to this node (root has no thought)."""
        return [n.thought for n in self.path() if n.parent is not None]

    def __repr__(self) -> str:
        return f"<Node#{self.id} d={self.depth} s={self.state} score={self.score:.3f}>"


@dataclass
class SearchResult:
    method: str
    success: bool
    goal_node: Optional[ThoughtNode]
    best_node: ThoughtNode
    expansions: int
    nodes_generated: int
    nodes_pruned: int
    max_expansions: int
    backtracks: int = 0
    visit_order: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    @property
    def solution(self) -> List[str]:
        node = self.goal_node or self.best_node
        return node.thought_texts()


class TreeOfThoughtsSolver:
    def __init__(self, problem, beam_width: int = 3, max_expansions: int = 50,
                 max_depth: Optional[int] = None,
                 prune_fn: Optional[Callable[[ThoughtNode], bool]] = None,
                 min_value: float = 1e-9):
        """
        beam_width     B: nodes kept per depth (BFS) / children tried per node (DFS)
        max_expansions hard compute budget (proposer calls)
        prune_fn       optional hook: prune_fn(child_node) -> True severs the branch
                       (the Event Horizon Pruner plugs in here)
        min_value      children whose heuristic value is below this are never kept
                       (dead states beyond the event horizon)
        """
        assert beam_width >= 1 and max_expansions >= 0
        self.problem = problem
        self.B = beam_width
        self.max_expansions = max_expansions
        self.max_depth = max_depth if max_depth is not None else getattr(problem, "max_depth", 6)
        self.prune_fn = prune_fn
        self.min_value = min_value
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self):
        self.expansions = 0
        self.generated = 0
        self.pruned = 0
        self.backtracks = 0
        self.visit_order: List[str] = []
        self.trace: List[str] = []
        self.best: Optional[ThoughtNode] = None

    def _make_root(self) -> ThoughtNode:
        s = self.problem.initial_state()
        root = ThoughtNode(s, "", None, self.problem.value(s))
        self.best = root
        return root

    def expand(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Spend ONE unit of budget to generate the children of `node`."""
        if self.expansions >= self.max_expansions:
            raise BudgetExhausted
        self.expansions += 1
        assert self.expansions <= self.max_expansions
        node.expanded = True
        kept = []
        for th in self.problem.propose(node.state):
            child = ThoughtNode(th.state, th.text, node, self.problem.value(th.state))
            self.generated += 1
            node.children.append(child)
            if child.score < self.min_value and not self.problem.is_goal(child.state):
                child.pruned = True
                self.pruned += 1
                self.trace.append(f"  x dead  {child.thought}  (event horizon)")
                continue
            if self.prune_fn is not None and self.prune_fn(child):
                child.pruned = True
                self.pruned += 1
                self.trace.append(f"  x prune {child.thought}  (pruner)")
                continue
            if child.score > self.best.score:
                self.best = child
            kept.append(child)
        return kept

    # ------------------------------------------------------------------
    def bfs_beam_search(self) -> SearchResult:
        """Breadth-first beam search: keep the top-B nodes at every depth."""
        self._reset()
        root = self._make_root()
        frontier = [root]
        goal = None
        try:
            for depth in range(self.max_depth):
                candidates: List[ThoughtNode] = []
                for node in frontier:
                    if self.problem.is_terminal(node.state):
                        continue
                    self.visit_order.append(f"{node.state[0]}@d{node.depth}")
                    candidates.extend(self.expand(node))
                    hit = next((c for c in candidates if self.problem.is_goal(c.state)), None)
                    if hit:
                        goal = hit
                        break
                if goal:
                    break
                candidates.sort(key=lambda n: (-n.score, n.id))
                frontier = candidates[: self.B]
                self.trace.append(f"depth {depth + 1}: {len(candidates)} candidates -> beam "
                                  f"{[n.state[0] for n in frontier]}")
                if not frontier:
                    break
        except BudgetExhausted:
            self.trace.append("BUDGET EXHAUSTED")
        return self._result("ToT-BFS", goal)

    def dfs_backtrack_search(self) -> SearchResult:
        """Depth-first search; tries the best B children, backtracks when stuck."""
        self._reset()
        root = self._make_root()
        goal = None

        def visit(node: ThoughtNode) -> Optional[ThoughtNode]:
            if self.problem.is_goal(node.state):
                return node
            if self.problem.is_terminal(node.state):
                self.backtracks += 1
                self.trace.append(f"{'  ' * node.depth}<- backtrack from {node.state[0]}@d{node.depth}")
                return None
            self.visit_order.append(f"{node.state[0]}@d{node.depth}")
            children = self.expand(node)
            children.sort(key=lambda n: (-n.score, n.id))
            for child in children[: self.B]:
                found = visit(child)
                if found:
                    return found
            self.backtracks += 1
            return None

        try:
            goal = visit(root)
        except BudgetExhausted:
            self.trace.append("BUDGET EXHAUSTED")
        return self._result("ToT-DFS", goal)

    def _result(self, method: str, goal: Optional[ThoughtNode]) -> SearchResult:
        return SearchResult(method, goal is not None, goal, self.best, self.expansions,
                            self.generated, self.pruned, self.max_expansions,
                            self.backtracks, list(self.visit_order), list(self.trace))


# ----------------------------------------------------------------------
def compare_bfs_dfs(problem, budgets=(10, 20, 40, 60, 80), beam_width: int = 3):
    rows = []
    for b in budgets:
        for runner in ("bfs_beam_search", "dfs_backtrack_search"):
            solver = TreeOfThoughtsSolver(problem, beam_width, b)
            r = getattr(solver, runner)()
            assert r.expansions <= b
            rows.append(dict(budget=b, method=r.method, success=r.success,
                             expansions=r.expansions, generated=r.nodes_generated,
                             pruned=r.nodes_pruned, backtracks=r.backtracks,
                             depth=(r.goal_node or r.best_node).depth,
                             best_value=round((r.goal_node or r.best_node).score, 3)))
    return rows


def format_table(rows) -> str:
    head = f"{'budget':>6} {'method':<8} {'solved':<6} {'expansions':>10} {'generated':>9} " \
           f"{'backtracks':>10} {'best depth':>10} {'best value':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['budget']:>6} {r['method']:<8} {str(r['success']):<6} {r['expansions']:>10} "
                     f"{r['generated']:>9} {r['backtracks']:>10} {r['depth']:>10} {r['best_value']:>10}")
    return "\n".join(lines)


def beam_width_sweep(problem, widths=(1, 2, 3, 5), budget: int = 40):
    out = []
    for w in widths:
        r = TreeOfThoughtsSolver(problem, w, budget).bfs_beam_search()
        out.append((w, r.success, r.expansions))
    return out


def main():
    problem = NavigationProblem(start=2, target=61, max_depth=6)
    print("SEC-07-SGR  Station A  Tree-of-Thoughts  (start=2, target=61, "
          f"optimal depth={problem.shortest_solution()})\n")
    rows = compare_bfs_dfs(problem)
    print(format_table(rows))

    print("\nBeam-width sweep (BFS, budget=40): width, solved, expansions")
    for w, ok, e in beam_width_sweep(problem):
        print(f"  B={w}: solved={ok}, expansions={e}")

    bfs = TreeOfThoughtsSolver(problem, 3, 80).bfs_beam_search()
    dfs = TreeOfThoughtsSolver(problem, 3, 80).dfs_backtrack_search()
    print("\nTraversal order (budget 80, B=3)")
    print(f"  BFS expanded ({bfs.expansions}):", " ".join(bfs.visit_order))
    print(f"  DFS expanded ({dfs.expansions}, first 20 shown):", " ".join(dfs.visit_order[:20]), "...")
    print("  BFS solution:", " | ".join(bfs.solution))
    print("  DFS solution:", " | ".join(dfs.solution))
    print("\nDFS trace (first 16 lines):")
    print("\n".join(dfs.trace[:16]))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tot_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
