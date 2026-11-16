"""mcts_protocol.py - Station B (Communications & Protocol Officer)
4-phase Monte Carlo Tree Search with PUCT selection.

    1. SELECTION       descend by argmax PUCT(s,a)
                       PUCT = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
    2. EXPANSION       create children; priors P(a|s) come from the "LLM policy"
                       (softmax over the heuristic value of each child by default)
    3. SIMULATION      seeded random rollout to a terminal state -> reward in [0, 1]
    4. BACKPROPAGATION add the reward to W and +1 to N on every node of the path

Zero-visit handling: Q(s,a) = W/N is defined as `fpu` (first-play value, default 0)
when N(s,a) = 0, so an unvisited child is scored by its prior-driven exploration term
alone.  When N(s) = 0 the exploration term is exactly 0 for every child; ties are then
broken by the prior P(a|s) and finally by action order, so selection is deterministic.

Run `python mcts_protocol.py` for the PUCT traces and the c_puct sweep.
Standard library only.
"""
from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_domain import NavigationProblem  # noqa: E402


# ----------------------------------------------------------------------
def puct_score(q: float, prior: float, parent_visits: int, child_visits: int,
               c_puct: float) -> float:
    """Exact PUCT: Q + c * P * sqrt(N(s)) / (1 + N(s,a))."""
    return q + c_puct * prior * math.sqrt(parent_visits) / (1 + child_visits)


class MCTSNode:
    def __init__(self, state, parent: Optional["MCTSNode"] = None, action: str = "",
                 prior: float = 1.0, fpu: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.fpu = fpu
        self.N = 0            # visit count
        self.W = 0.0          # total value
        self.children: List["MCTSNode"] = []
        self.expanded = False

    @property
    def Q(self) -> float:
        """Mean value; fpu for an unvisited node (clean zero-visit init)."""
        return self.W / self.N if self.N > 0 else self.fpu

    def puct(self, c_puct: float) -> float:
        assert self.parent is not None
        return puct_score(self.Q, self.prior, self.parent.N, self.N, c_puct)


def select_best_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """argmax PUCT over children; ties -> higher prior -> earlier action."""
    best, best_key = None, None
    for i, ch in enumerate(node.children):
        key = (round(ch.puct(c_puct), 12), ch.prior, -i)
        if best_key is None or key > best_key:
            best, best_key = ch, key
    return best


@dataclass
class SearchOutcome:
    best_action: str
    best_state: tuple
    visits: Dict[str, int]
    q_values: Dict[str, float]
    root_visits: int
    simulations: int
    trace: List[str] = field(default_factory=list)


class MCTSEngine:
    def __init__(self, problem, c_puct: float = 1.4, n_simulations: int = 200,
                 seed: int = 0, prior_temperature: float = 0.15, fpu: float = 0.0,
                 policy_fn: Optional[Callable] = None, log_trace: bool = False):
        self.problem = problem
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.rng = random.Random(seed)
        self.temp = prior_temperature
        self.fpu = fpu
        self.policy_fn = policy_fn
        self.log_trace = log_trace
        self.trace: List[str] = []
        self.expansions = 0

    # ---- phase 1: selection --------------------------------------------
    def select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.expanded and node.children and not self.problem.is_terminal(node.state):
            nxt = select_best_child(node, self.c_puct)
            if self.log_trace:
                self.trace.append(f"  select {nxt.action or 'root'}  N(s)={node.N} "
                                  f"N(s,a)={nxt.N} Q={nxt.Q:.3f} P={nxt.prior:.3f} "
                                  f"PUCT={nxt.puct(self.c_puct):.3f}")
            node = nxt
        return node

    # ---- phase 2: expansion --------------------------------------------
    def priors(self, thoughts) -> List[float]:
        if self.policy_fn is not None:
            p = list(self.policy_fn(thoughts))
        else:   # softmax over heuristic values = stand-in for LLM token probabilities
            vals = [self.problem.value(t.state) / self.temp for t in thoughts]
            m = max(vals)
            ex = [math.exp(v - m) for v in vals]
            p = [e / sum(ex) for e in ex]
        s = sum(p)
        assert s > 0
        return [x / s for x in p]

    def expand(self, node: MCTSNode) -> MCTSNode:
        if self.problem.is_terminal(node.state) or node.expanded:
            return node
        self.expansions += 1
        thoughts = self.problem.propose(node.state)
        for th, pr in zip(thoughts, self.priors(thoughts)):
            node.children.append(MCTSNode(th.state, node, th.text, pr, self.fpu))
        node.expanded = True
        return node

    # ---- phase 3: simulation -------------------------------------------
    def simulate(self, node: MCTSNode) -> float:
        state = node.state
        while not self.problem.is_terminal(state):
            state = self.rng.choice(self.problem.propose(state)).state
        return self.problem.value(state)

    # ---- phase 4: backpropagation --------------------------------------
    @staticmethod
    def backpropagate(node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

    # ---- driver -----------------------------------------------------------
    def search(self, state) -> SearchOutcome:
        root = MCTSNode(state, fpu=self.fpu)
        if self.problem.is_terminal(state):
            return SearchOutcome("", state, {}, {}, 0, 0)
        for i in range(self.n_simulations):
            leaf = self.select(root)
            if not self.problem.is_terminal(leaf.state):
                self.expand(leaf)
                if leaf.children:      # simulate from a freshly expanded child
                    leaf = select_best_child(leaf, self.c_puct)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        # final move: most-visited child (robust child), ties -> mean value
        best = max(root.children, key=lambda c: (c.N, c.Q))
        return SearchOutcome(best.action, best.state,
                             {c.action: c.N for c in root.children},
                             {c.action: round(c.Q, 4) for c in root.children},
                             root.N, self.n_simulations, list(self.trace))

    def solve(self, max_steps: Optional[int] = None):
        """Receding-horizon planning: search, commit to best action, repeat."""
        state = self.problem.initial_state()
        plan = []
        for _ in range(max_steps or self.problem.max_depth):
            if self.problem.is_terminal(state):
                break
            out = self.search(state)
            plan.append(out.best_action)
            state = out.best_state
        return self.problem.is_goal(state), plan, state


# ----------------------------------------------------------------------
# PUCT selection trace across c_puct values (fixed, hand-checkable example)
EXAMPLE_CHILDREN = [   # (action, prior P, visits N(s,a), total value W)
    ("A: x3 (high prior)", 0.60, 10, 5.0),
    ("B: +5 (mid prior)", 0.30, 3, 1.5),
    ("C: -3 (low prior, unvisited)", 0.10, 0, 0.0),
]


def puct_selection_trace(c_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                         children=EXAMPLE_CHILDREN):
    parent_N = sum(c[2] for c in children)
    rows = []
    for c in c_values:
        scores = []
        for name, p, n, w in children:
            q = w / n if n else 0.0
            scores.append(puct_score(q, p, parent_N, n, c))
        pick = max(range(len(children)), key=lambda i: (round(scores[i], 12), children[i][1], -i))
        rows.append((c, scores, children[pick][0]))
    return parent_N, rows


def c_puct_sweep(problem, c_values=(0.0, 0.5, 1.4, 3.0, 8.0), sims: int = 150, seeds=range(20)):
    rows = []
    for c in c_values:
        wins, entropy_sum = 0, 0.0
        for sd in seeds:
            eng = MCTSEngine(problem, c_puct=c, n_simulations=sims, seed=sd)
            ok, _, _ = eng.solve()
            wins += ok
            out = MCTSEngine(problem, c_puct=c, n_simulations=sims, seed=sd).search(problem.initial_state())
            tot = sum(out.visits.values())
            entropy_sum += -sum((v / tot) * math.log2(v / tot) for v in out.visits.values() if v)
        rows.append((c, wins / len(list(seeds)), entropy_sum / len(list(seeds))))
    return rows


def main():
    print("SEC-07-SGR  Station B  MCTS + PUCT\n")
    parent_N, rows = puct_selection_trace()
    print(f"PUCT = Q + c * P * sqrt(N(s)) / (1 + N(s,a))   example parent N(s) = {parent_N}")
    for name, p, n, w in EXAMPLE_CHILDREN:
        print(f"   {name:<30} P={p:.2f}  N={n:<2}  Q={(w / n if n else 0):.2f}")
    print(f"\n{'c_puct':>6} | {'PUCT(A)':>8} {'PUCT(B)':>8} {'PUCT(C)':>8} | selected")
    for c, sc, pick in rows:
        print(f"{c:>6} | {sc[0]:>8.3f} {sc[1]:>8.3f} {sc[2]:>8.3f} | {pick}")

    print("\nZero-visit init: N(s)=0 -> exploration term is 0 for all children;")
    print("  scores =", [puct_score(0.0, p, 0, 0, 1.4) for p in (0.6, 0.3, 0.1)],
          "-> tie broken by prior ->",
          select_best_child(_toy_parent(), 1.4).action)

    problem = NavigationProblem(2, 61, 6)
    eng = MCTSEngine(problem, c_puct=1.4, n_simulations=150, seed=3, log_trace=True)
    out = eng.search(problem.initial_state())
    print("\nRoot search (150 sims, c_puct=1.4): visits per action")
    for a, n in sorted(out.visits.items(), key=lambda kv: -kv[1]):
        print(f"   {a:<10} N={n:<4} Q={out.q_values[a]:.3f}")
    print("   first 6 selection-trace lines:")
    print("\n".join(out.trace[:6]))

    ok, plan, final = MCTSEngine(problem, 1.4, 150, seed=3).solve()
    print(f"\nPlan (c_puct=1.4, 150 sims/step): solved={ok}  final={final[0]}")
    print("   " + " | ".join(plan))

    print("\nc_puct sweep over 20 seeds (150 sims/step): success rate, root visit entropy (bits)")
    for c, rate, ent in c_puct_sweep(problem):
        print(f"   c_puct={c:<4} success={rate:>5.0%}  entropy={ent:.2f}")


def _toy_parent() -> MCTSNode:
    p = MCTSNode((0, 0))
    for name, pr in (("hi", 0.6), ("mid", 0.3), ("lo", 0.1)):
        p.children.append(MCTSNode((0, 1), p, name, pr))
    p.expanded = True
    return p


if __name__ == "__main__":
    main()
