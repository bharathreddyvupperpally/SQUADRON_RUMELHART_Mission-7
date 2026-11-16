"""run_mission.py - Integrated TTC stack (all four stations together).

Experiment: the AURA proposer hallucinates a wrong arithmetic result with probability h
and the search state then follows the wrong claim (error compounding).  A search can
therefore *claim* the goal while its reasoning chain is broken.  We measure:

    claimed  = search reached the target state
    verified = claimed AND every step in the chain passes the PRM check (true success)

Stack under test:
    ToT-BFS            (Station A)  alone vs + EventHorizonPruner (Station D, PRM from C)
    MCTS+PUCT          (Station B)  plain vs PRM-guided priors/rollouts (Station C)

Standard library only.  Run from anywhere:  python run_mission.py  (the four station folders must stay side by side)
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                       # nav_domain, tot_solver (this folder)
for d in ("01_Data_Specialist", "02_Protocol_Specialist", "03_Security_Sentinel"):
    sys.path.append(os.path.join(ROOT, d))     # prm_evaluator, mcts_protocol, horizon_pruner

from nav_domain import NavigationProblem              # noqa: E402
from tot_solver import TreeOfThoughtsSolver           # noqa: E402
from mcts_protocol import MCTSEngine                  # noqa: E402
from prm_evaluator import ProcessRewardEvaluator      # noqa: E402
from horizon_pruner import EventHorizonPruner         # noqa: E402

SEEDS = range(40)
RATES = (0.0, 0.1, 0.2, 0.3)
PRM = ProcessRewardEvaluator(reduction="min")


def verified(steps):
    return all(s >= 0.5 for s in PRM.score_steps(steps))


def tot_trial(h, seed, pruned):
    prob = NavigationProblem(2, 61, 6, hallucination_rate=h, seed=seed)
    pf = EventHorizonPruner(0.35, "geometric_mean").node_filter() if pruned else None
    r = TreeOfThoughtsSolver(prob, 3, 40, prune_fn=pf).bfs_beam_search()
    claimed = r.success
    return claimed, claimed and verified(r.solution), r.expansions


class PRMGuidedMCTS(MCTSEngine):
    """Station B engine with Station C's PRM shaping priors and gating rollouts."""

    def priors(self, thoughts):
        base = super().priors(thoughts)
        shaped = [p * PRM.step_scorer(t.text, ()) for p, t in zip(base, thoughts)]
        return [x / sum(shaped) for x in shaped]

    def simulate(self, node):
        state, ok = node.state, True
        while not self.problem.is_terminal(state):
            th = self.rng.choice(self.problem.propose(state))
            ok &= PRM.step_scorer(th.text, ()) >= 0.5
            state = th.state
        return self.problem.value(state) if ok else 0.0


def mcts_trial(h, seed, guided):
    prob = NavigationProblem(2, 61, 6, hallucination_rate=h, seed=seed)
    cls = PRMGuidedMCTS if guided else MCTSEngine
    claimed, plan, _ = cls(prob, 1.4, 120, seed=seed).solve()
    return claimed, claimed and verified(plan)


def pct(x, n):
    return f"{100 * x / n:5.1f}%"


def main():
    n = len(SEEDS)
    print("INTEGRATED TTC STACK   (start=2, target=61, 40 problem seeds per cell)\n")
    print("ToT-BFS (B=3, budget=40 expansions)")
    print(f"{'halluc.':>8} | {'no pruner':^30} | {'+ Event Horizon Pruner':^30}")
    print(f"{'rate h':>8} | {'claimed':>9} {'verified':>9} {'exp.':>8} | {'claimed':>9} {'verified':>9} {'exp.':>8}")
    for h in RATES:
        cells = []
        for pruned in (False, True):
            res = [tot_trial(h, s, pruned) for s in SEEDS]
            cells.append((sum(r[0] for r in res), sum(r[1] for r in res), sum(r[2] for r in res) / n))
        print(f"{h:>8} | {pct(cells[0][0], n):>9} {pct(cells[0][1], n):>9} {cells[0][2]:>8.1f} | "
              f"{pct(cells[1][0], n):>9} {pct(cells[1][1], n):>9} {cells[1][2]:>8.1f}")

    print("\nMCTS+PUCT (c_puct=1.4, 120 simulations / step)")
    print(f"{'halluc.':>8} | {'plain':^19} | {'PRM-guided':^19}")
    print(f"{'rate h':>8} | {'claimed':>9} {'verified':>9} | {'claimed':>9} {'verified':>9}")
    for h in RATES:
        cells = []
        for guided in (False, True):
            res = [mcts_trial(h, s, guided) for s in SEEDS]
            cells.append((sum(r[0] for r in res), sum(r[1] for r in res)))
        print(f"{h:>8} | {pct(cells[0][0], n):>9} {pct(cells[0][1], n):>9} | "
              f"{pct(cells[1][0], n):>9} {pct(cells[1][1], n):>9}")

    # one concrete mission trace
    prob = NavigationProblem(2, 61, 6, hallucination_rate=0.2, seed=30)
    plain = TreeOfThoughtsSolver(prob, 3, 40).bfs_beam_search()
    pf = EventHorizonPruner(0.35, "geometric_mean")
    guarded = TreeOfThoughtsSolver(prob, 3, 40, prune_fn=pf.node_filter()).bfs_beam_search()
    print("\nMission trace, h=0.2, seed 30")
    for label, r in (("no pruner", plain), ("with pruner", guarded)):
        print(f"  {label:<11} claimed={r.success}  verified={r.success and verified(r.solution)}  "
              f"expansions={r.expansions}  pruned nodes={r.nodes_pruned}")
        print("     " + " | ".join(r.solution))


if __name__ == "__main__":
    main()
