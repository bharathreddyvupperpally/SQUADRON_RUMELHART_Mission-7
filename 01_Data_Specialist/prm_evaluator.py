"""prm_evaluator.py - Station C (Data & Payload Specialist)
Process Reward Model (PRM) step evaluator + Math-Shepherd label estimation.

* ProcessRewardEvaluator   step-level scores r_t in [0,1], trajectory reductions
                           min / prod / geometric_mean
* outcome_reward           an Outcome Reward Model (ORM) that only sees the final answer
* benchmark_orm_vs_prm     false-positive "trap" benchmark (right answer, broken reasoning)
* MathShepherdEstimator    automatic step labels from M Monte-Carlo rollouts:
                               soft label = (# completions reaching the gold answer) / M
                               hard label = 1 if ANY completion reaches it, else 0

The step verifier is a transparent stand-in for a learned PRM: it parses a step of
the form "a op b = c" and returns p_valid when the arithmetic is right AND the step
continues from the previous result, p_invalid when not, and 0.5 for text it cannot
verify.  A learned PRM would output the same kind of per-step probability.

Run `python prm_evaluator.py`.  Standard library only.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

EQ = re.compile(r"(-?\d+)\s*([+\-*x×])\s*(-?\d+)\s*=\s*(-?\d+)")
ANSWER = re.compile(r"answer\s*[:=]\s*(-?\d+)", re.I)
REDUCTIONS = ("min", "prod", "geometric_mean")


# ----------------------------------------------------------------------
def parse_equation(step: str):
    """Return (a, op, b, claimed) or None."""
    m = EQ.search(step)
    if not m:
        return None
    a, op, b, c = m.groups()
    return int(a), ("*" if op in "x×" else op), int(b), int(c)


def evaluate_op(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def reduce_scores(scores: Sequence[float], method: str = "min") -> float:
    """Trajectory reward from step rewards."""
    if not scores:
        return 0.0
    if method == "min":
        return min(scores)
    if method == "prod":
        out = 1.0
        for s in scores:
            out *= s
        return out
    if method == "geometric_mean":
        if min(scores) <= 0.0:
            return 0.0
        return math.exp(sum(math.log(s) for s in scores) / len(scores))
    raise ValueError(f"unknown reduction {method!r}; choose from {REDUCTIONS}")


class ArithmeticStepVerifier:
    """Stand-in PRM head: P(step is correct | history)."""

    def __init__(self, p_valid: float = 0.96, p_invalid: float = 0.04, p_unknown: float = 0.5):
        self.p_valid, self.p_invalid, self.p_unknown = p_valid, p_invalid, p_unknown

    def __call__(self, step: str, history: Sequence[str] = ()) -> float:
        eq = parse_equation(step)
        if eq is None:
            return self.p_unknown
        a, op, b, c = eq
        if evaluate_op(a, op, b) != c:
            return self.p_invalid                      # arithmetic is wrong
        for prev in reversed(history):                 # must continue from last result
            pe = parse_equation(prev)
            if pe is not None:
                if pe[3] != a and pe != (a, op, b, c):
                    return self.p_invalid              # broken chain (a verbatim restatement is fine)
                break
        return self.p_valid


@dataclass
class TrajectoryScore:
    step_scores: List[float]
    reduction: str
    value: float
    first_bad_step: Optional[int]      # 1-based index of first step below `bad_below`


class ProcessRewardEvaluator:
    def __init__(self, step_scorer: Optional[Callable] = None, reduction: str = "min",
                 bad_below: float = 0.5):
        assert reduction in REDUCTIONS
        self.step_scorer = step_scorer or ArithmeticStepVerifier()
        self.reduction = reduction
        self.bad_below = bad_below

    def score_steps(self, steps: Sequence[str]) -> List[float]:
        return [self.step_scorer(s, steps[:i]) for i, s in enumerate(steps)]

    def score_trajectory(self, steps: Sequence[str], reduction: Optional[str] = None) -> TrajectoryScore:
        red = reduction or self.reduction
        sc = self.score_steps(steps)
        bad = next((i + 1 for i, s in enumerate(sc) if s < self.bad_below), None)
        return TrajectoryScore(sc, red, reduce_scores(sc, red), bad)

    def accepts(self, steps: Sequence[str], threshold: float = 0.5, reduction: Optional[str] = None) -> bool:
        return self.score_trajectory(steps, reduction).value >= threshold

    def rank(self, trajectories: Dict[str, Sequence[str]], reduction: Optional[str] = None):
        scored = [(k, self.score_trajectory(v, reduction).value) for k, v in trajectories.items()]
        return sorted(scored, key=lambda kv: -kv[1])


def compare_reductions(score_lists: Dict[str, Sequence[float]]):
    return {name: {m: reduce_scores(s, m) for m in REDUCTIONS} for name, s in score_lists.items()}


# ----------------------------------------------------------------------
# Problem + ORM trap benchmark
class ArithmeticProblem:
    def __init__(self, start: int, ops: Sequence[tuple]):
        self.start, self.ops = start, list(ops)
        self.states = [start]
        for op, k in self.ops:
            self.states.append(evaluate_op(self.states[-1], op, k))
        self.gold = self.states[-1]

    def correct_steps(self) -> List[str]:
        return [f"{self.states[i]} {op} {k} = {self.states[i + 1]}" for i, (op, k) in enumerate(self.ops)]


@dataclass
class Trajectory:
    name: str
    steps: List[str]
    final_answer: int
    truly_valid: bool          # is every step correct?


def outcome_reward(traj: Trajectory, gold: int) -> float:
    """ORM: looks ONLY at the final answer."""
    return 1.0 if traj.final_answer == gold else 0.0


def trap_benchmark(problem: ArithmeticProblem) -> List[Trajectory]:
    g = problem.gold
    return [
        Trajectory("valid", problem.correct_steps(), g, True),
        Trajectory("trap_cancelling_errors",   # two wrong steps whose errors cancel
                   ["7 + 5 = 12", "12 * 3 = 36", "36 - 4 = 30", "30 * 2 = 64"], g, False),
        Trajectory("trap_lucky_guess",         # nonsense steps, correct final answer
                   ["7 + 5 = 13", "13 * 3 = 41", "41 - 4 = 40", "40 * 2 = 80"], g, False),
        Trajectory("trap_broken_chain",        # each equation true, but chain jumps
                   ["7 + 5 = 12", "10 * 3 = 30", "30 - 4 = 26", "32 * 2 = 64"], g, False),
        Trajectory("wrong_final", ["7 + 5 = 12", "12 * 3 = 35", "35 - 4 = 31", "31 * 2 = 62"], 62, False),
    ]


def benchmark_orm_vs_prm(problem: ArithmeticProblem, threshold: float = 0.5, reduction: str = "min"):
    prm = ProcessRewardEvaluator(reduction=reduction)
    rows, orm_fp, prm_fp, invalid = [], 0, 0, 0
    for t in trap_benchmark(problem):
        orm = outcome_reward(t, problem.gold)
        ts = prm.score_trajectory(t.steps)
        prm_accept = ts.value >= threshold
        if not t.truly_valid:
            invalid += 1
            orm_fp += orm >= 1.0
            prm_fp += prm_accept
        rows.append((t.name, t.truly_valid, orm, ts.value, ts.first_bad_step))
    return rows, orm_fp, prm_fp, invalid


# ----------------------------------------------------------------------
# Math-Shepherd
class MathShepherdEstimator:
    """Label step k by sampling M completions from the state reached after step k."""

    def __init__(self, problem: ArithmeticProblem, step_error: float = 0.1, M: int = 8, seed: int = 0):
        self.problem, self.e, self.M = problem, step_error, M
        self.rng = random.Random(seed)

    def _complete(self, value: int, from_step: int) -> int:
        for op, k in self.problem.ops[from_step:]:
            value = evaluate_op(value, op, k)
            if self.rng.random() < self.e:             # completer slips
                value += self.rng.choice([-3, -2, -1, 1, 2, 3])
        return value

    def label_step(self, claimed_value: int, step_index: int):
        """step_index = number of ops already done (1-based step k -> k)."""
        hits = sum(self._complete(claimed_value, step_index) == self.problem.gold for _ in range(self.M))
        return hits / self.M, int(hits > 0)            # (soft, hard)

    def label_trajectory(self, steps: Sequence[str]):
        out = []
        for k, s in enumerate(steps, start=1):
            eq = parse_equation(s)
            claimed = eq[3] if eq else self.problem.start
            soft, hard = self.label_step(claimed, k)
            out.append((s, soft, hard))
        return out

    def analytic_soft(self, step_index: int) -> float:
        """P(correct final | correct state after step k), ignoring lucky cancellations."""
        return (1 - self.e) ** (len(self.problem.ops) - step_index)


def estimator_error(problem: ArithmeticProblem, Ms=(1, 4, 16, 64), e=0.1, trials=300, seed=1):
    """Mean absolute error of the soft label vs the analytic value (correct steps only)."""
    rows = []
    for M in Ms:
        errs = []
        for t in range(trials):
            est = MathShepherdEstimator(problem, e, M, seed=seed * 10_000 + t)
            for k in range(1, len(problem.ops) + 1):
                soft, _ = est.label_step(problem.states[k], k)
                errs.append(abs(soft - est.analytic_soft(k)))
        rows.append((M, sum(errs) / len(errs)))
    return rows


def hard_label_false_positive_rate(problem: ArithmeticProblem, M: int, offset: int = 1,
                                   e=0.1, trials=2000, seed=5):
    """How often a WRONG step (state off by `offset` after step 2) still gets hard label 1,
    because some completion slips in exactly the compensating way."""
    est = MathShepherdEstimator(problem, e, M, seed=seed)
    wrong_state = problem.states[2] + offset
    fp = sum(est.label_step(wrong_state, 2)[1] for _ in range(trials))
    return fp / trials


# ----------------------------------------------------------------------
def main():
    problem = ArithmeticProblem(7, [("+", 5), ("*", 3), ("-", 4), ("*", 2)])
    print(f"SEC-07-SGR  Station C  PRM   problem: 7 +5 *3 -4 *2 -> gold = {problem.gold}\n")

    lists = {
        "all good, 4 steps":        [0.96] * 4,
        "all good, 12 steps":       [0.96] * 12,
        "one bad step (0.04)":      [0.96, 0.96, 0.04, 0.96],
        "uniformly mediocre (.7)":  [0.7] * 4,
        "strong then weak (.55)":   [0.99, 0.99, 0.99, 0.55],
    }
    print("Reductions of step scores")
    print(f"{'trajectory':<26} {'min':>7} {'prod':>7} {'geo_mean':>9}")
    for name, r in compare_reductions(lists).items():
        print(f"{name:<26} {r['min']:>7.3f} {r['prod']:>7.3f} {r['geometric_mean']:>9.3f}")

    print("\nORM vs PRM on the false-positive trap benchmark (PRM = min, accept if >= 0.5)")
    rows, orm_fp, prm_fp, invalid = benchmark_orm_vs_prm(problem)
    print(f"{'trajectory':<30} {'valid':<6} {'ORM':>4} {'PRM(min)':>9} first bad step")
    for n, v, o, p, b in rows:
        print(f"{n:<30} {str(v):<6} {o:>4.0f} {p:>9.3f} {b if b else '-'}")
    print(f"flawed trajectories: {invalid}   ORM accepts (false positives): {orm_fp}   "
          f"PRM accepts: {prm_fp}")
    traj = trap_benchmark(problem)[1]
    print("\nStep scores for trap_cancelling_errors:",
          [round(s, 2) for s in ProcessRewardEvaluator().score_steps(traj.steps)])

    print("\nMath-Shepherd labels (M=16 rollouts, completer slip rate e=0.10) on a flawed trajectory")
    est = MathShepherdEstimator(problem, 0.10, 16, seed=2)
    flawed = ["7 + 5 = 12", "12 * 3 = 35", "35 - 4 = 31", "31 * 2 = 62"]
    print(f"{'step':<14} {'soft':>5} {'hard':>5} {'analytic soft (if step were right)':>36}")
    for k, (s, soft, hard) in enumerate(est.label_trajectory(flawed), start=1):
        print(f"{s:<14} {soft:>5.2f} {hard:>5} {est.analytic_soft(k):>36.3f}")

    print("\nSoft-label error vs number of rollouts M (mean abs. error vs analytic, 300 trials)")
    for M, err in estimator_error(problem):
        print(f"   M={M:<3} MAE={err:.3f}")
    print("\nHard-label false positives on a WRONG step after step 2 (2000 trials)")
    print(f"   {'M':<4} {'off by 1':>9} {'off by 7':>9}")
    for M in (1, 4, 16, 64):
        print(f"   {M:<4} {hard_label_false_positive_rate(problem, M, 1):>9.3f} "
              f"{hard_label_false_positive_rate(problem, M, 7):>9.3f}")


if __name__ == "__main__":
    main()
