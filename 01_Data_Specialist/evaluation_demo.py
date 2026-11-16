"""evaluation_demo.py - Station C demo: PRM vs ORM on the trap benchmark + Math-Shepherd labels.

Writes sample_trajectories.jsonl (one JSON object per trajectory: steps, ORM reward, PRM step
scores, min/prod/geometric-mean rewards, first bad step, Math-Shepherd soft/hard labels).
Run:  python evaluation_demo.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prm_evaluator import (ArithmeticProblem, MathShepherdEstimator, ProcessRewardEvaluator,  # noqa: E402
                           outcome_reward, reduce_scores, trap_benchmark)


def main():
    problem = ArithmeticProblem(7, [("+", 5), ("*", 3), ("-", 4), ("*", 2)])
    prm = ProcessRewardEvaluator(reduction="min")
    est = MathShepherdEstimator(problem, step_error=0.10, M=16, seed=2)
    out = os.path.join(HERE, "sample_trajectories.jsonl")
    print(f"problem: 7 +5 *3 -4 *2   gold = {problem.gold}\n")
    print(f"{'trajectory':<26}{'valid':<7}{'ORM':>4}{'min':>7}{'prod':>7}{'geo':>7}  first bad  soft labels")
    with open(out, "w") as f:
        for t in trap_benchmark(problem):
            ts = prm.score_trajectory(t.steps)
            ms = est.label_trajectory(t.steps)
            row = dict(name=t.name, steps=t.steps, final_answer=t.final_answer, truly_valid=t.truly_valid,
                       orm=outcome_reward(t, problem.gold), prm_step_scores=ts.step_scores,
                       prm_min=reduce_scores(ts.step_scores, "min"),
                       prm_prod=round(reduce_scores(ts.step_scores, "prod"), 4),
                       prm_geometric_mean=round(reduce_scores(ts.step_scores, "geometric_mean"), 4),
                       first_bad_step=ts.first_bad_step,
                       math_shepherd_soft=[round(s, 3) for _, s, _ in ms],
                       math_shepherd_hard=[h for _, _, h in ms])
            f.write(json.dumps(row) + "\n")
            print(f"{t.name:<26}{str(t.truly_valid):<7}{row['orm']:>4.0f}{row['prm_min']:>7.2f}"
                  f"{row['prm_prod']:>7.3f}{row['prm_geometric_mean']:>7.3f}  "
                  f"{str(ts.first_bad_step):<10} {row['math_shepherd_soft']}")
    print(f"\nwrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
