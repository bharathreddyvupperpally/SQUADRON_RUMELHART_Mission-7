"""anomaly_07_demo.py - Station D demo: Ship Anomaly 07 (adversarial PRM exploitation).

Shows, step by step, how a style-biased PRM is fooled by sycophantic filler and by step-splitting
inflation, and how the Event Horizon Pruner severs the branch.  Also writes pruned_branches_log.json.
Run:  python anomaly_07_demo.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from horizon_pruner import (ATTACKS, EventHorizonPruner, StyleBiasedPRM, reduce,  # noqa: E402
                            run_attack_matrix)

TAU = 0.35


def step_by_step(name):
    steps, _ = ATTACKS[name]
    prm = StyleBiasedPRM()
    pruner = EventHorizonPruner(TAU, "geometric_mean", prm)
    print(f"\n--- {name} ---")
    print(f"{'k':>2} {'PRM step':>8} {'raw run.':>9} {'defended':>9}  step")
    severed = False
    for k in range(1, len(steps) + 1):
        pre = steps[:k]
        step_score = prm(steps[k - 1], pre[:-1])
        raw = reduce([prm(s, pre[:i]) for i, s in enumerate(pre)], "geometric_mean")
        dfn = pruner.defended_score(pre)
        mark = ""
        if not severed and k >= pruner.grace and dfn < TAU:
            mark, severed = "  <== SEVERED (r_PRM < tau_horizon)", True
        print(f"{k:>2} {step_score:>8.2f} {raw:>9.3f} {dfn:>9.3f}  {steps[k - 1][:58]}{mark}")


def main():
    print(f"SHIP ANOMALY 07 - adversarial PRM exploitation   (tau_horizon={TAU}, geometric mean)")
    for name in ("atk1_flattery_on_wrong_step", "atk2_duplicate_splitting"):
        step_by_step(name)

    modes, table = run_attack_matrix(TAU, "geometric_mean")
    print("\nAttack matrix (PRUNED = branch severed)")
    print(f"{'trajectory':<30}{'valid':<7}" + "".join(f"{m[:26]:<28}" for m in modes))
    for name, (valid, row) in table.items():
        print(f"{name:<30}{str(valid):<7}" + "".join(f"{('PRUNED' if row[m] else 'survives'):<28}" for m in modes))

    pr = EventHorizonPruner(TAU, "geometric_mean", StyleBiasedPRM())
    for name, (steps, _) in ATTACKS.items():
        pr.evaluate(steps, name)
    s = pr.summary()
    print(f"\npruned {s['pruned']}/{s['trajectories']} trajectories; tokens saved {s['tokens_saved']} "
          f"of {s['tokens_total']} ({s['saved_pct']}%)")
    pr.save_log(os.path.join(HERE, "pruned_branches_log.json"))
    print("wrote pruned_branches_log.json")


if __name__ == "__main__":
    main()
