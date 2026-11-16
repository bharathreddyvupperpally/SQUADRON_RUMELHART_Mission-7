"""Unit tests for Station C (run: python test_prm_evaluator.py)."""
import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prm_evaluator import (ArithmeticProblem, ArithmeticStepVerifier, MathShepherdEstimator,
                           ProcessRewardEvaluator, benchmark_orm_vs_prm, reduce_scores,
                           estimator_error, parse_equation)

P = ArithmeticProblem(7, [("+", 5), ("*", 3), ("-", 4), ("*", 2)])


class TestPRM(unittest.TestCase):
    def test_reductions_exact(self):
        s = [0.5, 0.8, 0.9]
        self.assertEqual(reduce_scores(s, "min"), 0.5)
        self.assertAlmostEqual(reduce_scores(s, "prod"), 0.36)
        self.assertAlmostEqual(reduce_scores(s, "geometric_mean"), 0.36 ** (1 / 3))

    def test_prod_penalises_length_but_geo_mean_does_not(self):
        short, long_ = [0.9] * 3, [0.9] * 12
        self.assertGreater(reduce_scores(short, "prod"), reduce_scores(long_, "prod"))
        self.assertAlmostEqual(reduce_scores(short, "geometric_mean"), reduce_scores(long_, "geometric_mean"))

    def test_zero_score_geo_mean(self):
        self.assertEqual(reduce_scores([0.9, 0.0], "geometric_mean"), 0.0)

    def test_bad_reduction_raises(self):
        with self.assertRaises(ValueError):
            reduce_scores([1.0], "median")

    def test_parse_and_verify(self):
        self.assertEqual(parse_equation("so 12 * 3 = 36 ok"), (12, "*", 3, 36))
        v = ArithmeticStepVerifier()
        self.assertGreater(v("12 * 3 = 36"), 0.9)
        self.assertLess(v("12 * 3 = 35"), 0.1)
        self.assertEqual(v("just vibes"), 0.5)
        self.assertLess(v("10 * 3 = 30", ["7 + 5 = 12"]), 0.1)      # broken chain

    def test_prm_catches_all_orm_traps(self):
        rows, orm_fp, prm_fp, invalid = benchmark_orm_vs_prm(P)
        self.assertEqual(invalid, 4)
        self.assertEqual(orm_fp, 3)
        self.assertEqual(prm_fp, 0)

    def test_first_bad_step_located(self):
        ts = ProcessRewardEvaluator().score_trajectory(["7 + 5 = 12", "12 * 3 = 36", "36 - 4 = 30"])
        self.assertEqual(ts.first_bad_step, 3)

    def test_math_shepherd_correct_state_high_wrong_state_low(self):
        est = MathShepherdEstimator(P, 0.1, 64, seed=0)
        good, _ = est.label_step(P.states[1], 1)
        bad, _ = est.label_step(P.states[1] + 9, 1)
        self.assertGreater(good, 0.5)
        self.assertLess(bad, 0.1)

    def test_soft_label_error_shrinks_with_M(self):
        e = dict(estimator_error(P, (4, 64), trials=80))
        self.assertLess(e[64], e[4])

    def test_zero_slip_is_deterministic(self):
        est = MathShepherdEstimator(P, 0.0, 4)
        self.assertEqual(est.label_step(P.states[2], 2), (1.0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
