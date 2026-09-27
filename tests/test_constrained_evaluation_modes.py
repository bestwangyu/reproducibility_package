import unittest

from evaluate_constrained_pilot import select_sample


class ConstrainedEvaluationModeTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"makespan": 10.0, "stability_cost": 2.0},
            {"makespan": 12.0, "stability_cost": 1.6},
            {"makespan": 14.0, "stability_cost": 1.2},
        ]

    def test_budget_first_prefers_fastest_feasible_candidate(self):
        selected = select_sample(self.rows, 1.7, "budget_first")
        self.assertEqual(selected["makespan"], 12.0)

    def test_budget_first_falls_back_to_minimum_cost(self):
        selected = select_sample(self.rows, 1.0, "budget_first")
        self.assertEqual(selected["stability_cost"], 1.2)

    def test_makespan_only_ignores_budget(self):
        selected = select_sample(self.rows, 1.7, "makespan_only")
        self.assertEqual(selected["makespan"], 10.0)

    def test_min_cost_ignores_makespan_priority(self):
        selected = select_sample(self.rows, 1.7, "min_cost")
        self.assertEqual(selected["stability_cost"], 1.2)

    def test_empty_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            select_sample([], 1.7, "budget_first")


if __name__ == "__main__":
    unittest.main()
