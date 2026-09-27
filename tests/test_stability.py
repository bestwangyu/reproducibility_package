import unittest

from dynamic.stability import (
    StabilityBudgetTracker,
    StabilityMetricConfig,
    calculate_schedule_stability,
)


class ScheduleStabilityTest(unittest.TestCase):
    def test_identical_schedules_have_zero_cost(self):
        schedule = [
            [1, 0, 0, 4],
            [1, 1, 4, 6],
        ]
        result = calculate_schedule_stability(schedule, schedule)
        self.assertEqual(result.operation_count, 2)
        self.assertEqual(result.start_deviation_sum, 0)
        self.assertEqual(result.machine_changes, 0)
        self.assertEqual(result.stability_cost_mean, 0)

    def test_start_deviation_and_machine_change_are_weighted(self):
        reference = [
            [1, 0, 0, 4],
            [1, 1, 4, 6],
        ]
        disrupted = [
            [1, 0, 2, 6],
            [1, 2, 3, 5],
        ]
        result = calculate_schedule_stability(
            reference,
            disrupted,
            config=StabilityMetricConfig(
                start_time_weight=2.0,
                machine_change_weight=3.0,
            ),
        )
        self.assertAlmostEqual(result.normalized_start_deviation_sum, 1.0)
        self.assertEqual(result.machine_changes, 1)
        self.assertAlmostEqual(result.stability_cost_sum, 5.0)
        self.assertAlmostEqual(result.stability_cost_mean, 2.5)

    def test_budget_accounting_before_and_after_violation(self):
        tracker = StabilityBudgetTracker(budget=2.0)
        first = tracker.consume(0.75)
        self.assertAlmostEqual(first.remaining, 1.25)
        self.assertEqual(first.violation, 0)
        self.assertLess(first.accounting_error, 1e-12)

        second = tracker.consume(1.75)
        self.assertEqual(second.remaining, 0)
        self.assertAlmostEqual(second.violation, 0.5)
        self.assertLess(second.accounting_error, 1e-12)

    def test_unscheduled_operations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unscheduled"):
            calculate_schedule_stability(
                [[1, 0, 0, 2]],
                [[0, 0, 0, 0]],
            )


if __name__ == "__main__":
    unittest.main()
