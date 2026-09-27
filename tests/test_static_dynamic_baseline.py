import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.gym_compat import install_gym_stub_if_needed

install_gym_stub_if_needed()

from evaluate_static_dynamic_baseline import build_result_rows, build_summary


class StaticDynamicBaselineTest(unittest.TestCase):
    def test_build_result_rows_uses_explicit_disruption_delta(self):
        rows = build_result_rows(
            [Path("instance.fjs")],
            [10.0],
            [
                {
                    "makespan": 13.5,
                    "stability_cost": 1.25,
                    "budget_violation": 0.0,
                    "budget_met": True,
                    "release_violations": 0,
                }
            ],
        )
        self.assertEqual(rows[0]["file_name"], "instance.fjs")
        self.assertEqual(rows[0]["disruption_makespan_delta"], 3.5)
        self.assertTrue(rows[0]["budget_met"])

    def test_summary_passes_only_when_both_schedules_are_feasible(self):
        args = SimpleNamespace(
            policy_seed=20260805,
            seed=20260825,
            instances=1,
            offset=20,
            initial_jobs=6,
            mean_interarrival=8.0,
            mean_time_between_failures=8.0,
            mean_repair_time=2.0,
            events_per_machine=1,
            stability_budget=1.7,
        )
        rows = [
            {
                "reference_makespan": 10.0,
                "dynamic_makespan": 12.0,
                "disruption_makespan_delta": 2.0,
                "stability_cost": 1.5,
                "budget_violation": 0.0,
                "budget_met": True,
                "release_violations": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = build_summary(
                args,
                checkpoint,
                checkpoint,
                True,
                0,
                True,
                rows,
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["budget_met_rate"], 1.0)

            failed = build_summary(
                args,
                checkpoint,
                checkpoint,
                False,
                0,
                True,
                rows,
            )
            self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
