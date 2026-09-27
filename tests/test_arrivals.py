import json
import tempfile
import unittest
from pathlib import Path

from dynamic.arrivals import (
    JobArrivalGenerator,
    JobArrivalScenario,
    load_scenarios,
    save_scenarios,
)


class JobArrivalScenarioTest(unittest.TestCase):
    def test_static_scenario_disables_arrivals(self):
        scenario = JobArrivalScenario.static(num_jobs=10, seed=7)

        self.assertFalse(scenario.arrivals_enabled)
        self.assertEqual(scenario.initial_jobs, 10)
        self.assertEqual(scenario.release_times, (0.0,) * 10)

    def test_generator_is_reproducible_and_does_not_touch_global_rng(self):
        generator = JobArrivalGenerator(seed=20260728, mean_interarrival=8.0)

        first = generator.generate(num_jobs=10, initial_jobs=6, scenario_id="case_1")
        second = generator.generate(num_jobs=10, initial_jobs=6, scenario_id="case_1")
        different = generator.generate(num_jobs=10, initial_jobs=6, scenario_id="case_2")

        self.assertEqual(first, second)
        self.assertNotEqual(first.release_times, different.release_times)
        self.assertEqual(first.release_times[:6], (0.0,) * 6)
        self.assertTrue(all(value > 0 for value in first.release_times[6:]))
        self.assertEqual(tuple(sorted(first.release_times)), first.release_times)

    def test_json_round_trip_is_stable(self):
        scenarios = JobArrivalGenerator(
            seed=20260728, mean_interarrival=10.0
        ).generate_batch(count=3, num_jobs=10, initial_jobs=5)

        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            save_scenarios(scenarios, first_path)
            loaded = load_scenarios(first_path)
            save_scenarios(loaded, second_path)

            self.assertEqual(scenarios, loaded)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            payload = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["scenarios"]), 3)

    def test_intensity_levels_can_share_common_random_numbers(self):
        low = JobArrivalGenerator(seed=20260728, mean_interarrival=12.0).generate(
            num_jobs=10,
            initial_jobs=6,
            scenario_id="low:case_1",
            random_stream_id="case_1",
        )
        medium = JobArrivalGenerator(seed=20260728, mean_interarrival=8.0).generate(
            num_jobs=10,
            initial_jobs=6,
            scenario_id="medium:case_1",
            random_stream_id="case_1",
        )
        high = JobArrivalGenerator(seed=20260728, mean_interarrival=4.0).generate(
            num_jobs=10,
            initial_jobs=6,
            scenario_id="high:case_1",
            random_stream_id="case_1",
        )

        self.assertTrue(
            all(
                low_time >= medium_time >= high_time
                for low_time, medium_time, high_time in zip(
                    low.release_times,
                    medium.release_times,
                    high.release_times,
                )
            )
        )

    def test_invalid_non_prefix_release_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            JobArrivalScenario(
                num_jobs=4,
                initial_jobs=2,
                release_times=(0, 0, 5, 3),
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
