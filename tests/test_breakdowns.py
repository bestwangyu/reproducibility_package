import json
import tempfile
import unittest
from pathlib import Path

from dynamic.breakdowns import (
    MachineBreakdownEvent,
    MachineBreakdownGenerator,
    MachineBreakdownScenario,
    load_breakdown_scenarios,
    save_breakdown_scenarios,
)


class MachineBreakdownScenarioTest(unittest.TestCase):
    def test_static_scenario_disables_breakdowns(self):
        scenario = MachineBreakdownScenario.static(num_machines=5)
        self.assertFalse(scenario.breakdowns_enabled)
        self.assertEqual(scenario.events, ())

    def test_generator_is_reproducible_and_non_overlapping(self):
        generator = MachineBreakdownGenerator(
            seed=20260730,
            mean_time_between_failures=8.0,
            mean_repair_time=2.0,
        )
        first = generator.generate(5, events_per_machine=2, scenario_id="case_1")
        second = generator.generate(5, events_per_machine=2, scenario_id="case_1")
        different = generator.generate(5, events_per_machine=2, scenario_id="case_2")

        self.assertEqual(first, second)
        self.assertNotEqual(first.events, different.events)
        self.assertEqual(len(first.events), 10)
        for machine in range(5):
            events = [
                event for event in first.events if event.machine_id == machine
            ]
            self.assertEqual(len(events), 2)
            self.assertGreaterEqual(
                events[1].start_time, events[0].repair_end_time
            )

    def test_json_round_trip_is_stable(self):
        scenarios = MachineBreakdownGenerator(
            seed=20260730,
            mean_time_between_failures=10.0,
            mean_repair_time=3.0,
        ).generate_batch(3, num_machines=5, events_per_machine=1)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            save_breakdown_scenarios(scenarios, first_path)
            loaded = load_breakdown_scenarios(first_path)
            save_breakdown_scenarios(loaded, second_path)
            self.assertEqual(scenarios, loaded)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            payload = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)

    def test_overlapping_failures_on_one_machine_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            MachineBreakdownScenario(
                num_machines=2,
                events=(
                    MachineBreakdownEvent(0, 2, 4),
                    MachineBreakdownEvent(0, 5, 2),
                ),
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
