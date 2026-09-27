import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


BREAKDOWN_SCHEMA_VERSION = 1
PathLike = Union[str, Path]


@dataclass(frozen=True)
class MachineBreakdownEvent:
    machine_id: int
    start_time: float
    repair_duration: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "machine_id", int(self.machine_id))
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "repair_duration", float(self.repair_duration))
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative.")
        if not math.isfinite(self.start_time) or self.start_time <= 0:
            raise ValueError("start_time must be positive and finite.")
        if not math.isfinite(self.repair_duration) or self.repair_duration <= 0:
            raise ValueError("repair_duration must be positive and finite.")

    @property
    def repair_end_time(self) -> float:
        return self.start_time + self.repair_duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "start_time": self.start_time,
            "repair_duration": self.repair_duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            machine_id=int(data["machine_id"]),
            start_time=float(data["start_time"]),
            repair_duration=float(data["repair_duration"]),
        )


@dataclass(frozen=True)
class MachineBreakdownScenario:
    num_machines: int
    events: Tuple[MachineBreakdownEvent, ...]
    seed: int
    scenario_id: str = "0"
    mean_time_between_failures: Optional[float] = None
    mean_repair_time: Optional[float] = None
    schema_version: int = BREAKDOWN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        events = tuple(self.events)
        object.__setattr__(self, "num_machines", int(self.num_machines))
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "scenario_id", str(self.scenario_id))
        if self.schema_version != BREAKDOWN_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported breakdown schema version: {0}; expected {1}.".format(
                    self.schema_version, BREAKDOWN_SCHEMA_VERSION
                )
            )
        if self.num_machines <= 0:
            raise ValueError("num_machines must be positive.")
        if not all(isinstance(event, MachineBreakdownEvent) for event in events):
            raise TypeError("Every event must be a MachineBreakdownEvent.")
        if any(event.machine_id >= self.num_machines for event in events):
            raise ValueError("A breakdown event references an unknown machine.")
        if tuple(sorted(events, key=lambda event: (event.start_time, event.machine_id))) != events:
            raise ValueError("Breakdown events must be sorted by start time and machine.")

        previous_end_by_machine: Dict[int, float] = {}
        for event in events:
            previous_end = previous_end_by_machine.get(event.machine_id, 0.0)
            if event.start_time < previous_end - 1e-9:
                raise ValueError("Breakdown intervals cannot overlap on one machine.")
            previous_end_by_machine[event.machine_id] = event.repair_end_time

        for name, value in (
            ("mean_time_between_failures", self.mean_time_between_failures),
            ("mean_repair_time", self.mean_repair_time),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("{0} must be positive when provided.".format(name))

    @classmethod
    def static(
        cls, num_machines: int, seed: int = 0, scenario_id: str = "static"
    ):
        return cls(
            num_machines=num_machines,
            events=(),
            seed=seed,
            scenario_id=scenario_id,
        )

    @property
    def breakdowns_enabled(self) -> bool:
        return bool(self.events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "num_machines": self.num_machines,
            "mean_time_between_failures": self.mean_time_between_failures,
            "mean_repair_time": self.mean_repair_time,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        required = {
            "schema_version",
            "scenario_id",
            "seed",
            "num_machines",
            "events",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(
                "Breakdown scenario is missing fields: {0}".format(
                    ", ".join(missing)
                )
            )
        return cls(
            schema_version=int(data["schema_version"]),
            scenario_id=str(data["scenario_id"]),
            seed=int(data["seed"]),
            num_machines=int(data["num_machines"]),
            mean_time_between_failures=(
                None
                if data.get("mean_time_between_failures") is None
                else float(data["mean_time_between_failures"])
            ),
            mean_repair_time=(
                None
                if data.get("mean_repair_time") is None
                else float(data["mean_repair_time"])
            ),
            events=tuple(
                MachineBreakdownEvent.from_dict(event)
                for event in data["events"]
            ),
        )


class MachineBreakdownGenerator:
    """Generate reproducible non-overlapping breakdowns per machine."""

    def __init__(
        self,
        seed: int,
        mean_time_between_failures: float,
        mean_repair_time: float,
        min_time_between_failures: float = 1.0,
        min_repair_time: float = 0.5,
        decimals: int = 6,
    ) -> None:
        values = {
            "mean_time_between_failures": mean_time_between_failures,
            "mean_repair_time": mean_repair_time,
            "min_time_between_failures": min_time_between_failures,
            "min_repair_time": min_repair_time,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{0} must be positive and finite.".format(name))
        if decimals < 0:
            raise ValueError("decimals must be non-negative.")
        self.seed = int(seed)
        self.mean_time_between_failures = float(mean_time_between_failures)
        self.mean_repair_time = float(mean_repair_time)
        self.min_time_between_failures = float(min_time_between_failures)
        self.min_repair_time = float(min_repair_time)
        self.decimals = int(decimals)

    def _rng(self, stream_id: Union[str, int]) -> random.Random:
        material = "{0}:{1}".format(self.seed, stream_id).encode("utf-8")
        derived_seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big"
        )
        return random.Random(derived_seed)

    def generate(
        self,
        num_machines: int,
        events_per_machine: int = 1,
        scenario_id: Union[str, int] = 0,
        random_stream_id: Optional[Union[str, int]] = None,
    ) -> MachineBreakdownScenario:
        if num_machines <= 0:
            raise ValueError("num_machines must be positive.")
        if events_per_machine < 0:
            raise ValueError("events_per_machine must be non-negative.")
        stream_id = scenario_id if random_stream_id is None else random_stream_id
        rng = self._rng(stream_id)
        events: List[MachineBreakdownEvent] = []
        failure_rate = 1.0 / self.mean_time_between_failures
        repair_rate = 1.0 / self.mean_repair_time
        for machine_id in range(num_machines):
            previous_repair_end = 0.0
            for _ in range(events_per_machine):
                gap = max(
                    self.min_time_between_failures,
                    rng.expovariate(failure_rate),
                )
                start_time = round(previous_repair_end + gap, self.decimals)
                repair_duration = round(
                    max(self.min_repair_time, rng.expovariate(repair_rate)),
                    self.decimals,
                )
                event = MachineBreakdownEvent(
                    machine_id=machine_id,
                    start_time=start_time,
                    repair_duration=repair_duration,
                )
                events.append(event)
                previous_repair_end = event.repair_end_time
        events.sort(key=lambda event: (event.start_time, event.machine_id))
        return MachineBreakdownScenario(
            num_machines=num_machines,
            events=tuple(events),
            seed=self.seed,
            scenario_id=str(scenario_id),
            mean_time_between_failures=self.mean_time_between_failures,
            mean_repair_time=self.mean_repair_time,
        )

    def generate_batch(
        self,
        count: int,
        num_machines: int,
        events_per_machine: int = 1,
        id_prefix: str = "breakdown",
    ) -> List[MachineBreakdownScenario]:
        if count <= 0:
            raise ValueError("count must be positive.")
        return [
            self.generate(
                num_machines=num_machines,
                events_per_machine=events_per_machine,
                scenario_id="{0}_{1:03d}".format(id_prefix, index),
            )
            for index in range(count)
        ]


def save_breakdown_scenarios(
    scenarios: Iterable[MachineBreakdownScenario], path: PathLike
) -> None:
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one breakdown scenario is required.")
    payload = {
        "schema_version": BREAKDOWN_SCHEMA_VERSION,
        "scenarios": [scenario.to_dict() for scenario in scenario_list],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_breakdown_scenarios(path: PathLike) -> List[MachineBreakdownScenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != BREAKDOWN_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported breakdown scenario file schema version: {0}.".format(
                payload.get("schema_version")
            )
        )
    raw_scenarios: Sequence[Dict[str, Any]] = payload.get("scenarios", [])
    if not raw_scenarios:
        raise ValueError("Breakdown scenario file contains no scenarios.")
    return [MachineBreakdownScenario.from_dict(item) for item in raw_scenarios]
