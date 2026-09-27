import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = 1
PathLike = Union[str, Path]


@dataclass(frozen=True)
class JobArrivalScenario:
    """A reproducible release-time scenario for jobs already stored in an FJS instance."""

    num_jobs: int
    initial_jobs: int
    release_times: Tuple[float, ...]
    seed: int
    scenario_id: str = "0"
    mean_interarrival: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        release_times = tuple(float(value) for value in self.release_times)
        object.__setattr__(self, "release_times", release_times)
        object.__setattr__(self, "scenario_id", str(self.scenario_id))

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported arrival scenario schema version: "
                f"{self.schema_version}; expected {SCHEMA_VERSION}."
            )
        if self.num_jobs <= 0:
            raise ValueError("num_jobs must be positive.")
        if not 1 <= self.initial_jobs <= self.num_jobs:
            raise ValueError("initial_jobs must be between 1 and num_jobs.")
        if len(release_times) != self.num_jobs:
            raise ValueError(
                f"Expected {self.num_jobs} release times, got {len(release_times)}."
            )
        if any(not math.isfinite(value) or value < 0 for value in release_times):
            raise ValueError("Release times must be finite and non-negative.")
        if any(value != 0 for value in release_times[: self.initial_jobs]):
            raise ValueError("All initial jobs must have release time 0.")
        if any(value <= 0 for value in release_times[self.initial_jobs :]):
            raise ValueError("Every arriving job must have a positive release time.")
        if any(left > right for left, right in zip(release_times, release_times[1:])):
            raise ValueError(
                "Release times must be non-decreasing so visible operations form a prefix."
            )
        if self.mean_interarrival is not None:
            if not math.isfinite(self.mean_interarrival) or self.mean_interarrival <= 0:
                raise ValueError("mean_interarrival must be positive when provided.")

    @classmethod
    def static(cls, num_jobs: int, seed: int = 0, scenario_id: str = "static"):
        return cls(
            num_jobs=num_jobs,
            initial_jobs=num_jobs,
            release_times=(0.0,) * num_jobs,
            seed=seed,
            scenario_id=scenario_id,
            mean_interarrival=None,
        )

    @property
    def arrivals_enabled(self) -> bool:
        return self.initial_jobs < self.num_jobs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "num_jobs": self.num_jobs,
            "initial_jobs": self.initial_jobs,
            "mean_interarrival": self.mean_interarrival,
            "release_times": list(self.release_times),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        required = {
            "schema_version",
            "scenario_id",
            "seed",
            "num_jobs",
            "initial_jobs",
            "release_times",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Arrival scenario is missing fields: {', '.join(missing)}")
        return cls(
            schema_version=int(data["schema_version"]),
            scenario_id=str(data["scenario_id"]),
            seed=int(data["seed"]),
            num_jobs=int(data["num_jobs"]),
            initial_jobs=int(data["initial_jobs"]),
            mean_interarrival=(
                None
                if data.get("mean_interarrival") is None
                else float(data["mean_interarrival"])
            ),
            release_times=tuple(float(value) for value in data["release_times"]),
        )


class JobArrivalGenerator:
    """Generate Poisson-arrival scenarios without changing global random state."""

    def __init__(
        self,
        seed: int,
        mean_interarrival: float,
        min_interarrival: float = 1.0,
        decimals: int = 6,
    ) -> None:
        if not math.isfinite(mean_interarrival) or mean_interarrival <= 0:
            raise ValueError("mean_interarrival must be positive.")
        if not math.isfinite(min_interarrival) or min_interarrival <= 0:
            raise ValueError("min_interarrival must be positive.")
        if decimals < 0:
            raise ValueError("decimals must be non-negative.")
        self.seed = int(seed)
        self.mean_interarrival = float(mean_interarrival)
        self.min_interarrival = float(min_interarrival)
        self.decimals = int(decimals)

    def _rng(self, scenario_id: Union[str, int]) -> random.Random:
        material = f"{self.seed}:{scenario_id}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(derived_seed)

    def generate(
        self,
        num_jobs: int,
        initial_jobs: int,
        scenario_id: Union[str, int] = 0,
        random_stream_id: Optional[Union[str, int]] = None,
    ) -> JobArrivalScenario:
        if not 1 <= initial_jobs <= num_jobs:
            raise ValueError("initial_jobs must be between 1 and num_jobs.")

        stream_id = scenario_id if random_stream_id is None else random_stream_id
        rng = self._rng(stream_id)
        release_times: List[float] = [0.0] * initial_jobs
        current_time = 0.0
        arrival_rate = 1.0 / self.mean_interarrival
        for _ in range(num_jobs - initial_jobs):
            interarrival = max(self.min_interarrival, rng.expovariate(arrival_rate))
            current_time = round(current_time + interarrival, self.decimals)
            release_times.append(current_time)

        return JobArrivalScenario(
            num_jobs=num_jobs,
            initial_jobs=initial_jobs,
            release_times=tuple(release_times),
            seed=self.seed,
            scenario_id=str(scenario_id),
            mean_interarrival=self.mean_interarrival,
        )

    def generate_batch(
        self,
        count: int,
        num_jobs: int,
        initial_jobs: int,
        id_prefix: str = "scenario",
    ) -> List[JobArrivalScenario]:
        if count <= 0:
            raise ValueError("count must be positive.")
        return [
            self.generate(
                num_jobs=num_jobs,
                initial_jobs=initial_jobs,
                scenario_id=f"{id_prefix}_{index:03d}",
            )
            for index in range(count)
        ]


def save_scenarios(scenarios: Iterable[JobArrivalScenario], path: PathLike) -> None:
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one arrival scenario is required.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scenarios": [scenario.to_dict() for scenario in scenario_list],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_scenarios(path: PathLike) -> List[JobArrivalScenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported arrival scenario file schema version: "
            f"{payload.get('schema_version')}."
        )
    raw_scenarios: Sequence[Dict[str, Any]] = payload.get("scenarios", [])
    if not raw_scenarios:
        raise ValueError("Arrival scenario file contains no scenarios.")
    return [JobArrivalScenario.from_dict(item) for item in raw_scenarios]
