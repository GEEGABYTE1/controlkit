
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt


@dataclass(frozen=True)
class ClosedLoopMetrics:
    benchmark_name: str
    controller_type: str
    dt: float
    horizon_steps: int
    mean_runtime_us: float
    max_runtime_us: float
    p95_runtime_us: float
    final_state_norm: float
    max_state_norm: float
    total_control_effort: float
    passed: bool
    failure_reason: str
    generated_mean_runtime_us: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def vector_norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def control_effort(control: list[float], dt: float) -> float:
    return sum(abs(value) for value in control) * dt


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile_value / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(index, len(ordered) - 1))]
