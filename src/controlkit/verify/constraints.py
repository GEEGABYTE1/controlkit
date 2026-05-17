#constraint sanity checks

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class ConstraintResult:
    passed: bool
    warnings: list[str]
    errors: list[str]


def check_constraints(
    *,
    gain_matrix: list[list[float]] | None,
    input_lower: list[float] | None,
    input_upper: list[float] | None,
    state_lower: list[float] | None = None,
    state_upper: list[float] | None = None,
    saturation_declared: bool = False,
) -> ConstraintResult:
    warnings: list[str] = []
    errors: list[str] = []
    _check_bounds("input", input_lower, input_upper, errors)
    _check_bounds("state", state_lower, state_upper, errors)
    if (input_lower is not None or input_upper is not None) and not saturation_declared:
        errors.append("actuator limits declared without saturation policy")
    if (
        gain_matrix is not None
        and input_lower is not None
        and input_upper is not None
        and state_lower is not None
        and state_upper is not None
    ):
        worst = _worst_case_feedback(gain_matrix, state_lower, state_upper)
        for index, value in enumerate(worst):
            limit = max(abs(input_lower[index]), abs(input_upper[index]))
            if value > limit:
                warnings.append(
                    f"controller output dimension {index} can reach {value:.6g}, "
                    f"above actuator limit {limit:.6g}; saturation will clip"
                )
    return ConstraintResult(passed=not errors, warnings=warnings, errors=errors)


def _check_bounds(
    name: str,
    lower: list[float] | None,
    upper: list[float] | None,
    errors: list[str],
) -> None:
    if lower is None and upper is None:
        return
    if lower is None or upper is None:
        errors.append(f"{name} bounds require both lower and upper values")
        return
    if len(lower) != len(upper):
        errors.append(f"{name} bounds length mismatch")
        return
    for index, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        if lo >= hi:
            errors.append(f"{name} bounds invalid at index {index}: lower must be < upper")


def _worst_case_feedback(
    gain_matrix: list[list[float]],
    state_lower: list[float],
    state_upper: list[float],
) -> list[float]:
    corners = product(*[(state_lower[index], state_upper[index]) for index in range(len(state_lower))])
    worst = [0.0 for _ in gain_matrix]
    for corner in corners:
        for row, gain_row in enumerate(gain_matrix):
            value = abs(sum(gain_row[col] * corner[col] for col in range(len(corner))))
            worst[row] = max(worst[row], value)
    return worst
