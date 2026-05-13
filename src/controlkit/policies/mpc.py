"""MPC-lite policy frontend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from controlkit.compiler.ir import (
    DynamicsKind,
    IRModule,
    LinearSystemIR,
    Matrix,
    MpcControllerIR,
    Vector,
)
from controlkit.policies.base import PolicyKind, PolicySpec


class MpcSpecError(ValueError):
    """Raised when an MPC-lite specification is malformed."""


@dataclass(frozen=True)
class MpcControllerSpec(PolicySpec):
    a_matrix: tuple[tuple[float, ...], ...]
    b_matrix: tuple[tuple[float, ...], ...]
    state_dim: int
    control_dim: int
    horizon: int
    q_diagonal: tuple[float, ...]
    r_diagonal: tuple[float, ...]
    q_terminal_diagonal: tuple[float, ...]
    u_min: tuple[float, ...]
    u_max: tuple[float, ...]
    solver_iterations: int
    step_size: float
    state_name: str = "x"
    control_name: str = "u"

    def __post_init__(self) -> None:
        _validate_name(self.name, "controller name")
        _validate_name(self.state_name, "state vector name")
        _validate_name(self.control_name, "control vector name")
        _validate_positive_int(self.state_dim, "state_dim")
        _validate_positive_int(self.control_dim, "control_dim")
        _validate_positive_int(self.horizon, "horizon")
        _validate_positive_int(self.solver_iterations, "solver_iterations")
        if self.step_size <= 0.0:
            raise MpcSpecError("step_size must be positive")
        _validate_matrix_shape(self.a_matrix, self.state_dim, self.state_dim, "a_matrix")
        _validate_matrix_shape(self.b_matrix, self.state_dim, self.control_dim, "b_matrix")
        _validate_vector_shape(self.q_diagonal, self.state_dim, "q_diagonal")
        _validate_vector_shape(self.r_diagonal, self.control_dim, "r_diagonal")
        _validate_vector_shape(self.q_terminal_diagonal, self.state_dim, "q_terminal_diagonal")
        _validate_vector_shape(self.u_min, self.control_dim, "u_min")
        _validate_vector_shape(self.u_max, self.control_dim, "u_max")
        for lower, upper in zip(self.u_min, self.u_max, strict=True):
            if lower > upper:
                raise MpcSpecError("u_min entries must be <= u_max entries")


@dataclass(frozen=True)
class MpcPolicy:
    kind: PolicyKind = PolicyKind.MPC

    def load(self, spec_path: Path) -> PolicySpec:
        return PolicySpec(name=spec_path.stem, source_path=spec_path)

    def lower(self, spec: PolicySpec) -> IRModule:
        if not isinstance(spec, MpcControllerSpec):
            return IRModule(name=spec.name, policy=self.kind)

        state = Vector(spec.state_name, dim=spec.state_dim)
        control = Vector(spec.control_name, dim=spec.control_dim)
        a_matrix = Matrix("A", rows=spec.state_dim, cols=spec.state_dim, values=spec.a_matrix)
        b_matrix = Matrix("B", rows=spec.state_dim, cols=spec.control_dim, values=spec.b_matrix)
        system = LinearSystemIR(
            state=state,
            control=control,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            dynamics=DynamicsKind.DISCRETE,
        )
        controller = MpcControllerIR(
            name=spec.name,
            state=state,
            control=control,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            horizon=spec.horizon,
            q_diagonal=spec.q_diagonal,
            r_diagonal=spec.r_diagonal,
            q_terminal_diagonal=spec.q_terminal_diagonal,
            u_min=spec.u_min,
            u_max=spec.u_max,
            solver_iterations=spec.solver_iterations,
            step_size=spec.step_size,
        )
        return IRModule(
            name=spec.name,
            policy=self.kind,
            metadata={
                "frontend": "mpc",
                "state_name": spec.state_name,
                "control_name": spec.control_name,
                "state_dim": str(spec.state_dim),
                "control_dim": str(spec.control_dim),
                "horizon": str(spec.horizon),
                "solver_iterations": str(spec.solver_iterations),
                "step_size": f"{spec.step_size:g}",
            },
            systems=(system,),
            mpc_controllers=(controller,),
        )

    def from_matrices(
        self,
        *,
        name: str,
        a_matrix: Sequence[Sequence[float]],
        b_matrix: Sequence[Sequence[float]],
        horizon: int,
        q_diagonal: Sequence[float],
        r_diagonal: Sequence[float],
        u_min: Sequence[float],
        u_max: Sequence[float],
        solver_iterations: int,
        step_size: float,
        q_terminal_diagonal: Sequence[float] | None = None,
        state_dim: int | None = None,
        control_dim: int | None = None,
        state_name: str = "x",
        control_name: str = "u",
    ) -> MpcControllerSpec:
        a_values = _normalize_matrix(a_matrix, "a_matrix")
        b_values = _normalize_matrix(b_matrix, "b_matrix")
        inferred_state_dim = len(a_values)
        inferred_control_dim = len(b_values[0])
        resolved_state_dim = inferred_state_dim if state_dim is None else state_dim
        resolved_control_dim = inferred_control_dim if control_dim is None else control_dim
        q_values = _normalize_vector(q_diagonal, "q_diagonal")
        r_values = _normalize_vector(r_diagonal, "r_diagonal")
        q_terminal_values = (
            q_values
            if q_terminal_diagonal is None
            else _normalize_vector(q_terminal_diagonal, "q_terminal_diagonal")
        )

        return MpcControllerSpec(
            name=name,
            source_path=Path("<mpc-api>"),
            a_matrix=a_values,
            b_matrix=b_values,
            state_dim=resolved_state_dim,
            control_dim=resolved_control_dim,
            horizon=horizon,
            q_diagonal=q_values,
            r_diagonal=r_values,
            q_terminal_diagonal=q_terminal_values,
            u_min=_normalize_vector(u_min, "u_min"),
            u_max=_normalize_vector(u_max, "u_max"),
            solver_iterations=solver_iterations,
            step_size=float(step_size),
            state_name=state_name,
            control_name=control_name,
        )


def _normalize_matrix(
    values: Sequence[Sequence[float]],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a 2D numeric sequence")
    rows: list[tuple[float, ...]] = []
    for row in values:
        if isinstance(row, str):
            raise TypeError(f"{name} rows must be numeric sequences")
        rows.append(tuple(float(value) for value in row))
    if not rows:
        raise MpcSpecError(f"{name} must have at least one row")
    if not rows[0]:
        raise MpcSpecError(f"{name} must have at least one column")
    expected_cols = len(rows[0])
    if any(len(row) != expected_cols for row in rows):
        raise MpcSpecError(f"{name} rows must all have the same length")
    return tuple(rows)


def _normalize_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a numeric sequence")
    result = tuple(float(value) for value in values)
    if not result:
        raise MpcSpecError(f"{name} must have at least one entry")
    return result


def _validate_matrix_shape(
    values: tuple[tuple[float, ...], ...],
    rows: int,
    cols: int,
    name: str,
) -> None:
    actual = (len(values), len(values[0]))
    expected = (rows, cols)
    if actual != expected:
        raise MpcSpecError(f"{name} shape must be {expected}, got {actual}")


def _validate_vector_shape(values: tuple[float, ...], dim: int, name: str) -> None:
    if len(values) != dim:
        raise MpcSpecError(f"{name} must contain exactly {dim} entries")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise MpcSpecError(f"{name} must be positive")


def _validate_name(value: str, name: str) -> None:
    if not value:
        raise MpcSpecError(f"{name} must be non-empty")
