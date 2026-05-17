# LQR policy frontend.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from controlkit.compiler.ir import ControlLaw, Expr, IRModule, Matrix, Vector, clip, matvec, neg
from controlkit.policies.base import PolicyKind, PolicySpec


class LqrSpecError(ValueError):
    # Raised when an LQR specification is malformed.
    pass 


@dataclass(frozen=True)
class LqrSaturation:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise LqrSpecError("saturation lower bound must be <= upper bound")


@dataclass(frozen=True)
class LqrControllerSpec(PolicySpec):
    gain_matrix: tuple[tuple[float, ...], ...]
    state_dim: int
    control_dim: int
    saturation: LqrSaturation | None = None
    state_name: str = "x"
    control_name: str = "u"
    gain_name: str = "K"
    state_names: tuple[str, ...] = ()
    control_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_name(self.name, "controller name")
        _validate_name(self.state_name, "state vector name")
        _validate_name(self.control_name, "control vector name")
        _validate_name(self.gain_name, "gain matrix name")
        _validate_positive_dim(self.state_dim, "state_dim")
        _validate_positive_dim(self.control_dim, "control_dim")
        _validate_gain_shape(self.gain_matrix, self.control_dim, self.state_dim)
        _validate_component_names(self.state_names, self.state_dim, "state_names")
        _validate_component_names(self.control_names, self.control_dim, "control_names")


@dataclass(frozen=True)
class LqrPolicy:
    kind: PolicyKind = PolicyKind.LQR

    def load(self, spec_path: Path) -> PolicySpec:
        return PolicySpec(name=spec_path.stem, source_path=spec_path)

    def lower(self, spec: PolicySpec) -> IRModule:
        if not isinstance(spec, LqrControllerSpec):
            return IRModule(name=spec.name, policy=self.kind)

        state = Vector(spec.state_name, dim=spec.state_dim)
        control = Vector(spec.control_name, dim=spec.control_dim)
        gain = Matrix(
            spec.gain_name,
            rows=spec.control_dim,
            cols=spec.state_dim,
            values=spec.gain_matrix,
        )
        expression: Expr = neg(matvec(gain, state))

        if spec.saturation is not None:
            expression = clip(expression, lower=spec.saturation.lower, upper=spec.saturation.upper)

        metadata = {
            "frontend": "lqr",
            "gain_name": spec.gain_name,
            "state_name": spec.state_name,
            "control_name": spec.control_name,
            "state_dim": str(spec.state_dim),
            "control_dim": str(spec.control_dim),
        }
        if spec.state_names:
            metadata["state_names"] = ",".join(spec.state_names)
        if spec.control_names:
            metadata["control_names"] = ",".join(spec.control_names)
        if spec.saturation is not None:
            metadata["saturation"] = f"{spec.saturation.lower:g},{spec.saturation.upper:g}"

        return IRModule(
            name=spec.name,
            policy=self.kind,
            metadata=metadata,
            control_laws=(ControlLaw(output=control, expression=expression),),
        )

    def from_gain_matrix(
        self,
        *,
        name: str,
        gain_matrix: Sequence[Sequence[float]],
        state_dim: int | None = None,
        control_dim: int | None = None,
        saturation: LqrSaturation | tuple[float, float] | None = None,
        state_name: str = "x",
        control_name: str = "u",
        gain_name: str = "K",
        state_names: Sequence[str] = (),
        control_names: Sequence[str] = (),
    ) -> LqrControllerSpec:
        # Build a validated LQR spec from a numeric feedback gain matrix.

        normalized_gain = _normalize_gain_matrix(gain_matrix)
        inferred_control_dim = len(normalized_gain)
        inferred_state_dim = len(normalized_gain[0])
        resolved_state_dim = inferred_state_dim if state_dim is None else state_dim
        resolved_control_dim = inferred_control_dim if control_dim is None else control_dim
        resolved_saturation = _normalize_saturation(saturation)

        return LqrControllerSpec(
            name=name,
            source_path=Path("<lqr-api>"),
            gain_matrix=normalized_gain,
            state_dim=resolved_state_dim,
            control_dim=resolved_control_dim,
            saturation=resolved_saturation,
            state_name=state_name,
            control_name=control_name,
            gain_name=gain_name,
            state_names=tuple(state_names),
            control_names=tuple(control_names),
        )


def _normalize_gain_matrix(gain_matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if isinstance(gain_matrix, str):
        raise TypeError("gain_matrix must be a 2D numeric sequence")
    rows: list[tuple[float, ...]] = []
    for row in gain_matrix:
        if isinstance(row, str):
            raise TypeError("gain_matrix rows must be numeric sequences")
        rows.append(tuple(float(value) for value in row))
    if not rows:
        raise LqrSpecError("gain_matrix must have at least one row")
    if not rows[0]:
        raise LqrSpecError("gain_matrix must have at least one column")
    expected_cols = len(rows[0])
    if any(len(row) != expected_cols for row in rows):
        raise LqrSpecError("gain_matrix rows must all have the same length")
    return tuple(rows)


def _normalize_saturation(
    saturation: LqrSaturation | tuple[float, float] | None,
) -> LqrSaturation | None:
    if saturation is None or isinstance(saturation, LqrSaturation):
        return saturation
    if len(saturation) != 2:
        raise LqrSpecError("saturation tuple must contain lower and upper bounds")
    lower, upper = saturation
    return LqrSaturation(lower=float(lower), upper=float(upper))


def _validate_gain_shape(
    gain_matrix: tuple[tuple[float, ...], ...],
    control_dim: int,
    state_dim: int,
) -> None:
    actual_shape = (len(gain_matrix), len(gain_matrix[0]))
    expected_shape = (control_dim, state_dim)
    if actual_shape != expected_shape:
        raise LqrSpecError(
            "gain_matrix shape must match control_dim x state_dim, "
            f"expected {expected_shape}, got {actual_shape}"
        )


def _validate_positive_dim(value: int, name: str) -> None:
    if value <= 0:
        raise LqrSpecError(f"{name} must be positive")


def _validate_name(value: str, name: str) -> None:
    if not value:
        raise LqrSpecError(f"{name} must be non-empty")


def _validate_component_names(names: tuple[str, ...], expected_dim: int, field_name: str) -> None:
    if names and len(names) != expected_dim:
        raise LqrSpecError(f"{field_name} must contain exactly {expected_dim} names")
    if any(not name for name in names):
        raise LqrSpecError(f"{field_name} entries must be non-empty")
