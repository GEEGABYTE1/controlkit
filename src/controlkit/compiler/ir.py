"""Backend-neutral intermediate representation for control laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from controlkit.policies.base import PolicyKind


class IRValidationError(ValueError):
    """Raised when an IR node is malformed or shape-incompatible."""


class ValueKind(StrEnum):
    """Kinds of values expressible in ControlKit IR."""

    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass(frozen=True)
class Shape:
    """Static shape for an IR value."""

    kind: ValueKind
    rows: int = 1
    cols: int = 1

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise IRValidationError("shape dimensions must be positive")
        if self.kind is ValueKind.SCALAR and (self.rows, self.cols) != (1, 1):
            raise IRValidationError("scalar shape must be 1x1")
        if self.kind is ValueKind.VECTOR and self.cols != 1:
            raise IRValidationError("vector shape must have exactly one column")

    @classmethod
    def scalar(cls) -> Shape:
        return cls(ValueKind.SCALAR)

    @classmethod
    def vector(cls, dim: int) -> Shape:
        return cls(ValueKind.VECTOR, rows=dim, cols=1)

    @classmethod
    def matrix(cls, rows: int, cols: int) -> Shape:
        return cls(ValueKind.MATRIX, rows=rows, cols=cols)

    @property
    def dim(self) -> int:
        if self.kind is not ValueKind.VECTOR:
            raise IRValidationError("only vector shapes have a dimension")
        return self.rows

    def __repr__(self) -> str:
        if self.kind is ValueKind.SCALAR:
            return "scalar"
        if self.kind is ValueKind.VECTOR:
            return f"vector[{self.rows}]"
        return f"matrix[{self.rows}x{self.cols}]"


class Expr:
    """Base class for typed IR expressions."""

    @property
    def shape(self) -> Shape:
        raise NotImplementedError


@dataclass(frozen=True)
class ScalarConstant(Expr):
    """Scalar numeric constant."""

    value: float

    @property
    def shape(self) -> Shape:
        return Shape.scalar()

    def __repr__(self) -> str:
        return f"const({self.value:g})"


@dataclass(frozen=True)
class Zero(Expr):
    """Typed zero value."""

    value_shape: Shape

    @property
    def shape(self) -> Shape:
        return self.value_shape

    def __repr__(self) -> str:
        return f"zero({self.shape!r})"


@dataclass(frozen=True)
class Vector(Expr):
    """Named vector value."""

    name: str
    dim: int

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.dim <= 0:
            raise IRValidationError("vector dimension must be positive")

    @property
    def shape(self) -> Shape:
        return Shape.vector(self.dim)

    def __repr__(self) -> str:
        return f"{self.name}:vector[{self.dim}]"


@dataclass(frozen=True)
class Matrix(Expr):
    """Named matrix value."""

    name: str
    rows: int
    cols: int
    values: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.rows <= 0 or self.cols <= 0:
            raise IRValidationError("matrix dimensions must be positive")
        if self.values is not None:
            _validate_matrix_values(self.values, self.rows, self.cols)

    @property
    def shape(self) -> Shape:
        return Shape.matrix(self.rows, self.cols)

    def __repr__(self) -> str:
        return f"{self.name}:matrix[{self.rows}x{self.cols}]"


@dataclass(frozen=True)
class Neg(Expr):
    """Unary negation."""

    value: Expr

    def __post_init__(self) -> None:
        _require_expr(self.value, "value")

    @property
    def shape(self) -> Shape:
        return self.value.shape

    def __repr__(self) -> str:
        return f"(-{self.value!r})"


@dataclass(frozen=True)
class ScalarMul(Expr):
    """Scalar multiplication of a scalar, vector, or matrix expression."""

    scalar: Expr
    value: Expr

    def __post_init__(self) -> None:
        _require_expr(self.scalar, "scalar")
        _require_expr(self.value, "value")
        if self.scalar.shape.kind is not ValueKind.SCALAR:
            raise IRValidationError("scalar multiplication requires a scalar left operand")

    @property
    def shape(self) -> Shape:
        return self.value.shape

    def __repr__(self) -> str:
        return f"({self.scalar!r} * {self.value!r})"


@dataclass(frozen=True)
class MatVecMul(Expr):
    """Matrix-vector multiplication."""

    matrix: Expr
    vector: Expr

    def __post_init__(self) -> None:
        _require_expr(self.matrix, "matrix")
        _require_expr(self.vector, "vector")
        if self.matrix.shape.kind is not ValueKind.MATRIX:
            raise IRValidationError("matrix-vector multiplication requires a matrix left operand")
        if self.vector.shape.kind is not ValueKind.VECTOR:
            raise IRValidationError("matrix-vector multiplication requires a vector right operand")
        if self.matrix.shape.cols != self.vector.shape.rows:
            raise IRValidationError(
                "matrix-vector shape mismatch: "
                f"{self.matrix.shape!r} cannot multiply {self.vector.shape!r}"
            )

    @property
    def shape(self) -> Shape:
        return Shape.vector(self.matrix.shape.rows)

    def __repr__(self) -> str:
        return f"({self.matrix!r} @ {self.vector!r})"


@dataclass(frozen=True)
class Add(Expr):
    """Elementwise addition."""

    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        _validate_same_shape(self.left, self.right, "addition")

    @property
    def shape(self) -> Shape:
        return self.left.shape

    def __repr__(self) -> str:
        return f"({self.left!r} + {self.right!r})"


@dataclass(frozen=True)
class Sub(Expr):
    """Elementwise subtraction."""

    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        _validate_same_shape(self.left, self.right, "subtraction")

    @property
    def shape(self) -> Shape:
        return self.left.shape

    def __repr__(self) -> str:
        return f"({self.left!r} - {self.right!r})"


@dataclass(frozen=True)
class Clip(Expr):
    """Elementwise clipping/saturation."""

    value: Expr
    lower: Expr
    upper: Expr

    def __post_init__(self) -> None:
        _require_expr(self.value, "value")
        _validate_bound(self.value, self.lower, "lower")
        _validate_bound(self.value, self.upper, "upper")

    @property
    def shape(self) -> Shape:
        return self.value.shape

    def __repr__(self) -> str:
        return f"clip({self.value!r}, {self.lower!r}, {self.upper!r})"


@dataclass(frozen=True)
class ControlLaw:
    """Named control law such as `u = -Kx`."""

    output: Vector
    expression: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.output, Vector):
            raise IRValidationError("control law output must be a vector")
        _require_expr(self.expression, "expression")
        if self.expression.shape != self.output.shape:
            raise IRValidationError(
                f"control law output shape {self.output.shape!r} does not match "
                f"expression shape {self.expression.shape!r}"
            )

    def __repr__(self) -> str:
        return f"{self.output.name} = {self.expression!r}"


class DynamicsKind(StrEnum):
    """Supported linear-system dynamics forms."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass(frozen=True)
class LinearSystemIR:
    """Linear system `x_dot = Ax + Bu` or `x_next = Ax + Bu`."""

    state: Vector
    control: Vector
    a_matrix: Matrix
    b_matrix: Matrix
    dynamics: DynamicsKind

    def __post_init__(self) -> None:
        if not isinstance(self.state, Vector):
            raise IRValidationError("linear system state must be a vector")
        if not isinstance(self.control, Vector):
            raise IRValidationError("linear system control must be a vector")
        if not isinstance(self.a_matrix, Matrix):
            raise IRValidationError("linear system A must be a matrix")
        if not isinstance(self.b_matrix, Matrix):
            raise IRValidationError("linear system B must be a matrix")

        state_dim = self.state.dim
        control_dim = self.control.dim
        if (self.a_matrix.rows, self.a_matrix.cols) != (state_dim, state_dim):
            raise IRValidationError(
                f"A must be square with shape matrix[{state_dim}x{state_dim}], "
                f"got {self.a_matrix.shape!r}"
            )
        if (self.b_matrix.rows, self.b_matrix.cols) != (state_dim, control_dim):
            raise IRValidationError(
                f"B must have shape matrix[{state_dim}x{control_dim}], "
                f"got {self.b_matrix.shape!r}"
            )

    @property
    def expression(self) -> Add:
        return Add(MatVecMul(self.a_matrix, self.state), MatVecMul(self.b_matrix, self.control))

    @property
    def lhs_name(self) -> str:
        return "x_dot" if self.dynamics is DynamicsKind.CONTINUOUS else "x_next"

    def __repr__(self) -> str:
        return f"{self.lhs_name} = {self.expression!r}"


@dataclass(frozen=True)
class IRModule:
    """A policy lowered into ControlKit's backend-neutral representation."""

    name: str
    policy: PolicyKind
    metadata: Mapping[str, str] = field(default_factory=dict)
    systems: tuple[LinearSystemIR, ...] = field(default_factory=tuple)
    control_laws: tuple[ControlLaw, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        policy_name = getattr(self.policy, "value", str(self.policy))
        return (
            f"IRModule(name={self.name!r}, policy={policy_name!r}, "
            f"systems={len(self.systems)}, control_laws={len(self.control_laws)})"
        )


def matvec(matrix: Expr, vector: Expr) -> MatVecMul:
    """Create a validated matrix-vector multiplication node."""

    return MatVecMul(matrix=matrix, vector=vector)


def add(left: Expr, right: Expr) -> Add:
    """Create a validated addition node."""

    return Add(left=left, right=right)


def sub(left: Expr, right: Expr) -> Sub:
    """Create a validated subtraction node."""

    return Sub(left=left, right=right)


def neg(value: Expr) -> Neg:
    """Create a validated negation node."""

    return Neg(value=value)


def scalar_mul(scalar: Expr | float, value: Expr) -> ScalarMul:
    """Create a validated scalar multiplication node."""

    return ScalarMul(scalar=_coerce_bound(scalar), value=value)


def zero(shape: Shape) -> Zero:
    """Create a typed zero expression."""

    return Zero(value_shape=shape)


def clip(value: Expr, lower: Expr | float, upper: Expr | float) -> Clip:
    """Create a validated clipping/saturation node."""

    return Clip(value=value, lower=_coerce_bound(lower), upper=_coerce_bound(upper))


def _coerce_bound(value: Expr | float) -> Expr:
    if isinstance(value, int | float):
        return ScalarConstant(float(value))
    _require_expr(value, "bound")
    return value


def _require_expr(value: object, name: str) -> None:
    if not isinstance(value, Expr):
        raise TypeError(f"{name} must be an IR expression")


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("IR names must be strings")
    if not name:
        raise IRValidationError("IR names must be non-empty")


def _validate_matrix_values(
    values: tuple[tuple[float, ...], ...],
    rows: int,
    cols: int,
) -> None:
    if len(values) != rows:
        raise IRValidationError(f"matrix values must have {rows} rows")
    for row in values:
        if len(row) != cols:
            raise IRValidationError(f"matrix value rows must have {cols} columns")


def _validate_same_shape(left: Expr, right: Expr, operation: str) -> None:
    _require_expr(left, "left")
    _require_expr(right, "right")
    if left.shape != right.shape:
        raise IRValidationError(
            f"{operation} requires identical shapes, got {left.shape!r} and {right.shape!r}"
        )


def _validate_bound(value: Expr, bound: Expr, name: str) -> None:
    _require_expr(bound, name)
    if bound.shape.kind is ValueKind.SCALAR:
        return
    if bound.shape != value.shape:
        raise IRValidationError(
            f"{name} bound must be scalar or match value shape, "
            f"got {bound.shape!r} for {value.shape!r}"
        )
