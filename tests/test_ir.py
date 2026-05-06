from __future__ import annotations

import pytest

from controlkit.compiler.ir import (
    Add,
    Clip,
    ControlLaw,
    DynamicsKind,
    IRValidationError,
    LinearSystemIR,
    Matrix,
    MatVecMul,
    ScalarConstant,
    Shape,
    Sub,
    ValueKind,
    Vector,
    add,
    clip,
    matvec,
    neg,
    sub,
)


def test_shapes_have_readable_repr_and_validation() -> None:
    assert repr(Shape.scalar()) == "scalar"
    assert repr(Shape.vector(3)) == "vector[3]"
    assert repr(Shape.matrix(2, 3)) == "matrix[2x3]"

    with pytest.raises(IRValidationError, match="positive"):
        Shape.vector(0)
    with pytest.raises(IRValidationError, match="scalar"):
        Shape(ValueKind.SCALAR, rows=2, cols=1)


def test_matrix_vector_multiplication_validates_shape() -> None:
    k = Matrix("K", rows=2, cols=4)
    x = Vector("x", dim=4)

    expr = matvec(k, x)

    assert isinstance(expr, MatVecMul)
    assert expr.shape == Shape.vector(2)
    assert repr(expr) == "(K:matrix[2x4] @ x:vector[4])"
    with pytest.raises(IRValidationError, match="shape mismatch"):
        matvec(Matrix("Bad", rows=2, cols=3), x)


def test_addition_and_subtraction_require_identical_shapes() -> None:
    x = Vector("x", dim=2)
    y = Vector("y", dim=2)
    z = Vector("z", dim=3)

    assert isinstance(add(x, y), Add)
    assert add(x, y).shape == Shape.vector(2)
    assert isinstance(sub(x, y), Sub)

    with pytest.raises(IRValidationError, match="identical shapes"):
        add(x, z)
    with pytest.raises(TypeError, match="IR expression"):
        add(x, object())  # type: ignore[arg-type]


def test_clip_accepts_scalar_or_matching_vector_bounds() -> None:
    u = Vector("u", dim=2)

    scalar_clip = clip(u, -1.0, 1.0)
    vector_clip = Clip(u, Vector("lo", dim=2), Vector("hi", dim=2))

    assert scalar_clip.shape == Shape.vector(2)
    assert isinstance(scalar_clip.lower, ScalarConstant)
    assert vector_clip.shape == Shape.vector(2)
    assert repr(scalar_clip) == "clip(u:vector[2], const(-1), const(1))"

    with pytest.raises(IRValidationError, match="bound must be scalar or match"):
        Clip(u, Vector("lo_bad", dim=3), Vector("hi", dim=2))


def test_control_law_supports_negative_feedback() -> None:
    k = Matrix("K", rows=1, cols=4)
    x = Vector("x", dim=4)
    u = Vector("u", dim=1)

    law = ControlLaw(output=u, expression=neg(matvec(k, x)))
    assert law.expression.shape == u.shape
    assert repr(law) == "u = (-(K:matrix[1x4] @ x:vector[4]))"

    with pytest.raises(IRValidationError, match="output shape"):
        ControlLaw(output=Vector("u_bad", dim=2), expression=neg(matvec(k, x)))


def test_linear_system_validates_continuous_and_discrete_dynamics() -> None:
    x = Vector("x", dim=2)
    u = Vector("u", dim=1)
    a = Matrix("A", rows=2, cols=2)
    b = Matrix("B", rows=2, cols=1)

    continuous = LinearSystemIR(x, u, a, b, DynamicsKind.CONTINUOUS)
    discrete = LinearSystemIR(x, u, a, b, DynamicsKind.DISCRETE)
    assert continuous.expression.shape == Shape.vector(2)
    assert repr(continuous) == (
        "x_dot = ((A:matrix[2x2] @ x:vector[2]) + (B:matrix[2x1] @ u:vector[1]))"
    )
    assert repr(discrete).startswith("x_next =")
    with pytest.raises(IRValidationError, match="A must be square"):
        LinearSystemIR(x, u, Matrix("A_bad", rows=2, cols=3), b, DynamicsKind.CONTINUOUS)
    with pytest.raises(IRValidationError, match="B must have shape"):
        LinearSystemIR(x, u, a, Matrix("B_bad", rows=3, cols=1), DynamicsKind.CONTINUOUS)

