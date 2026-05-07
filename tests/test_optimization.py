from __future__ import annotations

from controlkit.backends.c import CBackend
from controlkit.backends.rust import RustBackend
from controlkit.compiler.ir import (
    ControlLaw,
    IRModule,
    Matrix,
    ScalarConstant,
    Shape,
    Vector,
    Zero,
    add,
    matvec,
    neg,
    scalar_mul,
    sub,
    zero,
)
from controlkit.optimization import estimate_operation_count, optimize_module, simplify_expr
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy


def test_simplify_constant_folding_and_double_negative() -> None:
    folded, rewrites = simplify_expr(add(ScalarConstant(2.0), neg(ScalarConstant(-3.0))))

    assert folded == ScalarConstant(5.0)
    assert rewrites == 2


def test_simplify_x_plus_zero_and_x_minus_zero() -> None:
    x = Vector("x", dim=2)

    simplified_add, add_rewrites = simplify_expr(add(x, zero(x.shape)))
    simplified_sub, sub_rewrites = simplify_expr(sub(x, zero(x.shape)))

    assert simplified_add == x
    assert add_rewrites == 1
    assert simplified_sub == x
    assert sub_rewrites == 1


def test_simplify_scalar_multiply_identities() -> None:
    x = Vector("x", dim=3)

    zeroed, zero_rewrites = simplify_expr(scalar_mul(0.0, x))
    unchanged, one_rewrites = simplify_expr(scalar_mul(1.0, x))
    folded, fold_rewrites = simplify_expr(scalar_mul(2.0, ScalarConstant(4.0)))

    assert zeroed == Zero(Shape.vector(3))
    assert zero_rewrites == 1
    assert unchanged == x
    assert one_rewrites == 1
    assert folded == ScalarConstant(8.0)
    assert fold_rewrites == 1


def test_optimize_module_reports_operation_count_reduction() -> None:
    x = Vector("x", dim=2)
    u = Vector("u", dim=2)
    expression = add(scalar_mul(1.0, x), zero(x.shape))
    module = IRModule(
        name="identity",
        policy=PolicyKind.LQR,
        control_laws=(ControlLaw(output=u, expression=expression),),
    )

    result = optimize_module(module)

    assert result.module.control_laws[0].expression == x
    assert result.report.rewrites == 2
    assert result.report.operations_before > result.report.operations_after
    assert estimate_operation_count(result.module) == 0


def test_matvec_zero_vector_simplifies_to_zero() -> None:
    matrix = Matrix("K", rows=1, cols=2, values=((1.0, 2.0),))
    expr = matvec(matrix, Zero(Shape.vector(2)))

    simplified, rewrites = simplify_expr(expr)

    assert simplified == Zero(Shape.vector(1))
    assert rewrites == 1


def test_c_backend_can_unroll_small_controller_loops() -> None:
    spec = LqrPolicy().from_gain_matrix(name="small_lqr", gain_matrix=[[1.0, 2.0]])
    module = LqrPolicy().lower(spec)

    source = CBackend(unroll_loops=True).generate(module).source

    assert "for (size_t row" not in source
    assert "for (size_t i" not in source
    assert "tmp0[0u] = K[0u][0u] * x[0u] + K[0u][1u] * x[1u];" in source
    assert "u[0u] = tmp1[0u];" in source


def test_rust_backend_can_unroll_small_controller_loops() -> None:
    spec = LqrPolicy().from_gain_matrix(name="small_lqr", gain_matrix=[[1.0, 2.0]])
    module = LqrPolicy().lower(spec)

    source = RustBackend(unroll_loops=True).generate(module).source

    assert "while row <" not in source
    assert "while i <" not in source
    assert "tmp0[0] = K[0][0] * x[0] + K[0][1] * x[1];" in source
    assert "u[0] = tmp1[0];" in source


def test_backends_can_emit_optimized_zero_control_law_using_module_metadata() -> None:
    frontend = LqrPolicy()
    module = frontend.lower(frontend.from_gain_matrix(name="zero_lqr", gain_matrix=[[1.0, 2.0]]))
    original_law = module.control_laws[0]
    zeroable_module = IRModule(
        name=module.name,
        policy=module.policy,
        metadata=module.metadata,
        control_laws=(
            ControlLaw(
                output=original_law.output,
                expression=scalar_mul(0.0, original_law.expression),
            ),
        ),
    )
    optimized = optimize_module(zeroable_module).module

    c_artifact = CBackend().generate(optimized)
    rust_artifact = RustBackend().generate(optimized)

    assert "#define CONTROLKIT_STATE_DIM 2u" in c_artifact.header
    assert "float tmp0[1u] = {0.0f};" in c_artifact.source
    assert "pub const STATE_DIM: usize = 2;" in rust_artifact.source
    assert "let tmp0 = [0.0_f32; 1];" in rust_artifact.source
