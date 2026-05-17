# Optimization passes

from __future__ import annotations

from dataclasses import dataclass

from controlkit.compiler.ir import (
    Add,
    Clip,
    ControlLaw,
    Expr,
    IRModule,
    ActivationLayerIR,
    LinearLayerIR,
    MatVecMul,
    MpcControllerIR,
    Neg,
    ScalarConstant,
    ScalarMul,
    Sub,
    Zero,
)


@dataclass(frozen=True)
class OptimizationReport:
    rewrites: int
    operations_before: int
    operations_after: int


@dataclass(frozen=True)
class OptimizationResult:
    module: IRModule
    report: OptimizationReport


class SimplifyPass:
    def run(self, module: IRModule) -> OptimizationResult:
        operations_before = estimate_operation_count(module)
        rewrites = 0
        optimized_laws: list[ControlLaw] = []
        for law in module.control_laws:
            optimized_expr, expr_rewrites = simplify_expr(law.expression)
            rewrites += expr_rewrites
            optimized_laws.append(ControlLaw(output=law.output, expression=optimized_expr))

        optimized_module = IRModule(
            name=module.name,
            policy=module.policy,
            metadata=module.metadata,
            systems=module.systems,
            control_laws=tuple(optimized_laws),
            mpc_controllers=module.mpc_controllers,
            rl_policies=module.rl_policies,
        )
        operations_after = estimate_operation_count(optimized_module)
        return OptimizationResult(
            module=optimized_module,
            report=OptimizationReport(
                rewrites=rewrites,
                operations_before=operations_before,
                operations_after=operations_after,
            ),
        )


def optimize_module(module: IRModule) -> OptimizationResult:
    return SimplifyPass().run(module)


def simplify_expr(expr: Expr) -> tuple[Expr, int]:
    if isinstance(expr, ScalarConstant | Zero):
        return expr, 0

    if isinstance(expr, Neg):
        value, rewrites = simplify_expr(expr.value)
        if isinstance(value, ScalarConstant):
            return ScalarConstant(-value.value), rewrites + 1
        if isinstance(value, Zero):
            return value, rewrites + 1
        if isinstance(value, Neg):
            return value.value, rewrites + 1
        return Neg(value), rewrites

    if isinstance(expr, ScalarMul):
        scalar, left_rewrites = simplify_expr(expr.scalar)
        value, right_rewrites = simplify_expr(expr.value)
        rewrites = left_rewrites + right_rewrites
        if isinstance(scalar, ScalarConstant):
            if _is_zero(scalar.value):
                return Zero(value.shape), rewrites + 1
            if _is_one(scalar.value):
                return value, rewrites + 1
            if isinstance(value, ScalarConstant):
                return ScalarConstant(scalar.value * value.value), rewrites + 1
            if isinstance(value, Zero):
                return value, rewrites + 1
        return ScalarMul(scalar, value), rewrites

    if isinstance(expr, Add):
        left, left_rewrites = simplify_expr(expr.left)
        right, right_rewrites = simplify_expr(expr.right)
        rewrites = left_rewrites + right_rewrites
        if isinstance(left, Zero):
            return right, rewrites + 1
        if isinstance(right, Zero):
            return left, rewrites + 1
        if isinstance(left, ScalarConstant) and isinstance(right, ScalarConstant):
            return ScalarConstant(left.value + right.value), rewrites + 1
        return Add(left, right), rewrites

    if isinstance(expr, Sub):
        left, left_rewrites = simplify_expr(expr.left)
        right, right_rewrites = simplify_expr(expr.right)
        rewrites = left_rewrites + right_rewrites
        if isinstance(right, Zero):
            return left, rewrites + 1
        if isinstance(left, Zero):
            return Neg(right), rewrites + 1
        if isinstance(left, ScalarConstant) and isinstance(right, ScalarConstant):
            return ScalarConstant(left.value - right.value), rewrites + 1
        return Sub(left, right), rewrites

    if isinstance(expr, MatVecMul):
        matrix, left_rewrites = simplify_expr(expr.matrix)
        vector, right_rewrites = simplify_expr(expr.vector)
        rewrites = left_rewrites + right_rewrites
        if isinstance(matrix, Zero):
            return Zero(expr.shape), rewrites + 1
        if isinstance(vector, Zero):
            return Zero(expr.shape), rewrites + 1
        return MatVecMul(matrix, vector), rewrites

    if isinstance(expr, Clip):
        value, value_rewrites = simplify_expr(expr.value)
        lower, lower_rewrites = simplify_expr(expr.lower)
        upper, upper_rewrites = simplify_expr(expr.upper)
        rewrites = value_rewrites + lower_rewrites + upper_rewrites
        if (
            isinstance(value, ScalarConstant)
            and isinstance(lower, ScalarConstant)
            and isinstance(upper, ScalarConstant)
        ):
            return ScalarConstant(min(max(value.value, lower.value), upper.value)), rewrites + 1
        return Clip(value, lower, upper), rewrites

    return expr, 0


def estimate_operation_count(module: IRModule) -> int:
    total = 0
    for law in module.control_laws:
        total += _estimate_expr_operations(law.expression)
    for controller in module.mpc_controllers:
        total += _estimate_mpc_operations(controller)
    for policy in module.rl_policies:
        total += _estimate_rl_operations(policy.layers)
    return total


def _estimate_rl_operations(layers: tuple[LinearLayerIR | ActivationLayerIR, ...]) -> int:
    total = 0
    for layer in layers:
        if isinstance(layer, LinearLayerIR):
            total += layer.input_dim * layer.output_dim * 2
            total += layer.output_dim
        else:
            total += layer.output_dim
    return total


def _estimate_mpc_operations(controller: MpcControllerIR) -> int:
    n = controller.state.dim
    m = controller.control.dim
    h = controller.horizon
    rollout_per_step = (n * n * 2) + (n * m * 2)
    terminal_costate = n
    grad_per_step = m + (m * n * 2)
    costate_per_step = n + (n * n * 2)
    projected_update_per_step = m * 4
    return controller.solver_iterations * (
        terminal_costate
        + h * (rollout_per_step + grad_per_step + costate_per_step + projected_update_per_step)
    )


def _estimate_expr_operations(expr: Expr) -> int:
    if isinstance(expr, ScalarConstant | Zero):
        return 0
    if isinstance(expr, Neg):
        return _estimate_expr_operations(expr.value) + expr.shape.rows
    if isinstance(expr, ScalarMul):
        return _estimate_expr_operations(expr.value) + expr.shape.rows * expr.shape.cols
    if isinstance(expr, Add | Sub):
        return (
            _estimate_expr_operations(expr.left)
            + _estimate_expr_operations(expr.right)
            + expr.shape.rows * expr.shape.cols
        )
    if isinstance(expr, MatVecMul):
        return (
            _estimate_expr_operations(expr.matrix)
            + _estimate_expr_operations(expr.vector)
            + expr.matrix.shape.rows * expr.matrix.shape.cols * 2
        )
    if isinstance(expr, Clip):
        return _estimate_expr_operations(expr.value) + expr.shape.rows * 2
    return 0


def _is_zero(value: float) -> bool:
    return value == 0.0


def _is_one(value: float) -> bool:
    return value == 1.0
