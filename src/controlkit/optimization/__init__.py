"""IR optimization passes."""

from __future__ import annotations

from controlkit.optimization.passes import (
    OptimizationReport,
    OptimizationResult,
    SimplifyPass,
    estimate_operation_count,
    optimize_module,
    simplify_expr,
)

__all__ = [
    "OptimizationReport",
    "OptimizationResult",
    "SimplifyPass",
    "estimate_operation_count",
    "optimize_module",
    "simplify_expr",
]

