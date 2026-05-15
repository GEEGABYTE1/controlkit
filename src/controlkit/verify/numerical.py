"""Numerical robustness checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class NumericalResult:
    passed: bool
    warnings: list[str]
    errors: list[str]
    condition_numbers: dict[str, float]


def check_numerical_robustness(
    matrices: dict[str, list[list[float]]],
    *,
    high_condition_threshold: float = 1e8,
) -> NumericalResult:
    warnings: list[str] = []
    errors: list[str] = []
    condition_numbers: dict[str, float] = {}
    for name, matrix in matrices.items():
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                if not math.isfinite(value):
                    errors.append(f"{name}[{row_index},{col_index}] is not finite")
        if matrix and len(matrix) == len(matrix[0]):
            condition = condition_number(matrix)
            condition_numbers[name] = condition
            if math.isfinite(condition) and condition > high_condition_threshold:
                warnings.append(f"{name} condition number is high: {condition:.6g}")
            if math.isinf(condition):
                warnings.append(f"{name} appears singular")
    return NumericalResult(
        passed=not errors,
        warnings=warnings,
        errors=errors,
        condition_numbers=condition_numbers,
    )


def condition_number(matrix: list[list[float]]) -> float:
    inv = invert_matrix(matrix)
    if inv is None:
        return math.inf
    return matrix_one_norm(matrix) * matrix_one_norm(inv)


def matrix_one_norm(matrix: list[list[float]]) -> float:
    if not matrix:
        return 0.0
    return max(sum(abs(matrix[row][col]) for row in range(len(matrix))) for col in range(len(matrix[0])))


def invert_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    n = len(matrix)
    augmented = [
        [float(matrix[row][col]) for col in range(n)]
        + [1.0 if row == col else 0.0 for col in range(n)]
        for row in range(n)
    ]
    for pivot_index in range(n):
        pivot_row = max(range(pivot_index, n), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return None
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row in range(n):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            augmented[row] = [
                augmented[row][col] - factor * augmented[pivot_index][col] for col in range(2 * n)
            ]
    return [row[n:] for row in augmented]


def compare_generated_outputs(
    reference_controller: Callable[[list[float]], list[float]],
    generated_controller: Callable[[list[float]], list[float]] | None,
    states: Iterable[list[float]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare reference and generated outputs.

    Pass ``generated_controller=None`` when generated-code execution is not wired in. This keeps
    the API available as an extension point without pretending execution exists.
    """

    if generated_controller is None:
        return {
            "available": False,
            "passed": False,
            "message": "generated-code execution hook not configured",
        }
    max_error = 0.0
    for state in states:
        reference = reference_controller(state)
        generated = generated_controller(state)
        for left, right in zip(reference, generated, strict=True):
            max_error = max(max_error, abs(left - right))
    return {
        "available": True,
        "passed": max_error <= tolerance,
        "max_error": max_error,
        "tolerance": tolerance,
    }
