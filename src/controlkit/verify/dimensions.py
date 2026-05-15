"""Dimension checks for controller and system matrices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str


def matrix_shape(matrix: list[list[float]]) -> tuple[int, int]:
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def verify_dimensions(
    *,
    a_matrix: list[list[float]] | None,
    b_matrix: list[list[float]] | None,
    gain_matrix: list[list[float]] | None,
    q_matrix: list[list[float]] | None = None,
    r_matrix: list[list[float]] | None = None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    n = None
    m = None
    if a_matrix is not None:
        rows, cols = matrix_shape(a_matrix)
        n = rows
        checks.append(CheckResult("A square", rows > 0 and rows == cols, f"A shape is {rows}x{cols}"))
    if b_matrix is not None:
        rows, cols = matrix_shape(b_matrix)
        m = cols
        expected_rows = n if n is not None else rows
        checks.append(
            CheckResult(
                "B rows match A",
                rows > 0 and cols > 0 and rows == expected_rows,
                f"B shape is {rows}x{cols}",
            )
        )
    if gain_matrix is not None:
        rows, cols = matrix_shape(gain_matrix)
        expected_rows = m if m is not None else rows
        expected_cols = n if n is not None else cols
        checks.append(
            CheckResult(
                "K compatible with u = -Kx",
                rows > 0 and cols > 0 and rows == expected_rows and cols == expected_cols,
                f"K shape is {rows}x{cols}; expected {expected_rows}x{expected_cols}",
            )
        )
    if q_matrix is not None and n is not None:
        rows, cols = matrix_shape(q_matrix)
        checks.append(CheckResult("Q matches state dimension", rows == n and cols == n, f"Q shape is {rows}x{cols}"))
    if r_matrix is not None and m is not None:
        rows, cols = matrix_shape(r_matrix)
        checks.append(CheckResult("R matches input dimension", rows == m and cols == m, f"R shape is {rows}x{cols}"))
    return checks
