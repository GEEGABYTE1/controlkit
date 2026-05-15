"""Closed-loop stability checks."""

from __future__ import annotations

import cmath
from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityResult:
    passed: bool
    system_type: str
    eigenvalues: list[complex]
    stability_margin: float
    message: str


def closed_loop_matrix(
    a_matrix: list[list[float]],
    b_matrix: list[list[float]],
    gain_matrix: list[list[float]],
) -> list[list[float]]:
    rows = len(a_matrix)
    cols = len(a_matrix[0])
    control_dim = len(gain_matrix)
    result = [[a_matrix[row][col] for col in range(cols)] for row in range(rows)]
    for row in range(rows):
        for col in range(cols):
            result[row][col] -= sum(b_matrix[row][k] * gain_matrix[k][col] for k in range(control_dim))
    return result


def check_closed_loop_stability(
    *,
    a_matrix: list[list[float]],
    b_matrix: list[list[float]],
    gain_matrix: list[list[float]],
    system_type: str = "discrete",
    boundary_tolerance: float = 1e-6,
) -> StabilityResult:
    a_cl = closed_loop_matrix(a_matrix, b_matrix, gain_matrix)
    eigenvalues = estimate_eigenvalues(a_cl)
    if system_type == "continuous":
        max_real = max(value.real for value in eigenvalues)
        margin = -max_real
        passed = max_real < -boundary_tolerance
        message = f"max real eigenvalue = {max_real:.6g}"
    else:
        spectral_radius = max(abs(value) for value in eigenvalues)
        margin = 1.0 - spectral_radius
        passed = spectral_radius < 1.0 - boundary_tolerance
        message = f"spectral radius = {spectral_radius:.6g}"
    return StabilityResult(
        passed=passed,
        system_type=system_type,
        eigenvalues=eigenvalues,
        stability_margin=margin,
        message=message,
    )


def estimate_eigenvalues(matrix: list[list[float]]) -> list[complex]:
    n = len(matrix)
    if n == 1:
        return [complex(matrix[0][0], 0.0)]
    if n == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        trace = a + d
        determinant = a * d - b * c
        discriminant = trace * trace - 4.0 * determinant
        root = cmath.sqrt(discriminant)
        return [(trace + root) / 2.0, (trace - root) / 2.0]
    return _qr_eigenvalues(matrix)


def _qr_eigenvalues(matrix: list[list[float]], iterations: int = 80) -> list[complex]:
    current = [row[:] for row in matrix]
    for _ in range(iterations):
        q, r = _qr_decompose(current)
        current = _matmul(r, q)
    return [complex(current[index][index], 0.0) for index in range(len(current))]


def _qr_decompose(matrix: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    n = len(matrix)
    columns = [[matrix[row][col] for row in range(n)] for col in range(n)]
    q_cols: list[list[float]] = []
    r = [[0.0 for _ in range(n)] for _ in range(n)]
    for col, vector in enumerate(columns):
        v = vector[:]
        for prev, q_col in enumerate(q_cols):
            r[prev][col] = sum(q_col[i] * vector[i] for i in range(n))
            v = [v[i] - r[prev][col] * q_col[i] for i in range(n)]
        norm = sum(value * value for value in v) ** 0.5
        if norm < 1e-12:
            q_col = [0.0 for _ in range(n)]
        else:
            q_col = [value / norm for value in v]
        r[col][col] = norm
        q_cols.append(q_col)
    q = [[q_cols[col][row] for col in range(n)] for row in range(n)]
    return q, r


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    n = len(left)
    return [
        [sum(left[row][k] * right[k][col] for k in range(n)) for col in range(n)]
        for row in range(n)
    ]
