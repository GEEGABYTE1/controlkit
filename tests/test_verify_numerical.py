from __future__ import annotations

from controlkit.verify.numerical import check_numerical_robustness


def test_nan_matrix_fails() -> None:
    result = check_numerical_robustness({"A": [[1.0, float("nan")], [0.0, 1.0]]})

    assert not result.passed
    assert result.errors


def test_finite_matrix_passes() -> None:
    result = check_numerical_robustness({"A": [[1.0, 0.0], [0.0, 2.0]]})

    assert result.passed
    assert result.condition_numbers["A"] >= 1.0
