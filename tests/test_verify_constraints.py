from __future__ import annotations

from controlkit.verify.constraints import check_constraints


def test_invalid_input_bounds_fail() -> None:
    result = check_constraints(
        gain_matrix=[[1.0, 2.0]],
        input_lower=[1.0],
        input_upper=[1.0],
        saturation_declared=True,
    )

    assert not result.passed
    assert result.errors


def test_valid_input_bounds_pass() -> None:
    result = check_constraints(
        gain_matrix=[[1.0, 2.0]],
        input_lower=[-1.0],
        input_upper=[1.0],
        saturation_declared=True,
    )

    assert result.passed
