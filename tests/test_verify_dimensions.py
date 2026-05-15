from __future__ import annotations

from controlkit.verify.dimensions import verify_dimensions


def test_wrong_k_shape_fails() -> None:
    checks = verify_dimensions(
        a_matrix=[[1.0, 0.0], [0.0, 1.0]],
        b_matrix=[[0.0], [1.0]],
        gain_matrix=[[1.0, 2.0, 3.0]],
    )

    assert any(check.name == "K compatible with u = -Kx" and not check.passed for check in checks)


def test_valid_dimensions_pass() -> None:
    checks = verify_dimensions(
        a_matrix=[[1.0, 0.0], [0.0, 1.0]],
        b_matrix=[[0.0], [1.0]],
        gain_matrix=[[1.0, 2.0]],
    )

    assert all(check.passed for check in checks)
