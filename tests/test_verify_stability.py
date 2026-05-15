from __future__ import annotations

from controlkit.verify.stability import check_closed_loop_stability


def test_stable_discrete_lqr_passes() -> None:
    result = check_closed_loop_stability(
        a_matrix=[[1.0, 0.1], [0.0, 1.0]],
        b_matrix=[[0.0], [0.1]],
        gain_matrix=[[2.0, 2.0]],
        system_type="discrete",
    )

    assert result.passed
    assert result.stability_margin > 0.0


def test_unstable_discrete_controller_fails() -> None:
    result = check_closed_loop_stability(
        a_matrix=[[1.1, 0.0], [0.0, 0.9]],
        b_matrix=[[0.0], [0.0]],
        gain_matrix=[[0.0, 0.0]],
        system_type="discrete",
    )

    assert not result.passed


def test_stable_continuous_lqr_passes() -> None:
    result = check_closed_loop_stability(
        a_matrix=[[0.0, 1.0], [-2.0, -3.0]],
        b_matrix=[[0.0], [1.0]],
        gain_matrix=[[0.0, 0.0]],
        system_type="continuous",
    )

    assert result.passed
    assert result.stability_margin > 0.0
