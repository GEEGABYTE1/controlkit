from __future__ import annotations

BENCHMARK = {
    "name": "cartpole_lqr_linearized",
    "description": "Small-angle cartpole stabilization around the upright equilibrium.",
    "dynamics_equations": "A compact discrete linearization with state [x, x_dot, theta, theta_dot].",
    "controller_description": "Single-force saturated LQR-style feedback controller.",
    "limitations": "Linearized small-angle model only; not a nonlinear cartpole simulator.",
    "dt": 0.02,
    "horizon_steps": 160,
    "initial_state": [0.2, 0.0, 0.12, 0.0],
    "pass_criteria": {
        "max_final_state_norm": 0.25,
        "max_state_norm": 0.6,
        "max_control_effort": 2.0,
    },
}

A = (
    (1.0, 0.02, 0.0, 0.0),
    (0.0, 0.96, 0.08, 0.0),
    (0.0, 0.0, 1.0, 0.02),
    (0.0, 0.04, 0.28, 0.94),
)
B = ((0.0,), (0.04,), (0.0,), (0.09,))


def step(state: list[float], control: list[float], _dt: float) -> list[float]:
    return [
        sum(A[row][col] * state[col] for col in range(4)) + B[row][0] * control[0]
        for row in range(4)
    ]
