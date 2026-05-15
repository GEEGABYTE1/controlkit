from __future__ import annotations

BENCHMARK = {
    "name": "rocket_hover_lqr",
    "description": "Toy rocket hover linearization with lateral/vertical position and pitch states.",
    "dynamics_equations": "State [x, z, vx, vz, theta, theta_dot], control [thrust_delta, torque].",
    "controller_description": "Two-input saturated LQR-style hover controller.",
    "limitations": "No mass depletion, aerodynamic forces, nonlinear attitude coupling, or guidance loop.",
    "dt": 0.02,
    "horizon_steps": 180,
    "initial_state": [0.25, -0.2, 0.0, 0.0, 0.08, 0.0],
    "pass_criteria": {
        "max_final_state_norm": 0.2,
        "max_state_norm": 0.5,
        "max_control_effort": 2.5,
    },
}

A = (
    (1.0, 0.0, 0.02, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.02, 0.0, 0.0),
    (0.0, 0.0, 0.98, 0.0, 0.08, 0.0),
    (0.0, 0.0, 0.0, 0.98, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.02),
    (0.0, 0.0, 0.0, 0.0, -0.10, 0.96),
)
B = (
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.02),
    (0.05, 0.0),
    (0.0, 0.0),
    (0.0, 0.08),
)


def step(state: list[float], control: list[float], _dt: float) -> list[float]:
    return [
        sum(A[row][col] * state[col] for col in range(6))
        + sum(B[row][col] * control[col] for col in range(2))
        for row in range(6)
    ]
