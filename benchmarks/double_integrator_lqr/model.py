from __future__ import annotations

BENCHMARK = {
    "name": "double_integrator_lqr",
    "description": "Discrete double-integrator stabilization with state [position, velocity].",
    "dynamics_equations": "x[k+1] = A x[k] + B u[k], with u = -Kx and acceleration saturation.",
    "controller_description": "Single-input LQR-style feedback controller compiled by ControlKit.",
    "limitations": "Linear deterministic model; no actuator lag, noise, or sensor delay.",
    "dt": 0.05,
    "horizon_steps": 120,
    "initial_state": [1.0, 0.0],
    "pass_criteria": {
        "max_final_state_norm": 0.15,
        "max_state_norm": 1.2,
        "max_control_effort": 2.5,
    },
}

A = ((1.0, 0.05), (0.0, 1.0))
B = ((0.0,), (0.05,))


def step(state: list[float], control: list[float], _dt: float) -> list[float]:
    return [
        A[0][0] * state[0] + A[0][1] * state[1] + B[0][0] * control[0],
        A[1][0] * state[0] + A[1][1] * state[1] + B[1][0] * control[0],
    ]
