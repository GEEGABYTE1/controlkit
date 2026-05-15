from __future__ import annotations

BENCHMARK = {
    "name": "pid_mass_spring_damper",
    "description": "PID regulation of a damped mass-spring system to the origin.",
    "dynamics_equations": "x_dot = v, v_dot = (u - c v - k x) / m, integrated with Euler steps.",
    "controller_description": "Scalar PID controller with force saturation.",
    "limitations": "Reference PID benchmark only; PID lowering/codegen is a documented future compiler path.",
    "dt": 0.01,
    "horizon_steps": 300,
    "initial_state": [0.5, 0.0],
    "pass_criteria": {
        "max_final_state_norm": 0.12,
        "max_state_norm": 1.0,
        "max_control_effort": 4.0,
    },
}

MASS = 1.0
DAMPING = 0.45
STIFFNESS = 2.0


def step(state: list[float], control: list[float], dt: float) -> list[float]:
    position, velocity = state
    force = control[0]
    acceleration = (force - DAMPING * velocity - STIFFNESS * position) / MASS
    return [position + dt * velocity, velocity + dt * acceleration]
