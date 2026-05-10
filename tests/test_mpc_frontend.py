from __future__ import annotations

import pytest

from controlkit.benchmarks.runner import _evaluate_mpc_controller
from controlkit.compiler.ir import DynamicsKind, MpcControllerIR
from controlkit.policies.base import PolicyKind
from controlkit.policies.mpc import MpcControllerSpec, MpcPolicy, MpcSpecError


def _valid_spec() -> MpcControllerSpec:
    return MpcPolicy().from_matrices(
        name="mpc_temperature",
        a_matrix=((1.0, 1.0), (0.0, 1.0)),
        b_matrix=((0.0,), (1.0,)),
        horizon=3,
        q_diagonal=(1.0, 0.1),
        r_diagonal=(0.05,),
        q_terminal_diagonal=(1.5, 0.2),
        u_min=(-0.5,),
        u_max=(0.5,),
        solver_iterations=4,
        step_size=0.1,
    )


def test_mpc_frontend_validates_and_lowers_to_ir() -> None:
    spec = _valid_spec()
    module = MpcPolicy().lower(spec)

    assert module.name == "mpc_temperature"
    assert module.policy == PolicyKind.MPC
    assert len(module.systems) == 1
    assert len(module.control_laws) == 0
    assert len(module.mpc_controllers) == 1
    assert module.systems[0].dynamics is DynamicsKind.DISCRETE
    assert isinstance(module.mpc_controllers[0], MpcControllerIR)
    assert repr(module.mpc_controllers[0]) == (
        "MpcControllerIR(name='mpc_temperature', state_dim=2, "
        "control_dim=1, horizon=3, solver_iterations=4)"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"a_matrix": ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0))}, "a_matrix shape"),
        ({"b_matrix": ((0.0, 1.0), (1.0, 0.0))}, "b_matrix shape"),
        ({"q_diagonal": (1.0,)}, "q_diagonal"),
        ({"r_diagonal": (0.05, 0.1)}, "r_diagonal"),
        ({"u_min": (-0.5, -0.25)}, "u_min"),
        ({"u_min": (1.0,), "u_max": (-1.0,)}, "u_min entries"),
        ({"horizon": 0}, "horizon"),
        ({"solver_iterations": 0}, "solver_iterations"),
        ({"step_size": 0.0}, "step_size"),
    ],
)
def test_mpc_frontend_rejects_invalid_shapes_and_solver_settings(kwargs, message) -> None:
    params = {
        "name": "bad_mpc",
        "a_matrix": ((1.0, 1.0), (0.0, 1.0)),
        "b_matrix": ((0.0,), (1.0,)),
        "state_dim": 2,
        "control_dim": 1,
        "horizon": 3,
        "q_diagonal": (1.0, 0.1),
        "r_diagonal": (0.05,),
        "q_terminal_diagonal": (1.5, 0.2),
        "u_min": (-0.5,),
        "u_max": (0.5,),
        "solver_iterations": 4,
        "step_size": 0.1,
    }
    params.update(kwargs)

    with pytest.raises(MpcSpecError, match=message):
        MpcPolicy().from_matrices(**params)


def test_mpc_python_reference_respects_control_bounds() -> None:
    controller = MpcPolicy().lower(_valid_spec()).mpc_controllers[0]

    output = _evaluate_mpc_controller(controller, [10.0, 10.0])

    assert controller.u_min[0] <= output[0] <= controller.u_max[0]
