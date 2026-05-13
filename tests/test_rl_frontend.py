from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from controlkit.benchmarks.runner import _evaluate_rl_policy
from controlkit.compiler.ir import RlPolicyIR
from controlkit.frontend import load_controller_spec
from controlkit.policies.base import PolicyKind
from controlkit.policies.rl import RlControllerSpec, RlPolicy, RlSpecError


def _layers():
    return [
        {
            "type": "linear",
            "weights": [
                [0.5, -0.25],
                [0.1, 0.4],
                [-0.3, 0.2],
                [0.7, -0.1],
            ],
            "bias": [0.05, -0.02, 0.1, 0.0],
        },
        {"type": "relu"},
        {"type": "linear", "weights": [[0.8, -0.6, 0.3, 0.2]], "bias": [-0.05]},
        {"type": "tanh"},
    ]


def _valid_spec() -> RlControllerSpec:
    return RlPolicy().from_layers(
        name="rl_balance",
        input_dim=2,
        output_dim=1,
        layers=_layers(),
    )


def test_rl_frontend_validates_and_lowers_to_ir() -> None:
    spec = _valid_spec()
    module = RlPolicy().lower(spec)

    assert module.name == "rl_balance"
    assert module.policy == PolicyKind.RL
    assert len(module.control_laws) == 0
    assert len(module.mpc_controllers) == 0
    assert len(module.rl_policies) == 1
    assert isinstance(module.rl_policies[0], RlPolicyIR)
    assert repr(module.rl_policies[0]) == (
        "RlPolicyIR(name='rl_balance', input_dim=2, output_dim=1, layers=4)"
    )


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ([], "at least one layer"),
        ([{"type": "linear", "weights": [[1.0], [2.0, 3.0]], "bias": [0.0]}], "same length"),
        ([{"type": "linear", "weights": [[1.0, 2.0]], "bias": [0.0, 1.0]}], "bias"),
        ([{"type": "sigmoid"}], "unsupported"),
        ([{"type": "linear", "weights": [[1.0, 2.0, 3.0]], "bias": [0.0]}], "does not match"),
    ],
)
def test_rl_frontend_rejects_invalid_networks(layers, message) -> None:
    with pytest.raises(RlSpecError, match=message):
        RlPolicy().from_layers(
            name="bad_rl",
            input_dim=2,
            output_dim=1,
            layers=layers,
        )


def test_rl_frontend_loads_yaml_with_json_weights(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps({"input_dim": 2, "output_dim": 1, "layers": _layers()}),
        encoding="utf-8",
    )
    spec_path = tmp_path / "rl.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: rl_balance",
                "policy: rl",
                "weights_path: weights.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_controller_spec(spec_path)

    assert loaded.policy == "rl"
    assert loaded.module.metadata["frontend"] == "rl"
    assert len(loaded.module.rl_policies) == 1


def test_rl_python_reference_inference_is_deterministic() -> None:
    policy = RlPolicy().lower(_valid_spec()).rl_policies[0]

    output = _evaluate_rl_policy(policy.layers, [1.0, 0.5])

    assert output == pytest.approx([math.tanh(0.252)])
