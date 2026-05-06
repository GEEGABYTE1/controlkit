from __future__ import annotations

import pytest
from controlkit.compiler.ir import Clip, ControlLaw, MatVecMul, Neg, Shape
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrControllerSpec, LqrPolicy, LqrSaturation, LqrSpecError

def test_lqr_frontend_infers_dimensions_and_lowers_negative_feedback() -> None:
    spec = LqrPolicy().from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.0, 2.0, 3.0, 4.0]],
    )

    module = LqrPolicy().lower(spec)
    law = module.control_laws[0]
    assert isinstance(spec, LqrControllerSpec)
    assert spec.state_dim == 4
    assert spec.control_dim == 1
    assert module.name == "cartpole_lqr"
    assert module.policy == PolicyKind.LQR
    assert module.metadata["gain_name"] == "K"
    assert isinstance(law, ControlLaw)
    assert law.output.name == "u"
    assert law.output.shape == Shape.vector(1)
    assert isinstance(law.expression, Neg)
    assert isinstance(law.expression.value, MatVecMul)
    assert repr(law) == "u = (-(K:matrix[1x4] @ x:vector[4]))"

def test_lqr_frontend_lowers_saturated_custom_names() -> None:
    spec = LqrPolicy().from_gain_matrix(
        name="attitude_lqr",
        gain_matrix=[[0.5, 0.1], [0.2, 0.7]],
        saturation=(-0.25, 0.25),
        state_name="state",
        control_name="torque",
        gain_name="K_att",
        state_names=("angle", "rate"),
        control_names=("left", "right"),
    )

    module = LqrPolicy().lower(spec)
    law = module.control_laws[0]
    assert spec.saturation == LqrSaturation(lower=-0.25, upper=0.25)
    assert module.metadata["state_names"] == "angle,rate"
    assert module.metadata["control_names"] == "left,right"
    assert module.metadata["saturation"] == "-0.25,0.25"
    assert law.output.name == "torque"
    assert isinstance(law.expression, Clip)
    assert repr(law) == (
        "torque = clip((-(K_att:matrix[2x2] @ state:vector[2])), const(-0.25), const(0.25))"
    )

def test_lqr_frontend_rejects_gain_shape_that_disagrees_with_dimensions() -> None:
    with pytest.raises(LqrSpecError, match="control_dim x state_dim"):
        LqrPolicy().from_gain_matrix(
            name="bad_lqr",
            gain_matrix=[[1.0, 2.0, 3.0]],
            state_dim=2,
            control_dim=1,
        )

def test_lqr_frontend_rejects_ragged_gain_matrix() -> None:
    with pytest.raises(LqrSpecError, match="same length"):
        LqrPolicy().from_gain_matrix(
            name="ragged_lqr",
            gain_matrix=[[1.0, 2.0], [3.0]],
        )

def test_lqr_frontend_rejects_invalid_saturation_and_component_names() -> None:
    with pytest.raises(LqrSpecError, match="lower bound"):
        LqrPolicy().from_gain_matrix(
            name="bad_saturation",
            gain_matrix=[[1.0, 2.0]],
            saturation=(1.0, -1.0),
        )

    with pytest.raises(LqrSpecError, match="state_names"):
        LqrPolicy().from_gain_matrix(
            name="bad_names",
            gain_matrix=[[1.0, 2.0]],
            state_names=("only_one",),
        )
