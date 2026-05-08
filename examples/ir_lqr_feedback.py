#Build a simple ControlKit IR module for `u = -Kx`.
#Run from the repository root with:
#PYTHONPATH=src python examples/ir_lqr_feedback.py
from __future__ import annotations

from controlkit.compiler.ir import (
    ControlLaw,
    DynamicsKind,
    IRModule,
    LinearSystemIR,
    Matrix,
    Vector,
    clip,
    matvec,
    neg,
)
from controlkit.policies.base import PolicyKind


def build_module() -> IRModule:
    x = Vector("x", dim=4)
    u = Vector("u", dim=1)
    a = Matrix("A", rows=4, cols=4)
    b = Matrix("B", rows=4, cols=1)
    k = Matrix("K", rows=1, cols=4)

    dynamics = LinearSystemIR(
        state=x,
        control=u,
        a_matrix=a,
        b_matrix=b,
        dynamics=DynamicsKind.DISCRETE,
    )
    law = ControlLaw(output=u, expression=clip(neg(matvec(k, x)), lower=-1.0, upper=1.0))

    return IRModule(
        name="lqr_feedback",
        policy=PolicyKind.LQR,
        systems=(dynamics,),
        control_laws=(law,),
    )


if __name__ == "__main__":
    module = build_module()
    print(module)
    print(module.systems[0])
    print(module.control_laws[0])

