#Run the default ControlKit optimization pass over a small IR module.
#Run from the repository root with:
#PYTHONPATH=src python examples/optimization_pass.py

from __future__ import annotations

from controlkit.compiler.ir import ControlLaw, IRModule, Vector, add, scalar_mul, zero
from controlkit.optimization import optimize_module
from controlkit.policies.base import PolicyKind


def main() -> None:
    x = Vector("x", dim=2)
    u = Vector("u", dim=2)
    module = IRModule(
        name="identity_feedback",
        policy=PolicyKind.LQR,
        control_laws=(
            ControlLaw(
                output=u,
                expression=add(scalar_mul(1.0, x), zero(x.shape)),
            ),
        ),
    )

    result = optimize_module(module)
    print(module.control_laws[0])
    print(result.module.control_laws[0])
    print(result.report)


if __name__ == "__main__":
    main()
