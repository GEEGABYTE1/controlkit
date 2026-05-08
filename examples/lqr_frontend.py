
#Build an LQR controller spec and lower it into ControlKit IR.
#Run from the repository root with:
#PYTHONPATH=src python examples/lqr_frontend.py

from __future__ import annotations

from controlkit.policies.lqr import LqrPolicy

def main() -> None:
    frontend = LqrPolicy()
    spec = frontend.from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
        state_names=("position", "velocity", "angle", "angular_velocity"),
        control_names=("force",),
    )
    module = frontend.lower(spec)

    print(module)
    print(module.control_laws[0])


if __name__ == "__main__":
    main()
