#Generate Rust for a small saturated LQR controller.
#Run from the repository root with:
#PYTHONPATH=src python examples/rust_backend_lqr.py
from __future__ import annotations

from pathlib import Path

from controlkit.backends.rust import RustBackend
from controlkit.policies.lqr import LqrPolicy


def main() -> None:
    frontend = LqrPolicy()
    spec = frontend.from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    module = frontend.lower(spec)
    artifact = RustBackend().generate(module)
    source_path = artifact.write_to(Path("build/controlkit_rust"))

    print(source_path)


if __name__ == "__main__":
    main()
