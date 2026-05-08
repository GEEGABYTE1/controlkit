
# Generate C for small sat LQR Controller 
#Run from the repository root with:
# PYTHONPATH=src python examples/c_backend_lqr.py


from __future__ import annotations

from pathlib import Path

from controlkit.backends.c import CBackend
from controlkit.policies.lqr import LqrPolicy


def main() -> None:
    frontend = LqrPolicy()
    spec = frontend.from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    module = frontend.lower(spec)
    artifact = CBackend().generate(module)
    header_path, source_path = artifact.write_to(Path("build/controlkit_c"))

    print(header_path)
    print(source_path)


if __name__ == "__main__":
    main()
