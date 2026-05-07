from __future__ import annotations

import shutil
import subprocess

import pytest

from controlkit.backends.rust import RustBackend, RustBackendError
from controlkit.compiler.ir import ControlLaw, IRModule, Matrix, Vector, matvec
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy


def test_rust_backend_generates_no_std_source_for_saturated_lqr() -> None:
    spec = LqrPolicy().from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    module = LqrPolicy().lower(spec)

    artifact = RustBackend().generate(module)

    assert artifact.source_name == "cartpole_lqr.rs"
    assert "#![no_std]" in artifact.source
    assert "pub const STATE_DIM: usize = 4;" in artifact.source
    assert "pub const CONTROL_DIM: usize = 1;" in artifact.source
    assert "const K: [[f32; 4]; 1] = [" in artifact.source
    assert "    [1.2_f32, 0.4_f32, 2.5_f32, 0.8_f32]," in artifact.source
    assert "pub fn control_step(x: &[f32; STATE_DIM], u: &mut [f32; CONTROL_DIM])" in artifact.source
    assert "acc += K[row][col] * x[col];" in artifact.source
    assert "tmp1[i] = -tmp0[i];" in artifact.source
    assert "if clipped < -1.0_f32" in artifact.source
    assert "if clipped > 1.0_f32" in artifact.source
    assert "u[i] = tmp2[i];" in artifact.source


def test_rust_backend_writes_artifact(tmp_path) -> None:
    spec = LqrPolicy().from_gain_matrix(name="simple", gain_matrix=[[2.0, 3.0]])
    artifact = RustBackend().generate(LqrPolicy().lower(spec))

    source_path = artifact.write_to(tmp_path)

    assert source_path.read_text(encoding="utf-8") == artifact.source


def test_rust_backend_rejects_symbolic_matrix_without_values() -> None:
    x = Vector("x", dim=2)
    symbolic_module = IRModule(
        name="symbolic",
        policy=PolicyKind.LQR,
        control_laws=(
            ControlLaw(
                output=Vector("u", dim=1),
                expression=matvec(Matrix("K_symbolic", rows=1, cols=2), x),
            ),
        ),
    )

    with pytest.raises(RustBackendError, match="no numeric values"):
        RustBackend().generate(symbolic_module)


def test_rust_backend_generated_source_compiles_when_rustc_is_available(tmp_path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")

    spec = LqrPolicy().from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    source_path = RustBackend().generate(LqrPolicy().lower(spec)).write_to(tmp_path)
    output_path = tmp_path / "libcartpole_lqr.rlib"

    subprocess.run(
        [
            rustc,
            "--edition=2021",
            "--crate-type",
            "lib",
            str(source_path),
            "-o",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
