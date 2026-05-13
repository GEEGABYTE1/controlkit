from __future__ import annotations

import shutil
import subprocess

import pytest

from controlkit.backends.rust import RustBackend, RustBackendError
from controlkit.compiler.ir import ControlLaw, IRModule, Matrix, Vector, matvec
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy
from controlkit.policies.mpc import MpcPolicy
from controlkit.policies.rl import RlPolicy


def _mpc_module() -> IRModule:
    spec = MpcPolicy().from_matrices(
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
    return MpcPolicy().lower(spec)


def _rl_module() -> IRModule:
    spec = RlPolicy().from_layers(
        name="rl_balance",
        input_dim=2,
        output_dim=1,
        layers=[
            {
                "type": "linear",
                "weights": [[0.5, -0.25], [0.1, 0.4], [-0.3, 0.2], [0.7, -0.1]],
                "bias": [0.05, -0.02, 0.1, 0.0],
            },
            {"type": "relu"},
            {"type": "linear", "weights": [[0.8, -0.6, 0.3, 0.2]], "bias": [-0.05]},
            {"type": "tanh"},
        ],
    )
    return RlPolicy().lower(spec)


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
    assert (
        "pub fn control_step(x: &[f32; STATE_DIM], u: &mut [f32; CONTROL_DIM])"
        in artifact.source
    )
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


def test_rust_backend_generates_deterministic_mpc_source() -> None:
    artifact = RustBackend().generate(_mpc_module())

    assert artifact.source_name == "mpc_temperature.rs"
    assert "#![no_std]" in artifact.source
    assert "pub const STATE_DIM: usize = 2;" in artifact.source
    assert "pub const CONTROL_DIM: usize = 1;" in artifact.source
    assert "const MPC_HORIZON: usize = 3;" in artifact.source
    assert "const MPC_SOLVER_ITERATIONS: usize = 4;" in artifact.source
    assert "const U_MIN: [f32; 1] = [-0.5_f32];" in artifact.source
    assert "let mut u_seq = [[0.0_f32; CONTROL_DIM]; 3];" in artifact.source
    assert "u[j] = u_seq[0][j];" in artifact.source
    assert RustBackend().generate(_mpc_module()).source == artifact.source


def test_rust_backend_generated_mpc_source_compiles_when_rustc_is_available(tmp_path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")

    source_path = RustBackend().generate(_mpc_module()).write_to(tmp_path)
    output_path = tmp_path / "libmpc_temperature.rlib"

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


def test_rust_backend_generates_deterministic_rl_source() -> None:
    artifact = RustBackend().generate(_rl_module())

    assert artifact.source_name == "rl_balance.rs"
    assert "#![no_std]" in artifact.source
    assert "pub const STATE_DIM: usize = 2;" in artifact.source
    assert "pub const CONTROL_DIM: usize = 1;" in artifact.source
    assert "const LAYER_0_WEIGHTS: [[f32; 2]; 4]" in artifact.source
    assert "const LAYER_2_BIAS: [f32; 1] = [-0.05_f32];" in artifact.source
    assert "fn controlkit_tanh(x: f32) -> f32" in artifact.source
    assert "layer_1[i] = if value > 0.0_f32 { value } else { 0.0_f32 };" in artifact.source
    assert "layer_3[i] = controlkit_tanh(layer_2[i]);" in artifact.source
    assert RustBackend().generate(_rl_module()).source == artifact.source


def test_rust_backend_generated_rl_source_compiles_when_rustc_is_available(tmp_path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")

    source_path = RustBackend().generate(_rl_module()).write_to(tmp_path)
    output_path = tmp_path / "librl_balance.rlib"

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


def test_rust_backend_generated_rl_output_matches_reference_when_rustc_is_available(
    tmp_path,
) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")

    generated_source = RustBackend().generate(_rl_module()).source
    source = "\n".join(
        line for line in generated_source.splitlines() if line.strip() != "#![no_std]"
    )
    runner_path = tmp_path / "run_rl.rs"
    binary_path = tmp_path / "run_rl"
    runner_path.write_text(
        "\n".join(
            [
                source,
                "",
                "fn main() {",
                "    let x = [1.0_f32, 0.5_f32];",
                "    let mut u = [0.0_f32; CONTROL_DIM];",
                "    control_step(&x, &mut u);",
                '    println!("{:.9}", u[0]);',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            rustc,
            "--edition=2021",
            str(runner_path),
            "-o",
            str(binary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert float(completed.stdout.strip()) == pytest.approx(0.246798, abs=2e-3)
