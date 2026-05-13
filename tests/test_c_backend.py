from __future__ import annotations

import shutil
import subprocess

import pytest

from controlkit.backends.c import CBackend, CBackendError
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


def test_c_backend_generates_header_and_source_for_saturated_lqr() -> None:
    spec = LqrPolicy().from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    module = LqrPolicy().lower(spec)

    artifact = CBackend().generate(module)

    assert artifact.header_name == "cartpole_lqr.h"
    assert artifact.source_name == "cartpole_lqr.c"
    assert "void cartpole_lqr_control_step" in artifact.header
    assert "#define CONTROLKIT_STATE_DIM 4u" in artifact.header
    assert "#define CONTROLKIT_CONTROL_DIM 1u" in artifact.header
    assert 'static const float K[1u][4u] = {' in artifact.source
    assert "    {1.2f, 0.4f, 2.5f, 0.8f}" in artifact.source
    assert "acc += K[row][col] * x[col];" in artifact.source
    assert "tmp1[i] = -tmp0[i];" in artifact.source
    assert "if (clipped < -1.0f)" in artifact.source
    assert "if (clipped > 1.0f)" in artifact.source
    assert "u[i] = tmp2[i];" in artifact.source


def test_c_backend_writes_artifacts(tmp_path) -> None:
    spec = LqrPolicy().from_gain_matrix(name="simple", gain_matrix=[[2.0, 3.0]])
    artifact = CBackend().generate(LqrPolicy().lower(spec))

    header_path, source_path = artifact.write_to(tmp_path)

    assert header_path.read_text(encoding="utf-8") == artifact.header
    assert source_path.read_text(encoding="utf-8") == artifact.source


def test_c_backend_rejects_symbolic_matrix_without_values() -> None:
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

    with pytest.raises(CBackendError, match="no numeric values"):
        CBackend().generate(symbolic_module)


def test_c_backend_generates_deterministic_mpc_source() -> None:
    artifact = CBackend().generate(_mpc_module())

    assert artifact.header_name == "mpc_temperature.h"
    assert artifact.source_name == "mpc_temperature.c"
    assert "#define CONTROLKIT_STATE_DIM 2u" in artifact.header
    assert "#define CONTROLKIT_CONTROL_DIM 1u" in artifact.header
    assert "#define CONTROLKIT_MPC_HORIZON 3u" in artifact.source
    assert "#define CONTROLKIT_MPC_SOLVER_ITERATIONS 4u" in artifact.source
    assert "static const float A[2u][2u]" in artifact.source
    assert "static const float U_MIN[1u] = {-0.5f};" in artifact.source
    assert "float U[CONTROLKIT_MPC_HORIZON][1u] = {0};" in artifact.source
    assert "u[j] = U[0u][j];" in artifact.source
    assert CBackend().generate(_mpc_module()).source == artifact.source


def test_c_backend_generated_mpc_source_compiles_when_cc_is_available(tmp_path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("cc is not installed")

    header_path, source_path = CBackend().generate(_mpc_module()).write_to(tmp_path)
    output_path = tmp_path / "mpc_temperature.o"

    subprocess.run(
        [
            cc,
            "-std=c99",
            "-c",
            str(source_path),
            "-o",
            str(output_path),
            "-I",
            str(header_path.parent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()


def test_c_backend_generates_deterministic_rl_source() -> None:
    artifact = CBackend().generate(_rl_module())

    assert artifact.header_name == "rl_balance.h"
    assert artifact.source_name == "rl_balance.c"
    assert "#define CONTROLKIT_STATE_DIM 2u" in artifact.header
    assert "#define CONTROLKIT_CONTROL_DIM 1u" in artifact.header
    assert "#include <math.h>" in artifact.source
    assert "static const float LAYER_0_WEIGHTS[4u][2u]" in artifact.source
    assert "static const float LAYER_2_BIAS[1u] = {-0.05f};" in artifact.source
    assert "layer_1[i] = value > 0.0f ? value : 0.0f;" in artifact.source
    assert "layer_3[i] = tanhf(layer_2[i]);" in artifact.source
    assert CBackend().generate(_rl_module()).source == artifact.source


def test_c_backend_generated_rl_source_compiles_when_cc_is_available(tmp_path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("cc is not installed")

    header_path, source_path = CBackend().generate(_rl_module()).write_to(tmp_path)
    output_path = tmp_path / "rl_balance.o"

    subprocess.run(
        [
            cc,
            "-std=c99",
            "-c",
            str(source_path),
            "-o",
            str(output_path),
            "-I",
            str(header_path.parent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()


def test_c_backend_generated_rl_output_matches_reference_when_cc_is_available(tmp_path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("cc is not installed")

    header_path, source_path = CBackend().generate(_rl_module()).write_to(tmp_path)
    runner_path = tmp_path / "run_rl.c"
    binary_path = tmp_path / "run_rl"
    runner_path.write_text(
        "\n".join(
            [
                "#include <stdio.h>",
                f'#include "{header_path.name}"',
                "",
                "int main(void) {",
                "    float x[CONTROLKIT_STATE_DIM] = {1.0f, 0.5f};",
                "    float u[CONTROLKIT_CONTROL_DIM] = {0.0f};",
                "    rl_balance_control_step(x, u);",
                '    printf("%.9f\\n", u[0]);',
                "    return 0;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            cc,
            "-std=c99",
            str(source_path),
            str(runner_path),
            "-o",
            str(binary_path),
            "-I",
            str(header_path.parent),
            "-lm",
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

    assert float(completed.stdout.strip()) == pytest.approx(0.246798, abs=1e-5)
