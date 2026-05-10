from __future__ import annotations

import shutil
import subprocess

import pytest

from controlkit.backends.c import CBackend, CBackendError
from controlkit.compiler.ir import ControlLaw, IRModule, Matrix, Vector, matvec
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy
from controlkit.policies.mpc import MpcPolicy


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
