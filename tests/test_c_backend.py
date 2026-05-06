from __future__ import annotations

from controlkit.backends.c import CBackend, CBackendError
import pytest

from controlkit.compiler.ir import ControlLaw, IRModule, Matrix, Vector, matvec
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy


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
