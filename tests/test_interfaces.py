from __future__ import annotations

from pathlib import Path

from controlkit.compiler.ir import IRModule
from controlkit.policies import LqrPolicy, MpcPolicy, PidPolicy, RlPolicy
from controlkit.policies.base import PolicyKind


def test_policy_frontends_lower_to_named_ir_modules(tmp_path: Path) -> None:
    spec_path = tmp_path / "controller.yaml"
    spec_path.write_text("name: controller\n", encoding="utf-8")

    frontends = [
        (PidPolicy(), PolicyKind.PID),
        (LqrPolicy(), PolicyKind.LQR),
        (MpcPolicy(), PolicyKind.MPC),
        (RlPolicy(), PolicyKind.RL),
    ]

    for frontend, expected_kind in frontends:
        spec = frontend.load(spec_path)
        module = frontend.lower(spec)

        assert isinstance(module, IRModule)
        assert module.name == "controller"
        assert module.policy == expected_kind

