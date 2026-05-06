#mpc placeholder

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from controlkit.compiler.ir import IRModule
from controlkit.policies.base import PolicyKind, PolicySpec


@dataclass(frozen=True)
class MpcPolicy:
    kind: PolicyKind = PolicyKind.MPC

    def load(self, spec_path: Path) -> PolicySpec:
        return PolicySpec(name=spec_path.stem, source_path=spec_path)

    def lower(self, spec: PolicySpec) -> IRModule:
        return IRModule(name=spec.name, policy=self.kind)

