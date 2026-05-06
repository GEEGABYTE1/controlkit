#policy frontend interface
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from controlkit.compiler.ir import IRModule


class PolicyKind(StrEnum):
    PID = "pid"
    LQR = "lqr"
    MPC = "mpc"
    RL = "rl"


class PolicyFrontend(Protocol):
    kind: PolicyKind

    def load(self, spec_path: Path) -> "PolicySpec":
        """Load a policy specification from disk."""

    def lower(self, spec: "PolicySpec") -> "IRModule":
        """Lower a validated policy specification into ControlKit IR."""


@dataclass(frozen=True)
class PolicySpec:
    name: str
    source_path: Path
