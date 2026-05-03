"""Policy frontend interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from controlkit.compiler.ir import IRModule


class PolicyKind(StrEnum):
    """Policy families ControlKit is expected to support."""

    PID = "pid"
    LQR = "lqr"
    MPC = "mpc"
    RL = "rl"


class PolicyFrontend(Protocol):
    """Frontend contract for policy parsers and lowerers."""

    kind: PolicyKind

    def load(self, spec_path: Path) -> "PolicySpec":
        """Load a policy specification from disk."""

    def lower(self, spec: "PolicySpec") -> "IRModule":
        """Lower a validated policy specification into ControlKit IR."""


@dataclass(frozen=True)
class PolicySpec:
    """Minimal policy specification wrapper used until schema validation lands."""

    name: str
    source_path: Path
