"""Backend target definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetLanguage(StrEnum):
    """Supported code generation targets."""

    C = "c"
    RUST = "rust"


@dataclass(frozen=True)
class TargetSpec:
    """Backend target configuration."""

    language: TargetLanguage
    fixed_point: bool = False
    no_std: bool = True

