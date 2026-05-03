"""Intermediate representation placeholders.

The IR is deliberately small for now. Future work will turn policy frontends into this stable
representation before optimization and backend code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from controlkit.policies.base import PolicyKind


@dataclass(frozen=True)
class IRModule:
    """A policy lowered into ControlKit's backend-neutral representation."""

    name: str
    policy: PolicyKind
    metadata: Mapping[str, str] = field(default_factory=dict)

