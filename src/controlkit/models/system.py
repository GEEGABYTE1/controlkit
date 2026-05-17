# Shared control-system model placeholders.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateSpaceShape:
    # Dimensions for a state-space system.

    states: int
    inputs: int
    outputs: int


@dataclass(frozen=True)
class LinearSystem:
    
    name: str
    shape: StateSpaceShape
    sample_time_seconds: float

