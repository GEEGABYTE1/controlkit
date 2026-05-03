"""Project-specific exception hierarchy."""

from __future__ import annotations


class ControlKitError(Exception):
    """Base class for all expected ControlKit errors."""


class UnsupportedPolicyError(ControlKitError):
    """Raised when a requested policy frontend is not supported."""


class UnsupportedTargetError(ControlKitError):
    """Raised when a requested backend target is not supported."""


class CompilationNotImplementedError(ControlKitError):
    """Raised by placeholder compiler stages that are intentionally deferred."""

