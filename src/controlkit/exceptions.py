# Project-specific exception hierarchy.

from __future__ import annotations


class ControlKitError(Exception):
    # Base class for all expected ControlKit errors.
    pass


class UnsupportedPolicyError(ControlKitError):
    # Raised when a requested policy frontend is not supported.
    pass


class UnsupportedTargetError(ControlKitError):
    # Raised when a requested backend target is not supported.
    pass


class CompilationNotImplementedError(ControlKitError):
    # Raised by placeholder compiler stages that are intentionally deferred.
    pass

