#pipeline skeleton

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from controlkit.compiler.targets import TargetLanguage
from controlkit.exceptions import CompilationNotImplementedError, ControlKitError
from controlkit.policies.base import PolicyKind


@dataclass(frozen=True)
class CompileRequest:

    spec_path: Path
    policy: PolicyKind
    target: TargetLanguage
    output_dir: Path


@dataclass(frozen=True)
class CompileResult:
    success: bool
    message: str


class CompilerPipeline:
    
    def compile(self, request: CompileRequest) -> CompileResult:
        self._validate_request(request)
        try:
            self._lower_to_ir(request)
        except CompilationNotImplementedError as exc:
            return CompileResult(success=False, message=str(exc))
        return CompileResult(success=True, message="compilation completed")

    def _validate_request(self, request: CompileRequest) -> None:
        if not request.spec_path.exists():
            raise ControlKitError(f"spec file does not exist: {request.spec_path}")
        if not request.spec_path.is_file():
            raise ControlKitError(f"spec path is not a file: {request.spec_path}")

    def _lower_to_ir(self, request: CompileRequest) -> None:
        raise CompilationNotImplementedError(
            "compilation is not implemented yet; validated "
            f"{request.policy.value} policy for {request.target.value} target"
        )

