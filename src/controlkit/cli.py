"""Command line interface for ControlKit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from controlkit import __version__
from controlkit.compiler.pipeline import CompileRequest, CompilerPipeline
from controlkit.compiler.targets import TargetLanguage
from controlkit.exceptions import ControlKitError
from controlkit.policies.base import PolicyKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="controlkit",
        description="Compile control policies into embedded C or Rust artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the ControlKit version.")
    version_parser.set_defaults(handler=_handle_version)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a policy specification file.")
    inspect_parser.add_argument("spec", type=Path, help="Path to a policy specification file.")
    inspect_parser.set_defaults(handler=_handle_inspect)

    compile_parser = subparsers.add_parser(
        "compile",
        help="Validate compile inputs and prepare a future compilation request.",
    )
    compile_parser.add_argument("spec", type=Path, help="Path to a policy specification file.")
    compile_parser.add_argument(
        "--policy",
        choices=[kind.value for kind in PolicyKind],
        required=True,
        help="Policy frontend to use.",
    )
    compile_parser.add_argument(
        "--target",
        choices=[target.value for target in TargetLanguage],
        required=True,
        help="Code generation target.",
    )
    compile_parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/controlkit"),
        help="Output directory for generated artifacts.",
    )
    compile_parser.set_defaults(handler=_handle_compile)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.handler(args)
    except ControlKitError as exc:
        parser.exit(status=2, message=f"controlkit: error: {exc}\n")
    return 0


def _handle_version(_args: argparse.Namespace) -> None:
    print(__version__)


def _handle_inspect(args: argparse.Namespace) -> None:
    spec_path: Path = args.spec
    if not spec_path.exists():
        raise ControlKitError(f"spec file does not exist: {spec_path}")
    if not spec_path.is_file():
        raise ControlKitError(f"spec path is not a file: {spec_path}")

    print(f"spec: {spec_path}")
    print(f"bytes: {spec_path.stat().st_size}")
    print("status: readable")


def _handle_compile(args: argparse.Namespace) -> None:
    request = CompileRequest(
        spec_path=args.spec,
        policy=PolicyKind(args.policy),
        target=TargetLanguage(args.target),
        output_dir=args.output,
    )
    result = CompilerPipeline().compile(request)
    print(result.message)

