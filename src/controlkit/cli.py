"""Command line interface for ControlKit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from controlkit import __version__
from controlkit.backends.c import CBackend
from controlkit.backends.rust import RustBackend
from controlkit.benchmarks import BenchmarkConfig, benchmark_module
from controlkit.compiler.targets import TargetLanguage
from controlkit.exceptions import ControlKitError
from controlkit.frontend import ControllerSpecError, load_controller_spec


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

    validate_parser = subparsers.add_parser("validate", help="Validate a controller spec.")
    validate_parser.add_argument("spec", type=Path, help="Path to a controller YAML spec.")
    validate_parser.set_defaults(handler=_handle_validate)

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a controller spec to C or Rust.",
    )
    compile_parser.add_argument("spec", type=Path, help="Path to a controller YAML spec.")
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
    compile_parser.add_argument(
        "--unroll-loops",
        action="store_true",
        help="Unroll fixed-size loops in generated code when supported.",
    )
    compile_parser.set_defaults(handler=_handle_compile)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a controller spec and write JSON/Markdown reports.",
    )
    benchmark_parser.add_argument("spec", type=Path, help="Path to a controller YAML spec.")
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/benchmarks"),
        help="Output directory for benchmark reports.",
    )
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Measured benchmark iterations.",
    )
    benchmark_parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1_000,
        help="Warmup iterations before measurement.",
    )
    benchmark_parser.add_argument("--no-c", action="store_true", help="Skip generated C benchmark.")
    benchmark_parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Skip generated Rust benchmark.",
    )
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.handler(args)
    except (ControlKitError, ControllerSpecError) as exc:
        parser.exit(status=2, message=f"controlkit: error: {exc}\n")
    return 0


def _handle_version(_args: argparse.Namespace) -> None:
    print(__version__)


def _handle_inspect(args: argparse.Namespace) -> None:
    loaded = load_controller_spec(args.spec)
    print(f"spec: {loaded.path}")
    print(f"bytes: {loaded.path.stat().st_size}")
    print(f"policy: {loaded.policy}")
    print(f"module: {loaded.module.name}")
    print(f"control_laws: {len(loaded.module.control_laws)}")
    print(f"mpc_controllers: {len(loaded.module.mpc_controllers)}")
    print(f"rl_policies: {len(loaded.module.rl_policies)}")
    for key in sorted(loaded.module.metadata):
        print(f"{key}: {loaded.module.metadata[key]}")


def _handle_compile(args: argparse.Namespace) -> None:
    loaded = load_controller_spec(args.spec)
    target = TargetLanguage(args.target)
    if target is TargetLanguage.C:
        artifact = CBackend(unroll_loops=args.unroll_loops).generate(loaded.module)
        header_path, source_path = artifact.write_to(args.output)
        print(header_path)
        print(source_path)
        return
    if target is TargetLanguage.RUST:
        source_path = RustBackend(unroll_loops=args.unroll_loops).generate(loaded.module).write_to(
            args.output
        )
        print(source_path)
        return
    raise ControlKitError(f"unsupported target: {target.value}")


def _handle_validate(args: argparse.Namespace) -> None:
    loaded = load_controller_spec(args.spec)
    print(f"valid: {loaded.path}")
    print(f"policy: {loaded.policy}")
    print(f"module: {loaded.module.name}")


def _handle_benchmark(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ControlKitError("--iterations must be positive")
    if args.warmup_iterations < 0:
        raise ControlKitError("--warmup-iterations must be non-negative")
    loaded = load_controller_spec(args.spec)
    report = benchmark_module(
        loaded.module,
        BenchmarkConfig(
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            include_c=not args.no_c,
            include_rust=not args.no_rust,
        ),
    )
    json_path = report.write_json(args.output / f"{loaded.module.name}.json")
    markdown_path = report.write_markdown(args.output / f"{loaded.module.name}.md")
    print(json_path)
    print(markdown_path)
