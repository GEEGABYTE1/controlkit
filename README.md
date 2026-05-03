# ControlKit

ControlKit is a high-performance control-system compiler CLI. Its long-term goal is to compile
LQR, MPC, PID, and reinforcement-learning policies into optimized C or Rust for embedded
deployment.

This repository is currently a production-grade scaffold: package boundaries, public interfaces,
CLI entrypoint, tests, examples, and project documentation are in place, while the heavy compiler
implementation is intentionally deferred.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
controlkit --help
pytest
```

Without installing the package, the CLI can also be run from the repo root:

```bash
PYTHONPATH=src python -m controlkit --help
```

## Current CLI

```bash
controlkit version
controlkit inspect examples/pid_basic.yaml
controlkit compile examples/pid_basic.yaml --policy pid --target c
```

`compile` validates the requested policy and target, then reports that backend code generation is
not implemented yet. That behavior is deliberate until the IR, optimization passes, and backend
contracts are hardened.

## Repository Layout

```text
src/controlkit/        Python package
src/controlkit/cli.py  Console entrypoint
src/controlkit/compiler/
                       Compiler interfaces, IR, pipeline, and target definitions
src/controlkit/policies/
                       Placeholder policy frontends for PID, LQR, MPC, and RL
src/controlkit/models/ Shared control-system model types
tests/                 Unit tests for scaffold behavior
examples/              Example policy specifications
docs/                  Architecture and usage notes
```

## Status

ControlKit is pre-alpha. See [roadmap.md](roadmap.md) and
[design_decisions.md](design_decisions.md) for the intended path.

