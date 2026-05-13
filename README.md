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
controlkit inspect examples/lqr_cartpole.yaml
controlkit validate examples/lqr_cartpole.yaml
controlkit compile examples/lqr_cartpole.yaml --target c --output build/controlkit_c
controlkit compile examples/lqr_cartpole.yaml --target rust --output build/controlkit_rust
controlkit benchmark examples/lqr_cartpole.yaml --output build/benchmarks
controlkit inspect examples/mpc_temperature.yaml
controlkit compile examples/mpc_temperature.yaml --target c --output build/mpc_c
controlkit compile examples/mpc_temperature.yaml --target rust --output build/mpc_rust
controlkit inspect examples/rl_balance.yaml
controlkit compile examples/rl_balance.yaml --target c --output build/rl_c
controlkit compile examples/rl_balance.yaml --target rust --output build/rl_rust
PYTHONPATH=src python examples/c_backend_lqr.py
PYTHONPATH=src python examples/rust_backend_lqr.py
PYTHONPATH=src python examples/optimization_pass.py
PYTHONPATH=src python examples/benchmark_lqr.py
```

The Phase 9 CLI supports LQR, MPC-lite, and RL MLP YAML specs. MPC-lite specs use inline arrays
for discrete dynamics, diagonal costs, finite horizons, and input box constraints. RL specs point
to dependency-free JSON weight files for fixed-shape MLP inference. PID YAML lowering is still
planned for later.

The Phase 3 C backend is available as a Python API through `controlkit.backends.CBackend`. The CLI
will be wired to this backend in a later CLI-focused phase. The Phase 4 Rust backend is available as
`controlkit.backends.RustBackend` and emits no-std-compatible fixed-array Rust source.

The Phase 5 optimizer is available through `controlkit.optimization.optimize_module`. It performs
conservative constant folding and algebraic simplification while reporting rough operation counts.

The Phase 6 benchmark runner is available through `controlkit.benchmarks.benchmark_module`. It
measures Python reference latency, generated C latency when `cc` is available, generated Rust
latency when `rustc` is available, and writes JSON/Markdown reports.

The Phase 8 MPC-lite frontend lowers finite-horizon linear MPC controllers into a first-class IR
node. C and Rust backends emit a stateless projected-gradient solver that returns the first control
input from a zero-initialized control sequence.

The Phase 9 RL frontend lowers small MLP policies into a first-class IR node. C uses `tanhf` for
Tanh activations, while Rust keeps `#![no_std]` with a small deterministic Tanh approximation.

## Website And Blog

Phase 10 adds a static product-style website and technical blog:

- [Product website](website/index.html)
- [Technical blog](blog/controlkit_v1.md)
- [Demo walkthrough](docs/demo_walkthrough.md)
- [Benchmark summary](docs/benchmark_summary.md)

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
