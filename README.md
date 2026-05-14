# ControlKit

ControlKit is a high-performance control-system compiler CLI. It compiles LQR, MPC-lite, and small
reinforcement-learning policies into deterministic C or Rust artifacts for embedded deployment.

[Website](website/index.html)

## Quick Start

Requirements:

- Python 3.11+
- `pip`
- `cc` for generated C benchmark/compile smoke tests
- `rustc` for generated Rust compile checks, optional

Create a virtual environment and install ControlKit in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Verify the install:

```bash
controlkit --help
pytest
```

The installed console command and Python module entrypoint are equivalent:

```bash
controlkit version
python -m controlkit version
```

From a fresh checkout without editable install, run the module with `PYTHONPATH`:

```bash
PYTHONPATH=src python -m controlkit --help
```

## Example Run

The fastest way to see the toolchain is to run one of the shipped controller specs through the full
pipeline.

Validate the YAML spec:

```bash
controlkit validate examples/lqr_cartpole.yaml
```

Inspect the compiler module that ControlKit builds from it:

```bash
controlkit inspect examples/lqr_cartpole.yaml
```

Compile the controller to C:

```bash
controlkit compile examples/lqr_cartpole.yaml --target c --output build/lqr_c
```

Benchmark the Python reference against the generated artifact:

```bash
controlkit benchmark examples/lqr_cartpole.yaml --output build/lqr_bench --no-rust
```

Expected outputs:

```text
build/lqr_c/lqr_cartpole.h
build/lqr_c/lqr_cartpole.c
build/lqr_bench/lqr_cartpole.json
build/lqr_bench/lqr_cartpole.md
```

You can run the same CLI through Python's module entrypoint:

```bash
python -m controlkit inspect examples/lqr_cartpole.yaml
python -m controlkit compile examples/lqr_cartpole.yaml --target rust --output build/lqr_rust
```

The same flow works for MPC-lite and RL policies:

```bash
controlkit inspect examples/mpc_temperature.yaml
controlkit compile examples/mpc_temperature.yaml --target c --output build/mpc_c
controlkit benchmark examples/mpc_temperature.yaml --output build/mpc_bench --no-rust

controlkit inspect examples/rl_balance.yaml
controlkit compile examples/rl_balance.yaml --target c --output build/rl_c
controlkit benchmark examples/rl_balance.yaml --output build/rl_bench --no-rust
```

## Write Your Own Controller

ControlKit specs are YAML files. Start with a small LQR-style feedback controller if you want the
shortest path from a control law to generated C.

Create `my_room_lqr.yaml`:

```yaml
policy: lqr
name: room_temperature
state_dim: 2
control_dim: 1
state_name: x
control_name: u
K:
  - [0.75, 0.18]
u_min: [-1.0]
u_max: [1.0]
```

This represents:

```text
x[0] = temperature error
x[1] = error-rate estimate
u[0] = clipped heater/cooler command
u = clip(-Kx, -1.0, 1.0)
```

Run it end to end:

```bash
controlkit validate my_room_lqr.yaml
controlkit inspect my_room_lqr.yaml
controlkit compile my_room_lqr.yaml --target c --output build/room_c
controlkit benchmark my_room_lqr.yaml --output build/room_bench --no-rust
```

The generated C exposes a fixed interface:

```c
void control_step(
    const float x[CONTROLKIT_STATE_DIM],
    float u[CONTROLKIT_CONTROL_DIM]
);
```

For Rust output, change the target:

```bash
controlkit compile my_room_lqr.yaml --target rust --output build/room_rust
```

### MPC-lite Spec Shape

Use `policy: mpc` when you have discrete dynamics, diagonal costs, a finite horizon, and box
constraints on the control input:

```yaml
policy: mpc
name: temperature_mpc
state_dim: 2
control_dim: 1
horizon: 6
A:
  - [1.0, 0.1]
  - [0.0, 0.98]
B:
  - [0.0]
  - [0.05]
q: [1.0, 0.2]
r: [0.05]
u_min: [-1.0]
u_max: [1.0]
solver_iterations: 8
step_size: 0.05
```

### RL Policy Spec Shape

Use `policy: rl` for a small fixed-shape MLP policy. The YAML points to a JSON weight file:

```yaml
policy: rl
name: balance_policy
input_dim: 2
output_dim: 1
weights_path: rl_balance_weights.json
```

The weights JSON contains ordered layers:

```json
{
  "input_dim": 2,
  "output_dim": 1,
  "layers": [
    {
      "type": "linear",
      "weights": [[0.4, -0.2], [0.1, 0.3]],
      "bias": [0.0, 0.1]
    },
    {"type": "relu"},
    {
      "type": "linear",
      "weights": [[0.7, -0.5]],
      "bias": [0.02]
    },
    {"type": "tanh"}
  ]
}
```

Then run:

```bash
controlkit validate balance_policy.yaml
controlkit compile balance_policy.yaml --target c --output build/balance_c
controlkit benchmark balance_policy.yaml --output build/balance_bench --no-rust
```

## Current CLI

ControlKit supports LQR, MPC-lite, and RL MLP YAML specs. MPC-lite specs use inline arrays for
discrete dynamics, diagonal costs, finite horizons, and input box constraints. RL specs point to
dependency-free JSON weight files for fixed-shape MLP inference. PID YAML lowering is planned.

Useful commands:

```bash
controlkit version
controlkit validate examples/lqr_cartpole.yaml
controlkit inspect examples/lqr_cartpole.yaml
controlkit compile examples/lqr_cartpole.yaml --target c --output build/controlkit_c
controlkit compile examples/lqr_cartpole.yaml --target rust --output build/controlkit_rust
controlkit benchmark examples/lqr_cartpole.yaml --output build/benchmarks
PYTHONPATH=src python examples/c_backend_lqr.py
PYTHONPATH=src python examples/rust_backend_lqr.py
PYTHONPATH=src python examples/optimization_pass.py
PYTHONPATH=src python examples/benchmark_lqr.py
```

The C backend is available through `controlkit.backends.CBackend`. The Rust backend is available as
`controlkit.backends.RustBackend` and emits no-std-compatible fixed-array Rust source.

The optimizer is available through `controlkit.optimization.optimize_module`. It performs
conservative constant folding and algebraic simplification while reporting rough operation counts.

The benchmark runner is available through `controlkit.benchmarks.benchmark_module`. It measures
Python reference latency, generated C latency when `cc` is available, generated Rust latency when
`rustc` is available, and writes JSON/Markdown reports.

MPC-lite lowers finite-horizon linear MPC controllers into a first-class IR node. C and Rust
backends emit a stateless projected-gradient solver that returns the first control input from a
zero-initialized control sequence.

The RL frontend lowers small MLP policies into a first-class IR node. C uses `tanhf` for Tanh
activations, while Rust keeps `#![no_std]` with a small deterministic Tanh approximation.

## Phase Timeline

| Phase | Focus |
| --- | --- |
| 1 | ControlKit IR for expressions, control laws, shape checks, and linear systems |
| 2 | LQR frontend that lowers YAML/Python specs into IR |
| 3 | Standalone C backend for generated controller code |
| 4 | Rust backend with fixed-size arrays and `#![no_std]`-oriented output |
| 5 | Symbolic simplification and lightweight optimization passes |
| 6 | Latency benchmarking, operation counts, and report generation |
| 7 | CLI commands for validate, inspect, compile, benchmark, and YAML specs |
| 8 | MPC-lite frontend and projected-gradient generated solvers |
| 9 | RL MLP policy compilation from dependency-free JSON weights |
| 10 | Product website, demo walkthrough, and public polish |

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
