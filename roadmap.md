# Roadmap

## Phase 0: Scaffold

- Package structure
- CLI entrypoint
- Tests
- Examples
- Docs and project logs

## Phase 1: ControlKit IR

- Typed IR nodes for vectors, matrices, scalar constants, operations, control laws, and linear
  systems.
- Shape validation and readable representations.
- Phase 1 textbook content.

## Phase 2: LQR Frontend

- User-facing LQR API from gain matrix.
- State/control dimension validation.
- Optional saturation.
- Optional state/control naming.
- Lowering into ControlKit IR.

## Phase 3: C Backend

- Generate `.c` and `.h` outputs.
- Support float32 `control_step`.
- Lower matrix-vector multiplication and saturation.
- Add deterministic formatting and generated-code tests.
- Compile generated C as a smoke check.

## Phase 4: Rust Backend

- Generate Rust code with fixed-size arrays.
- Favor `no_std`-compatible style.
- Add snapshot tests.
- Add optional generated-source compile check when `rustc` is installed.

## Phase 5: Symbolic Simplification and Optimization

- Constant folding.
- Algebraic simplification.
- Dead computation elimination.
- Optional loop unrolling for small controllers.
- Operation count estimates before and after optimization.

## Phase 6: Latency Benchmarking

- Python reference latency.
- Generated C and Rust latency where available.
- Operation count and memory footprint estimates.
- JSON and Markdown reports.
- Toolchain-aware skips when `cc` or `rustc` are unavailable.

## Phase 7: CLI

- `controlkit compile`
- `controlkit inspect`
- `controlkit benchmark`
- `controlkit validate`
- YAML controller specs.
- LQR YAML lowering into IR.
- C/Rust artifact generation from the CLI.

## Phase 8: MPC-lite

- Linear dynamics `x_next = Ax + Bu`.
- Diagonal quadratic stage and terminal costs.
- Finite horizon.
- Box constraints on `u`.
- First-class MPC IR node.
- Projected-gradient Python reference solver.
- Generated C and Rust stateless solver.
- CLI validation, inspection, compilation, and benchmarking for MPC YAML specs.

## Phase 9: RL Policy Compilation

- MLP policies with Linear, ReLU, and Tanh.
- Fixed input/output sizes.
- JSON weight files.
- First-class RL policy IR node.
- Python reference inference.
- Generated C and Rust inference code.
- CLI validation, inspection, compilation, and benchmarking for RL YAML specs.

## Phase 10: Blog and Demo Polish

- `blog/controlkit_v1.md`.
- Product-style static website that links to the GitHub repository.
- Benchmark summary.
- Architecture diagram.
- Demo walkthrough.
- Final textbook cleanup.
