# Compiler Architecture

The intended pipeline is:

1. Parse a policy specification.
2. Validate the policy schema and numeric assumptions.
3. Lower the policy into ControlKit IR.
4. Run optimization passes for the chosen embedded target.
5. Generate C or Rust.
6. Verify generated artifacts with simulation fixtures and target-specific tests.

## Frontends

Initial frontends are PID, LQR, MPC, and RL. Each frontend should own policy-specific parsing,
validation, and lowering.

## IR

The IR should be backend-neutral and explicit about timing, state shape, numeric precision,
saturation, constraints, and memory layout.

## Backends

Backends should expose small target specs for C and Rust. Embedded constraints such as `no_std`,
fixed-point arithmetic, stack allocation, and deterministic execution should be first-class
configuration, not incidental backend behavior.

The Phase 3 C backend currently generates deterministic standalone `.h` and `.c` files for a
single supported control law, using float arrays, explicit matrix-vector loops, and scalar
saturation branches.

The Phase 4 Rust backend generates deterministic `.rs` files with `#![no_std]`, fixed-size arrays,
a mutable-output `control_step` function, matrix-vector loops, and scalar saturation branches.

## Optimization

The Phase 5 optimizer runs over ControlKit IR before backend generation. It performs conservative
constant folding and algebraic simplification, reports estimated operation counts, and leaves loop
unrolling as an explicit backend option for C and Rust.

## Benchmarking

The Phase 6 benchmark runner measures the Python reference evaluator and any generated backends
available in the local toolchain. Reports include latency in nanoseconds per call, operation-count
estimates, memory-footprint estimates, and JSON/Markdown serialization.

## CLI

The Phase 7 CLI loads controller YAML specs, lowers supported specs into IR, and exposes
`inspect`, `validate`, `compile`, and `benchmark` commands. LQR YAML specs lower to ordinary
control-law expression trees. Phase 8 MPC-lite YAML specs lower to a first-class finite-horizon MPC
IR node with discrete dynamics, diagonal costs, input box constraints, and projected-gradient
solver settings. Phase 9 RL YAML specs point at JSON MLP weights and lower to a first-class RL
policy IR node.

## MPC-lite

Phase 8 supports small embedded-friendly MPC controllers for discrete linear systems
`x_next = Ax + Bu`. Generated C and Rust initialize a horizon-length control sequence to zero on
each call, run a fixed number of projected-gradient iterations, and write only the first input to
the `control_step` output buffer.

## RL Policy Compilation

Phase 9 supports dependency-free fixed-shape MLP policies with `linear`, `relu`, and `tanh` layers.
The Python frontend validates JSON weights, the IR stores typed layers, and generated C/Rust expose
the same `control_step` array interface used by the other controller families.
