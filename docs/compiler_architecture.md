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

