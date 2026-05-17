# ControlKit v1 Improvements

This update adds two engineering-oriented layers: a real benchmark suite and a controller
verification layer.

## Benchmark Cases

- `double_integrator_lqr`
- `cartpole_lqr_linearized`
- `rocket_hover_lqr`
- `pid_mass_spring_damper`

Each benchmark includes a problem description, deterministic model, controller spec, local runner,
expected results, JSON output, and Markdown report.

## Verification Checks

- Matrix dimension validity.
- Continuous and discrete closed-loop stability.
- Constraint sanity checks.
- NaN/Inf detection.
- Condition number warnings.
- Stability margin reporting.
- Generated-code consistency hook for future target execution integration.

## Why This Matters

ControlKit now has a clearer path from controller spec to deployable artifact:

1. Validate and verify the controller.
2. Compile the controller to target code.
3. Benchmark the reference and generated implementation.
4. Inspect machine-readable JSON and human-readable Markdown reports.

## Future Work

- Fixed-point support.
- Hardware-in-the-loop simulation.
- STM32 timing harnesses.
- MPC-specific verification.
- Generated-code execution adapters for broader target backends.
