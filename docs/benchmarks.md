# Benchmarks

ControlKit benchmarks are designed to test controller deployment behavior, not just isolated Python
function speed. Each case combines a simple deterministic plant, a controller spec, closed-loop
simulation, runtime timing, and pass/fail criteria.

## Run One Benchmark

```bash
controlkit benchmark benchmarks/double_integrator_lqr/controller.yaml
```

Outputs are written to:

```text
outputs/benchmarks/double_integrator_lqr/results.json
outputs/benchmarks/double_integrator_lqr/report.md
```

## Run All Benchmarks

```bash
controlkit benchmark --all
```

## Metrics

- `mean_runtime_us`: mean Python/reference controller call time.
- `max_runtime_us`: slowest measured controller call.
- `p95_runtime_us`: 95th percentile controller call time.
- `final_state_norm`: Euclidean norm of the final simulated state.
- `max_state_norm`: largest state norm seen during simulation.
- `total_control_effort`: sum of absolute control effort over time.
- `generated_mean_runtime_us`: generated C timing when the controller can be compiled and `cc` is available.
- `passed`: whether the benchmark met its configured limits.

## Add a Benchmark

Create a folder under `benchmarks/`:

```text
benchmarks/my_case/
  problem.md
  model.py
  controller.yaml
  run_benchmark.py
  expected_results.md
```

`model.py` must define:

- `BENCHMARK`: metadata, `dt`, `horizon_steps`, `initial_state`, and pass criteria.
- `step(state, control, dt)`: deterministic dynamics update.

Keep benchmark plants small and deterministic. The suite should stress the compiler/control stack,
not hide correctness behind a large simulator.
