# Benchmark Summary

Current smoke benchmarks are recorded in `benchmark_log.md`. They are local machine checks, not
cross-platform claims.

| Example | Policy | Python reference | Generated C | Rust |
| --- | --- | ---: | ---: | --- |
| `lqr_cartpole` | LQR | 2145.84 ns/call | 10.00 ns/call | skipped |
| `mpc_temperature` | MPC-lite | 37710.29 ns/call | 33.00 ns/call | skipped |
| `rl_balance` | RL MLP | 2693.08 ns/call | 6.00 ns/call | skipped |

The benchmark runner also reports operation-count and memory-footprint estimates for each module.
