# ControlKit: Building a Compiler for Control Systems

ControlKit is a compiler-shaped workflow for embedded controllers. Instead of treating controller
deployment as a manual rewrite from research code into firmware code, ControlKit lowers controller
specifications into a typed intermediate representation, generates deterministic C or Rust, and
benchmarks the result.

Repository: [https://github.com/GEEGABYTE1/controlkit](https://github.com/GEEGABYTE1/controlkit)

## 1. Why Deploying Controllers Is Painful

Control algorithms often start in simulation, notebooks, or high-level engineering tools. The hard
part begins when the controller needs to run on an embedded target:

- matrix dimensions must stay correct
- saturation and constraints must match the design
- generated code must avoid hidden allocation
- runtime latency must fit the control loop
- validation needs to compare generated code against a reference

ControlKit treats those requirements as compiler contracts.

## 2. Why MATLAB/Simulink-Style Workflows Can Feel Heavy

Graphical and monolithic toolchains can be powerful, but they can also make lightweight iteration
hard. ControlKit is intentionally small: specs, IR, codegen, tests, and benchmarks are all plain
repo artifacts.

The goal is not to replace every industrial workflow. The goal is to make a transparent compiler
path for small embedded controllers.

## 3. Control Laws as Compiler IR

The ControlKit pipeline is:

```text
controller spec -> frontend -> IR -> optimization -> backend -> validation and benchmarks
```

LQR controllers lower to expression trees such as `u = -Kx`. MPC-lite controllers lower to a
structured finite-horizon solver node. RL policies lower to fixed-shape MLP layer nodes.

That shared IR boundary lets the rest of the toolchain operate consistently.

## 4. LQR Example

Inspect an LQR spec:

```bash
controlkit inspect examples/lqr_cartpole.yaml
```

Compile it to C:

```bash
controlkit compile examples/lqr_cartpole.yaml --target c --output build/lqr_c
```

The generated C exposes a fixed function interface:

```c
void lqr_cartpole_control_step(
    const float x[CONTROLKIT_STATE_DIM],
    float u[CONTROLKIT_CONTROL_DIM]
);
```

## 5. C/Rust Code Generation

The C backend emits standalone `.h` and `.c` files. The Rust backend emits a single
`#![no_std]`-friendly `.rs` file with fixed arrays.

Compile MPC-lite to Rust:

```bash
controlkit compile examples/mpc_temperature.yaml --target rust --output build/mpc_rust
```

Compile the RL MLP example to C:

```bash
controlkit compile examples/rl_balance.yaml --target c --output build/rl_c
```

## 6. Symbolic Optimization

ControlKit includes conservative symbolic simplification:

- constant folding
- `x + 0 -> x`
- `x * 0 -> 0`
- `x * 1 -> x`
- optional backend loop unrolling

The optimizer preserves IR invariants by constructing validated IR nodes after rewrites.

## 7. Latency Benchmarks

Benchmark reports include Python reference latency, generated C latency when `cc` is available,
generated Rust latency when `rustc` is available, operation-count estimates, and memory estimates.

Example:

```bash
controlkit benchmark examples/rl_balance.yaml --output build/benchmarks --no-rust
```

Current smoke results recorded in `benchmark_log.md` include:

| Example | Python reference | Generated C |
| --- | ---: | ---: |
| `lqr_cartpole` | 2145.84 ns/call | 10.00 ns/call |
| `mpc_temperature` | 37710.29 ns/call | 33.00 ns/call |
| `rl_balance` | 2693.08 ns/call | 6.00 ns/call |

## 8. RL Policy Compilation

Phase 9 added dependency-free JSON weights for small MLP policies. Supported layers are:

- `linear`
- `relu`
- `tanh`

C uses `tanhf`. Rust stays `#![no_std]` by emitting a deterministic Tanh approximation.

## 9. Roadmap

ControlKit is pre-alpha. The major compiler path is now visible:

- typed IR
- LQR frontend
- C backend
- Rust backend
- symbolic optimization
- benchmarking
- CLI
- MPC-lite
- RL MLP compilation

Next work is product/demo polish, richer examples, and future controller families such as PID
lowering and broader model import support.
