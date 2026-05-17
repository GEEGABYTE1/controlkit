# ControlKit Demo Walkthrough

This demo shows the current end-to-end compiler path.

## 1. Install

```bash
python -m pip install -e ".[dev]"
```

## 2. Inspect A Controller

```bash
controlkit inspect examples/lqr_cartpole.yaml
controlkit inspect examples/mpc_temperature.yaml
controlkit inspect examples/rl_balance.yaml
```

## 3. Generate C

```bash
controlkit compile examples/lqr_cartpole.yaml --target c --output build/demo_lqr_c
controlkit compile examples/mpc_temperature.yaml --target c --output build/demo_mpc_c
controlkit compile examples/rl_balance.yaml --target c --output build/demo_rl_c
```

## 4. Generate Rust

```bash
controlkit compile examples/lqr_cartpole.yaml --target rust --output build/demo_lqr_rust
controlkit compile examples/mpc_temperature.yaml --target rust --output build/demo_mpc_rust
controlkit compile examples/rl_balance.yaml --target rust --output build/demo_rl_rust
```

## 5. Benchmark

```bash
controlkit benchmark examples/lqr_cartpole.yaml --output build/demo_benchmarks --no-rust
controlkit benchmark examples/mpc_temperature.yaml --output build/demo_benchmarks --no-rust
controlkit benchmark examples/rl_balance.yaml --output build/demo_benchmarks --no-rust
```

## 6. Open The Website

Open [https://controlkit.vercel.app/](https://controlkit.vercel.app/) in a browser. The page
presents ControlKit as a product-style technical project and links to the GitHub repository:

[https://github.com/GEEGABYTE1/controlkit](https://github.com/GEEGABYTE1/controlkit)
