"""Benchmark a small LQR controller and write JSON/Markdown reports.

Run from the repository root with:

    PYTHONPATH=src python examples/benchmark_lqr.py
"""

from __future__ import annotations

from pathlib import Path

from controlkit.benchmarks import BenchmarkConfig, benchmark_module
from controlkit.policies.lqr import LqrPolicy


def main() -> None:
    frontend = LqrPolicy()
    spec = frontend.from_gain_matrix(
        name="cartpole_lqr",
        gain_matrix=[[1.2, 0.4, 2.5, 0.8]],
        saturation=(-1.0, 1.0),
    )
    report = benchmark_module(
        frontend.lower(spec),
        BenchmarkConfig(iterations=10_000, warmup_iterations=1_000),
    )
    output_dir = Path("build/benchmarks")
    json_path = report.write_json(output_dir / "cartpole_lqr.json")
    markdown_path = report.write_markdown(output_dir / "cartpole_lqr.md")

    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
