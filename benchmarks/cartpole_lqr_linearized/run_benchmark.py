from __future__ import annotations

from pathlib import Path

from controlkit.benchmarks import run_benchmark_case


if __name__ == "__main__":
    result = run_benchmark_case(Path(__file__).with_name("controller.yaml"))
    print(result.to_dict())
