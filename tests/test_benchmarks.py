from __future__ import annotations

import json

from controlkit.benchmarks import BenchmarkConfig, benchmark_module
from controlkit.compiler.ir import IRModule
from controlkit.policies.lqr import LqrPolicy
from controlkit.policies.mpc import MpcPolicy


def _module() -> IRModule:
    frontend = LqrPolicy()
    spec = frontend.from_gain_matrix(
        name="bench_lqr",
        gain_matrix=[[1.0, 2.0]],
        saturation=(-10.0, 10.0),
    )
    return frontend.lower(spec)


def _mpc_module() -> IRModule:
    spec = MpcPolicy().from_matrices(
        name="bench_mpc",
        a_matrix=((1.0, 1.0), (0.0, 1.0)),
        b_matrix=((0.0,), (1.0,)),
        horizon=3,
        q_diagonal=(1.0, 0.1),
        r_diagonal=(0.05,),
        q_terminal_diagonal=(1.5, 0.2),
        u_min=(-0.5,),
        u_max=(0.5,),
        solver_iterations=4,
        step_size=0.1,
    )
    return MpcPolicy().lower(spec)


def test_benchmark_module_reports_python_latency_and_estimates() -> None:
    report = benchmark_module(
        _module(),
        BenchmarkConfig(iterations=20, warmup_iterations=5, include_c=False, include_rust=False),
    )

    assert report.module_name == "bench_lqr"
    assert report.operation_count_estimate > 0
    assert report.memory_footprint_bytes >= 16
    assert len(report.results) == 1
    assert report.results[0].name == "python"
    assert report.results[0].status == "ok"
    assert report.results[0].latency_ns_per_call is not None
    assert report.results[0].latency_ns_per_call > 0.0


def test_benchmark_report_serializes_json_and_markdown(tmp_path) -> None:
    report = benchmark_module(
        _module(),
        BenchmarkConfig(iterations=5, warmup_iterations=1, include_c=False, include_rust=False),
    )

    json_text = report.to_json()
    markdown_text = report.to_markdown()

    assert json.loads(json_text)["module_name"] == "bench_lqr"
    assert "| Implementation | Status | Iterations | Latency ns/call | Notes |" in markdown_text

    json_path = report.write_json(tmp_path / "report.json")
    markdown_path = report.write_markdown(tmp_path / "report.md")

    assert json_path.read_text(encoding="utf-8") == json_text
    assert markdown_path.read_text(encoding="utf-8") == markdown_text


def test_benchmark_module_includes_backend_results_or_skips() -> None:
    report = benchmark_module(
        _module(),
        BenchmarkConfig(iterations=3, warmup_iterations=1, include_c=True, include_rust=True),
    )

    result_by_name = {result.name: result for result in report.results}

    assert set(result_by_name) == {"python", "c", "rust"}
    assert result_by_name["python"].status == "ok"
    assert result_by_name["c"].status in {"ok", "skipped", "error"}
    assert result_by_name["rust"].status in {"ok", "skipped", "error"}


def test_benchmark_module_reports_mpc_python_reference() -> None:
    report = benchmark_module(
        _mpc_module(),
        BenchmarkConfig(iterations=5, warmup_iterations=1, include_c=False, include_rust=False),
    )

    assert report.module_name == "bench_mpc"
    assert report.operation_count_estimate > 0
    assert report.memory_footprint_bytes > 0
    assert len(report.results) == 1
    assert report.results[0].name == "python"
    assert report.results[0].status == "ok"
    assert "last_output=" in report.results[0].notes
