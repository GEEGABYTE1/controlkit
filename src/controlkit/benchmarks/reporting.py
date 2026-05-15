"""Markdown and JSON output for benchmark suite runs."""

from __future__ import annotations

import json
from pathlib import Path

from controlkit.benchmarks.metrics import ClosedLoopMetrics


def write_benchmark_outputs(
    *,
    output_dir: Path,
    metrics: ClosedLoopMetrics,
    description: str,
    dynamics_equations: str,
    controller_description: str,
    limitations: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n")
    report_path.write_text(
        render_benchmark_markdown(
            metrics=metrics,
            description=description,
            dynamics_equations=dynamics_equations,
            controller_description=controller_description,
            limitations=limitations,
        ),
        encoding="utf-8",
    )
    return json_path, report_path


def render_benchmark_markdown(
    *,
    metrics: ClosedLoopMetrics,
    description: str,
    dynamics_equations: str,
    controller_description: str,
    limitations: str,
) -> str:
    status = "PASS" if metrics.passed else "FAIL"
    generated = (
        "n/a"
        if metrics.generated_mean_runtime_us is None
        else f"{metrics.generated_mean_runtime_us:.3f}"
    )
    return "\n".join(
        [
            f"# Benchmark Report: {metrics.benchmark_name}",
            "",
            f"**Result:** {status}",
            "",
            "## Description",
            "",
            description,
            "",
            "## Dynamics",
            "",
            dynamics_equations,
            "",
            "## Controller",
            "",
            controller_description,
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| dt | {metrics.dt:g} |",
            f"| horizon steps | {metrics.horizon_steps} |",
            f"| mean runtime us | {metrics.mean_runtime_us:.3f} |",
            f"| max runtime us | {metrics.max_runtime_us:.3f} |",
            f"| p95 runtime us | {metrics.p95_runtime_us:.3f} |",
            f"| generated mean runtime us | {generated} |",
            f"| final state norm | {metrics.final_state_norm:.6g} |",
            f"| max state norm | {metrics.max_state_norm:.6g} |",
            f"| total control effort | {metrics.total_control_effort:.6g} |",
            "",
            "## Pass/Fail",
            "",
            f"- Passed: `{str(metrics.passed).lower()}`",
            f"- Failure reason: {metrics.failure_reason or 'none'}",
            "",
            "## Limitations",
            "",
            limitations,
            "",
        ]
    )
