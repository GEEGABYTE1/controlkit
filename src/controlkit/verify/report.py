"""Verification report orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from controlkit.frontend.specs import _parse_simple_yaml
from controlkit.verify.constraints import check_constraints
from controlkit.verify.dimensions import CheckResult, verify_dimensions
from controlkit.verify.numerical import check_numerical_robustness
from controlkit.verify.stability import StabilityResult, check_closed_loop_stability


@dataclass(frozen=True)
class VerificationFileReport:
    controller_name: str
    system_type: str
    passed: bool
    checks: list[dict[str, object]]
    warnings: list[str]
    errors: list[str]
    stability_margin: float | None
    eigenvalues: list[dict[str, float]]
    condition_numbers: dict[str, float]
    recommendations: list[str]
    json_path: Path
    markdown_path: Path


def verify_controller_file(path: Path, output_dir: Path = Path("outputs/verification")) -> VerificationFileReport:
    raw = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    name = str(raw.get("name", path.stem))
    system_type = str(raw.get("system_type", "discrete"))
    a_matrix = _matrix(raw.get("a_matrix"))
    b_matrix = _matrix(raw.get("b_matrix"))
    gain_matrix = _matrix(raw.get("gain_matrix"))
    q_matrix = _matrix(raw.get("q_matrix"))
    r_matrix = _matrix(raw.get("r_matrix"))

    dimension_checks = verify_dimensions(
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        gain_matrix=gain_matrix,
        q_matrix=q_matrix,
        r_matrix=r_matrix,
    )
    warnings: list[str] = []
    errors: list[str] = [check.message for check in dimension_checks if not check.passed]
    stability: StabilityResult | None = None
    if a_matrix is not None and b_matrix is not None and gain_matrix is not None:
        try:
            stability = check_closed_loop_stability(
                a_matrix=a_matrix,
                b_matrix=b_matrix,
                gain_matrix=gain_matrix,
                system_type=system_type,
            )
            if not stability.passed:
                errors.append(stability.message)
            if stability.stability_margin < 0.05:
                warnings.append("closed-loop eigenvalues are close to the stability boundary")
        except Exception as exc:
            errors.append(f"stability check failed: {exc}")
    else:
        warnings.append("stability check skipped because A, B, or K is missing")

    input_lower, input_upper, saturation_declared = _input_bounds(raw)
    constraint_result = check_constraints(
        gain_matrix=gain_matrix,
        input_lower=input_lower,
        input_upper=input_upper,
        state_lower=_vector(raw.get("state_min")),
        state_upper=_vector(raw.get("state_max")),
        saturation_declared=saturation_declared,
    )
    warnings.extend(constraint_result.warnings)
    errors.extend(constraint_result.errors)

    matrices = {
        key: value
        for key, value in {
            "A": a_matrix,
            "B": b_matrix,
            "K": gain_matrix,
            "Q": q_matrix,
            "R": r_matrix,
        }.items()
        if value is not None
    }
    numerical_result = check_numerical_robustness(matrices)
    warnings.extend(numerical_result.warnings)
    errors.extend(numerical_result.errors)

    checks = [_check_to_dict(check) for check in dimension_checks]
    checks.append({"name": "constraints", "passed": constraint_result.passed, "message": ""})
    checks.append({"name": "numerical robustness", "passed": numerical_result.passed, "message": ""})
    if stability is not None:
        checks.append({"name": "closed-loop stability", "passed": stability.passed, "message": stability.message})

    recommendations = _recommendations(errors, warnings)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}_verification.json"
    markdown_path = output_dir / f"{name}_verification.md"
    report = VerificationFileReport(
        controller_name=name,
        system_type=system_type,
        passed=not errors,
        checks=checks,
        warnings=warnings,
        errors=errors,
        stability_margin=None if stability is None else stability.stability_margin,
        eigenvalues=[] if stability is None else _eigen_to_json(stability.eigenvalues),
        condition_numbers=numerical_result.condition_numbers,
        recommendations=recommendations,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    json_path.write_text(_report_json(report), encoding="utf-8")
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    return report


def _report_json(report: VerificationFileReport) -> str:
    data = asdict(report)
    data["json_path"] = str(report.json_path)
    data["markdown_path"] = str(report.markdown_path)
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _report_markdown(report: VerificationFileReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Verification Report: {report.controller_name}",
        "",
        f"**Result:** {status}",
        "",
        "## System Summary",
        "",
        f"- System type: `{report.system_type}`",
        f"- Stability margin: `{report.stability_margin}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Message |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        lines.append(f"| {check['name']} | {check['passed']} | {check['message']} |")
    lines.extend(["", "## Eigenvalues", ""])
    if report.eigenvalues:
        for value in report.eigenvalues:
            lines.append(f"- `{value['real']:.6g} + {value['imag']:.6g}j`")
    else:
        lines.append("- n/a")
    lines.extend(["", "## Condition Numbers", ""])
    if report.condition_numbers:
        for name, value in report.condition_numbers.items():
            lines.append(f"- {name}: `{value:.6g}`")
    else:
        lines.append("- n/a")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.recommendations)
    lines.append("")
    return "\n".join(lines)


def _check_to_dict(check: CheckResult) -> dict[str, object]:
    return {"name": check.name, "passed": check.passed, "message": check.message}


def _matrix(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("matrix values must be lists")
    return [[float(item) for item in row] for row in value]


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("vector values must be lists")
    return [float(item) for item in value]


def _input_bounds(raw: dict[str, Any]) -> tuple[list[float] | None, list[float] | None, bool]:
    if isinstance(raw.get("saturation"), dict):
        lower = float(raw["saturation"]["lower"])
        upper = float(raw["saturation"]["upper"])
        control_dim = int(raw.get("control_dim", 1))
        return [lower] * control_dim, [upper] * control_dim, True
    lower = _vector(raw.get("u_min"))
    upper = _vector(raw.get("u_max"))
    return lower, upper, lower is not None and upper is not None


def _eigen_to_json(values: list[complex]) -> list[dict[str, float]]:
    return [{"real": value.real, "imag": value.imag} for value in values]


def _recommendations(errors: list[str], warnings: list[str]) -> list[str]:
    if errors:
        return [
            "Fix failing verification checks before generating deployment artifacts.",
            "Re-run `controlkit verify` after changing controller gains or constraints.",
        ]
    if warnings:
        return [
            "Review warnings before deploying to constrained hardware.",
            "Consider adding wider validation sweeps for expected operating states.",
        ]
    return ["Controller passed static verification checks; continue with benchmark and target tests."]
