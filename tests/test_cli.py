from __future__ import annotations

from pathlib import Path

from controlkit.cli import main


def test_version_command_prints_version(capsys) -> None:
    assert main(["version"]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"


def test_inspect_reports_readable_file(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "pid.yaml"
    spec.write_text("policy: pid\n", encoding="utf-8")

    assert main(["inspect", str(spec)]) == 0

    captured = capsys.readouterr()
    assert f"spec: {spec}" in captured.out
    assert "status: readable" in captured.out


def test_compile_reports_placeholder_status(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "pid.yaml"
    spec.write_text("policy: pid\n", encoding="utf-8")

    assert main(["compile", str(spec), "--policy", "pid", "--target", "c"]) == 0

    captured = capsys.readouterr()
    assert "compilation is not implemented yet" in captured.out
    assert "validated pid policy for c target" in captured.out

