from __future__ import annotations

from pathlib import Path

import pytest

from controlkit.cli import main


def _write_lqr_spec(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "name: cli_lqr",
                "policy: lqr",
                "state_dim: 2",
                "control_dim: 1",
                "gain_matrix:",
                "  - [1.0, 2.0]",
                "saturation:",
                "  lower: -1.0",
                "  upper: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_mpc_spec(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "name: cli_mpc",
                "policy: mpc",
                "state_dim: 2",
                "control_dim: 1",
                "horizon: 3",
                "a_matrix:",
                "  - [1.0, 1.0]",
                "  - [0.0, 1.0]",
                "b_matrix:",
                "  - [0.0]",
                "  - [1.0]",
                "q_diagonal: [1.0, 0.1]",
                "r_diagonal: [0.05]",
                "q_terminal_diagonal: [1.5, 0.2]",
                "u_min: [-0.5]",
                "u_max: [0.5]",
                "solver_iterations: 4",
                "step_size: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_rl_spec(path: Path) -> None:
    (path.parent / "rl_weights.json").write_text(
        "\n".join(
            [
                "{",
                '  "input_dim": 2,',
                '  "output_dim": 1,',
                '  "layers": [',
                '    {"type": "linear", "weights": [[0.5, -0.25], [0.1, 0.4]],',
                '      "bias": [0.05, -0.02]},',
                '    {"type": "relu"},',
                '    {"type": "linear", "weights": [[0.8, -0.6]], "bias": [-0.05]},',
                '    {"type": "tanh"}',
                "  ]",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.write_text(
        "\n".join(
            [
                "name: cli_rl",
                "policy: rl",
                "weights_path: rl_weights.json",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_version_command_prints_version(capsys) -> None:
    assert main(["version"]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"


def test_inspect_reports_readable_file(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "lqr.yaml"
    _write_lqr_spec(spec)

    assert main(["inspect", str(spec)]) == 0

    captured = capsys.readouterr()
    assert f"spec: {spec}" in captured.out
    assert "policy: lqr" in captured.out
    assert "module: cli_lqr" in captured.out
    assert "control_laws: 1" in captured.out


def test_validate_reports_valid_spec(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "lqr.yaml"
    _write_lqr_spec(spec)

    assert main(["validate", str(spec)]) == 0

    captured = capsys.readouterr()
    assert f"valid: {spec}" in captured.out
    assert "policy: lqr" in captured.out


def test_compile_writes_c_artifacts(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "lqr.yaml"
    output = tmp_path / "generated"
    _write_lqr_spec(spec)

    assert main(["compile", str(spec), "--target", "c", "--output", str(output)]) == 0

    captured = capsys.readouterr()
    assert str(output / "cli_lqr.h") in captured.out
    assert str(output / "cli_lqr.c") in captured.out
    assert (output / "cli_lqr.h").exists()
    assert (output / "cli_lqr.c").exists()


def test_compile_writes_rust_artifact_with_unroll(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "lqr.yaml"
    output = tmp_path / "generated"
    _write_lqr_spec(spec)

    assert (
        main(["compile", str(spec), "--target", "rust", "--output", str(output), "--unroll-loops"])
        == 0
    )

    captured = capsys.readouterr()
    assert str(output / "cli_lqr.rs") in captured.out
    assert "while row <" not in (output / "cli_lqr.rs").read_text(encoding="utf-8")


def test_benchmark_writes_reports(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "lqr.yaml"
    output = tmp_path / "benchmarks"
    _write_lqr_spec(spec)

    assert (
        main(
            [
                "benchmark",
                str(spec),
                "--output",
                str(output),
                "--iterations",
                "5",
                "--warmup-iterations",
                "1",
                "--no-c",
                "--no-rust",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert str(output / "cli_lqr.json") in captured.out
    assert str(output / "cli_lqr.md") in captured.out
    assert (output / "cli_lqr.json").exists()
    assert (output / "cli_lqr.md").exists()


def test_mpc_validate_inspect_compile_and_benchmark(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "mpc.yaml"
    generated = tmp_path / "generated"
    benchmarks = tmp_path / "benchmarks"
    _write_mpc_spec(spec)

    assert main(["validate", str(spec)]) == 0
    assert "policy: mpc" in capsys.readouterr().out

    assert main(["inspect", str(spec)]) == 0
    inspected = capsys.readouterr().out
    assert "control_laws: 0" in inspected
    assert "mpc_controllers: 1" in inspected
    assert "horizon: 3" in inspected

    assert main(["compile", str(spec), "--target", "c", "--output", str(generated)]) == 0
    c_output = capsys.readouterr().out
    assert str(generated / "cli_mpc.h") in c_output
    assert (generated / "cli_mpc.c").exists()

    assert main(["compile", str(spec), "--target", "rust", "--output", str(generated)]) == 0
    rust_output = capsys.readouterr().out
    assert str(generated / "cli_mpc.rs") in rust_output
    assert (generated / "cli_mpc.rs").exists()

    assert (
        main(
            [
                "benchmark",
                str(spec),
                "--output",
                str(benchmarks),
                "--iterations",
                "3",
                "--warmup-iterations",
                "1",
                "--no-c",
                "--no-rust",
            ]
        )
        == 0
    )
    benchmark_output = capsys.readouterr().out
    assert str(benchmarks / "cli_mpc.json") in benchmark_output
    assert (benchmarks / "cli_mpc.md").exists()


def test_rl_validate_inspect_compile_and_benchmark(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "rl.yaml"
    generated = tmp_path / "generated"
    benchmarks = tmp_path / "benchmarks"
    _write_rl_spec(spec)

    assert main(["validate", str(spec)]) == 0
    assert "policy: rl" in capsys.readouterr().out

    assert main(["inspect", str(spec)]) == 0
    inspected = capsys.readouterr().out
    assert "control_laws: 0" in inspected
    assert "rl_policies: 1" in inspected
    assert "layers: 4" in inspected

    assert main(["compile", str(spec), "--target", "c", "--output", str(generated)]) == 0
    c_output = capsys.readouterr().out
    assert str(generated / "cli_rl.h") in c_output
    assert (generated / "cli_rl.c").exists()

    assert main(["compile", str(spec), "--target", "rust", "--output", str(generated)]) == 0
    rust_output = capsys.readouterr().out
    assert str(generated / "cli_rl.rs") in rust_output
    assert (generated / "cli_rl.rs").exists()

    assert (
        main(
            [
                "benchmark",
                str(spec),
                "--output",
                str(benchmarks),
                "--iterations",
                "3",
                "--warmup-iterations",
                "1",
                "--no-c",
                "--no-rust",
            ]
        )
        == 0
    )
    benchmark_output = capsys.readouterr().out
    assert str(benchmarks / "cli_rl.json") in benchmark_output
    assert (benchmarks / "cli_rl.md").exists()


def test_unsupported_policy_exits_with_error(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "pid.yaml"
    spec.write_text("name: pid\npolicy: pid\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["validate", str(spec)])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "unsupported policy" in captured.err
