from __future__ import annotations

import json

from controlkit.verify.report import verify_controller_file


def test_verification_report_outputs_json_and_markdown(tmp_path) -> None:
    spec_path = tmp_path / "controller.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: controller",
                "policy: lqr",
                "system_type: discrete",
                "state_dim: 2",
                "control_dim: 1",
                "gain_matrix:",
                "  - [2.0, 2.0]",
                "a_matrix:",
                "  - [1.0, 0.1]",
                "  - [0.0, 1.0]",
                "b_matrix:",
                "  - [0.0]",
                "  - [0.1]",
                "saturation:",
                "  lower: -2.0",
                "  upper: 2.0",
            ]
        ),
        encoding="utf-8",
    )
    report = verify_controller_file(
        spec_path,
        tmp_path / "out",
    )

    assert report.json_path.exists()
    assert report.markdown_path.exists()
    data = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert data["controller_name"] == "controller"
    assert "checks" in data
