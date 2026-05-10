# controller specification loading for CLI

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controlkit.compiler.ir import IRModule
from controlkit.policies.base import PolicyKind
from controlkit.policies.lqr import LqrPolicy, LqrSpecError
from controlkit.policies.mpc import MpcPolicy, MpcSpecError


class ControllerSpecError(ValueError):
    """Raised when a controller specification cannot be loaded or lowered."""


@dataclass(frozen=True)
class LoadedControllerSpec:
    path: Path
    raw: dict[str, Any]
    module: IRModule

    @property
    def policy(self) -> str:
        return str(self.raw["policy"])


def load_controller_spec(path: Path) -> LoadedControllerSpec:
    if not path.exists():
        raise ControllerSpecError(f"spec file does not exist: {path}")
    if not path.is_file():
        raise ControllerSpecError(f"spec path is not a file: {path}")

    raw = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    _require_mapping(raw, "spec")
    policy = str(_required(raw, "policy"))
    if policy == PolicyKind.LQR.value:
        module = _lower_lqr(path, raw)
    elif policy == PolicyKind.MPC.value:
        module = _lower_mpc(path, raw)
    else:
        raise ControllerSpecError(f"unsupported policy for CLI lowering: {policy}")
    return LoadedControllerSpec(path=path, raw=raw, module=module)


def _lower_lqr(path: Path, raw: dict[str, Any]) -> IRModule:
    name = str(raw.get("name", path.stem))
    gain_matrix = raw.get("gain_matrix")
    if gain_matrix is None:
        raise ControllerSpecError("lqr spec requires gain_matrix")

    state_dim = _optional_int(raw.get("state_dim"))
    control_dim = _optional_int(raw.get("control_dim"))
    state_name = str(raw.get("state_name", "x"))
    control_name = str(raw.get("control_name", "u"))
    gain_name = str(raw.get("gain_name", "K"))
    state_names = tuple(str(value) for value in raw.get("state_names", ()))
    control_names = tuple(str(value) for value in raw.get("control_names", ()))
    saturation = _parse_saturation(raw.get("saturation"))

    try:
        frontend = LqrPolicy()
        spec = frontend.from_gain_matrix(
            name=name,
            gain_matrix=gain_matrix,
            state_dim=state_dim,
            control_dim=control_dim,
            saturation=saturation,
            state_name=state_name,
            control_name=control_name,
            gain_name=gain_name,
            state_names=state_names,
            control_names=control_names,
        )
        return frontend.lower(spec)
    except (TypeError, LqrSpecError) as exc:
        raise ControllerSpecError(str(exc)) from exc


def _parse_saturation(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return (float(_required(value, "lower")), float(_required(value, "upper")))
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ControllerSpecError("saturation list must contain lower and upper bounds")
        return (float(value[0]), float(value[1]))
    raise ControllerSpecError("saturation must be a mapping or [lower, upper] list")


def _lower_mpc(path: Path, raw: dict[str, Any]) -> IRModule:
    name = str(raw.get("name", path.stem))
    try:
        frontend = MpcPolicy()
        spec = frontend.from_matrices(
            name=name,
            a_matrix=_required(raw, "a_matrix"),
            b_matrix=_required(raw, "b_matrix"),
            state_dim=_optional_int(raw.get("state_dim")),
            control_dim=_optional_int(raw.get("control_dim")),
            horizon=int(_required(raw, "horizon")),
            q_diagonal=_required(raw, "q_diagonal"),
            r_diagonal=_required(raw, "r_diagonal"),
            q_terminal_diagonal=raw.get("q_terminal_diagonal"),
            u_min=_required(raw, "u_min"),
            u_max=_required(raw, "u_max"),
            solver_iterations=int(_required(raw, "solver_iterations")),
            step_size=float(_required(raw, "step_size")),
            state_name=str(raw.get("state_name", "x")),
            control_name=str(raw.get("control_name", "u")),
        )
        return frontend.lower(spec)
    except (TypeError, MpcSpecError) as exc:
        raise ControllerSpecError(str(exc)) from exc


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = _logical_lines(text)
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, (indent, content) in enumerate(lines):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ControllerSpecError("list item appears outside a list")
            parent.append(_parse_scalar(content[2:].strip()))
            continue

        if ":" not in content:
            raise ControllerSpecError(f"invalid spec line: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ControllerSpecError("empty spec key")
        if not isinstance(parent, dict):
            raise ControllerSpecError(f"cannot assign key {key!r} inside a list")

        if raw_value:
            parent[key] = _parse_scalar(raw_value)
            continue

        child: dict[str, Any] | list[Any]
        child = [] if _next_line_is_list(lines, index, indent) else {}
        parent[key] = child
        stack.append((indent, child))

    return root


def _logical_lines(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        result.append((indent, without_comment.strip()))
    return result


def _next_line_is_list(lines: list[tuple[int, str]], index: int, indent: int) -> bool:
    if index + 1 >= len(lines):
        return False
    next_indent, next_content = lines[index + 1]
    return next_indent > indent and next_content.startswith("- ")


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") or value.startswith("{") or value.startswith("("):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ControllerSpecError(f"invalid inline value: {value}") from exc
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ControllerSpecError(f"missing required field: {key}")
    return mapping[key]


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ControllerSpecError(f"{name} must be a mapping")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
