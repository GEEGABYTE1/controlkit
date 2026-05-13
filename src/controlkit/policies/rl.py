"""Small fixed-shape RL policy frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from controlkit.compiler.ir import (
    ActivationLayerIR,
    IRModule,
    LinearLayerIR,
    RlLayerIR,
    RlLayerKind,
    RlPolicyIR,
    Vector,
)
from controlkit.policies.base import PolicyKind, PolicySpec


class RlSpecError(ValueError):
    """Raised when an RL policy specification is malformed."""


@dataclass(frozen=True)
class RlLinearLayerSpec:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @property
    def input_dim(self) -> int:
        return len(self.weights[0])

    @property
    def output_dim(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class RlActivationLayerSpec:
    kind: RlLayerKind
    dim: int

    @property
    def input_dim(self) -> int:
        return self.dim

    @property
    def output_dim(self) -> int:
        return self.dim


RlLayerSpec = RlLinearLayerSpec | RlActivationLayerSpec


@dataclass(frozen=True)
class RlControllerSpec(PolicySpec):
    input_dim: int
    output_dim: int
    layers: tuple[RlLayerSpec, ...]
    input_name: str = "x"
    output_name: str = "u"

    def __post_init__(self) -> None:
        _validate_name(self.name, "controller name")
        _validate_name(self.input_name, "input vector name")
        _validate_name(self.output_name, "output vector name")
        _validate_positive_int(self.input_dim, "input_dim")
        _validate_positive_int(self.output_dim, "output_dim")
        if not self.layers:
            raise RlSpecError("RL policy must contain at least one layer")

        current_dim = self.input_dim
        saw_linear = False
        for layer in self.layers:
            if layer.input_dim != current_dim:
                raise RlSpecError(
                    f"layer input dimension {layer.input_dim} does not match {current_dim}"
                )
            current_dim = layer.output_dim
            saw_linear = saw_linear or isinstance(layer, RlLinearLayerSpec)

        if not saw_linear:
            raise RlSpecError("RL policy must contain at least one linear layer")
        if current_dim != self.output_dim:
            raise RlSpecError(
                f"final layer output dimension {current_dim} does not match output_dim"
            )


@dataclass(frozen=True)
class RlPolicy:
    kind: PolicyKind = PolicyKind.RL

    def load(self, spec_path: Path) -> PolicySpec:
        return PolicySpec(name=spec_path.stem, source_path=spec_path)

    def lower(self, spec: PolicySpec) -> IRModule:
        if not isinstance(spec, RlControllerSpec):
            return IRModule(name=spec.name, policy=self.kind)

        input_vector = Vector(spec.input_name, dim=spec.input_dim)
        output_vector = Vector(spec.output_name, dim=spec.output_dim)
        layers: list[RlLayerIR] = []
        for layer in spec.layers:
            if isinstance(layer, RlLinearLayerSpec):
                layers.append(LinearLayerIR(weights=layer.weights, bias=layer.bias))
            else:
                layers.append(ActivationLayerIR(kind=layer.kind, dim=layer.dim))

        policy = RlPolicyIR(
            name=spec.name,
            input_vector=input_vector,
            output_vector=output_vector,
            layers=tuple(layers),
        )
        return IRModule(
            name=spec.name,
            policy=self.kind,
            metadata={
                "frontend": "rl",
                "input_name": spec.input_name,
                "output_name": spec.output_name,
                "input_dim": str(spec.input_dim),
                "output_dim": str(spec.output_dim),
                "layers": str(len(spec.layers)),
            },
            rl_policies=(policy,),
        )

    def from_layers(
        self,
        *,
        name: str,
        input_dim: int,
        output_dim: int,
        layers: Sequence[Mapping[str, Any]],
        input_name: str = "x",
        output_name: str = "u",
        source_path: Path | None = None,
    ) -> RlControllerSpec:
        normalized_layers = _normalize_layers(layers, input_dim)
        return RlControllerSpec(
            name=name,
            source_path=source_path or Path("<rl-api>"),
            input_dim=input_dim,
            output_dim=output_dim,
            layers=normalized_layers,
            input_name=input_name,
            output_name=output_name,
        )

    def from_json_file(
        self,
        *,
        name: str,
        weights_path: Path,
        input_name: str = "x",
        output_name: str = "u",
    ) -> RlControllerSpec:
        if not weights_path.exists():
            raise RlSpecError(f"weights file does not exist: {weights_path}")
        if not weights_path.is_file():
            raise RlSpecError(f"weights path is not a file: {weights_path}")
        try:
            raw = json.loads(weights_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RlSpecError(f"invalid JSON weights file: {weights_path}") from exc
        if not isinstance(raw, dict):
            raise RlSpecError("RL weights JSON must be an object")

        return self.from_layers(
            name=name,
            input_dim=int(_required(raw, "input_dim")),
            output_dim=int(_required(raw, "output_dim")),
            layers=_required(raw, "layers"),
            input_name=str(raw.get("input_name", input_name)),
            output_name=str(raw.get("output_name", output_name)),
            source_path=weights_path,
        )


def _normalize_layers(
    raw_layers: Sequence[Mapping[str, Any]],
    input_dim: int,
) -> tuple[RlLayerSpec, ...]:
    if isinstance(raw_layers, str):
        raise TypeError("layers must be a sequence of layer mappings")
    if not raw_layers:
        raise RlSpecError("RL policy must contain at least one layer")

    current_dim = input_dim
    layers: list[RlLayerSpec] = []
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, Mapping):
            raise RlSpecError("each RL layer must be a mapping")
        kind = str(raw_layer.get("type", raw_layer.get("kind", ""))).lower()
        if kind == RlLayerKind.LINEAR.value:
            weights = _normalize_matrix(_required(raw_layer, "weights"), "weights")
            bias = _normalize_vector(_required(raw_layer, "bias"), "bias")
            layer = RlLinearLayerSpec(weights=weights, bias=bias)
            if layer.input_dim != current_dim:
                raise RlSpecError(
                    f"linear layer input dimension {layer.input_dim} does not match {current_dim}"
                )
            if len(bias) != layer.output_dim:
                raise RlSpecError("linear layer bias length must match output dimension")
            current_dim = layer.output_dim
            layers.append(layer)
            continue
        if kind in {RlLayerKind.RELU.value, RlLayerKind.TANH.value}:
            layers.append(RlActivationLayerSpec(kind=RlLayerKind(kind), dim=current_dim))
            continue
        raise RlSpecError(f"unsupported RL layer type: {kind}")
    return tuple(layers)


def _normalize_matrix(
    values: Sequence[Sequence[float]],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a 2D numeric sequence")
    rows: list[tuple[float, ...]] = []
    for row in values:
        if isinstance(row, str):
            raise TypeError(f"{name} rows must be numeric sequences")
        rows.append(tuple(float(value) for value in row))
    if not rows:
        raise RlSpecError(f"{name} must have at least one row")
    if not rows[0]:
        raise RlSpecError(f"{name} must have at least one column")
    expected_cols = len(rows[0])
    if any(len(row) != expected_cols for row in rows):
        raise RlSpecError(f"{name} rows must all have the same length")
    return tuple(rows)


def _normalize_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a numeric sequence")
    result = tuple(float(value) for value in values)
    if not result:
        raise RlSpecError(f"{name} must have at least one entry")
    return result


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise RlSpecError(f"missing required field: {key}")
    return mapping[key]


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise RlSpecError(f"{name} must be positive")


def _validate_name(value: str, name: str) -> None:
    if not value:
        raise RlSpecError(f"{name} must be non-empty")
