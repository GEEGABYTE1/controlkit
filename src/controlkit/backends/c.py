#c code generation backend

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from controlkit.compiler.ir import (
    Add,
    Clip,
    ControlLaw,
    Expr,
    IRModule,
    Matrix,
    MatVecMul,
    Neg,
    ScalarConstant,
    Sub,
    ValueKind,
    Vector,
)

class CBackendError(ValueError):
    """Raised when an IR module cannot be lowered to C."""


@dataclass(frozen=True)
class CGeneratedArtifact:
    """Generated C header/source pair."""
    header_name: str
    source_name: str
    header: str
    source: str

    def write_to(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        header_path = output_dir / self.header_name
        source_path = output_dir / self.source_name
        header_path.write_text(self.header, encoding="utf-8")
        source_path.write_text(self.source, encoding="utf-8")
        return header_path, source_path


@dataclass
class _CValue:
    name: str
    dim: int


@dataclass
class _EmitContext:
    input_vector: Vector
    lines: list[str] = field(default_factory=list)
    temp_index: int = 0

    def new_temp(self) -> str:
        name = f"tmp{self.temp_index}"
        self.temp_index += 1
        return name


class CBackend:
    """Generate deterministic float32 C for supported ControlKit IR modules."""

    def generate(self, module: IRModule) -> CGeneratedArtifact:
        law = self._select_control_law(module)
        input_vector = self._select_input_vector(law)

        module_id = _sanitize_identifier(module.name)
        header_name = f"{module_id}.h"
        source_name = f"{module_id}.c"
        function_name = f"{module_id}_control_step"
        guard = f"CONTROLKIT_{module_id.upper()}_H"

        matrices = _collect_matrices(law.expression)
        header = self._render_header(
            guard=guard,
            function_name=function_name,
            input_vector=input_vector,
            output_vector=law.output,
        )
        source = self._render_source(
            header_name=header_name,
            function_name=function_name,
            input_vector=input_vector,
            output_vector=law.output,
            expression=law.expression,
            matrices=matrices,
        )
        return CGeneratedArtifact(
            header_name=header_name,
            source_name=source_name,
            header=header,
            source=source,
        )

    def _select_control_law(self, module: IRModule) -> ControlLaw:
        if len(module.control_laws) != 1:
            raise CBackendError("C backend currently supports exactly one control law")
        return module.control_laws[0]

    def _select_input_vector(self, law: ControlLaw) -> Vector:
        vectors = {
            vector.name: vector
            for vector in _collect_vectors(law.expression)
            if vector.name != law.output.name
        }
        if len(vectors) != 1:
            raise CBackendError("C backend currently supports exactly one input vector")
        return next(iter(vectors.values()))

    def _render_header(
        self,
        *,
        guard: str,
        function_name: str,
        input_vector: Vector,
        output_vector: Vector,
    ) -> str:
        return "\n".join(
            [
                f"#ifndef {guard}",
                f"#define {guard}",
                "",
                "#include <stddef.h>",
                "",
                "#ifdef __cplusplus",
                'extern "C" {',
                "#endif",
                "",
                f"#define CONTROLKIT_STATE_DIM {input_vector.dim}u",
                f"#define CONTROLKIT_CONTROL_DIM {output_vector.dim}u",
                "",
                (
                    f"void {function_name}(const float {_sanitize_identifier(input_vector.name)}"
                    f"[CONTROLKIT_STATE_DIM], float {_sanitize_identifier(output_vector.name)}"
                    "[CONTROLKIT_CONTROL_DIM]);"
                ),
                "",
                "#ifdef __cplusplus",
                "}",
                "#endif",
                "",
                f"#endif /* {guard} */",
                "",
            ]
        )

    def _render_source(
        self,
        *,
        header_name: str,
        function_name: str,
        input_vector: Vector,
        output_vector: Vector,
        expression: Expr,
        matrices: tuple[Matrix, ...],
    ) -> str:
        lines = [f'#include "{header_name}"', ""]
        for matrix in matrices:
            lines.extend(_render_matrix(matrix))
            lines.append("")

        input_name = _sanitize_identifier(input_vector.name)
        output_name = _sanitize_identifier(output_vector.name)
        lines.append(
            f"void {function_name}(const float {input_name}[CONTROLKIT_STATE_DIM], "
            f"float {output_name}[CONTROLKIT_CONTROL_DIM])"
        )
        lines.append("{")
        ctx = _EmitContext(input_vector=input_vector)
        value = _emit_expr(expression, ctx)
        lines.extend(ctx.lines)
        lines.append("    for (size_t i = 0u; i < CONTROLKIT_CONTROL_DIM; ++i) {")
        lines.append(f"        {output_name}[i] = {value.name}[i];")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)


def _emit_expr(expr: Expr, ctx: _EmitContext) -> _CValue:
    if isinstance(expr, Vector):
        if expr.name != ctx.input_vector.name:
            raise CBackendError(f"unsupported vector reference in expression: {expr.name}")
        return _CValue(name=_sanitize_identifier(expr.name), dim=expr.dim)

    if isinstance(expr, MatVecMul):
        return _emit_matvec(expr, ctx)

    if isinstance(expr, Neg):
        value = _emit_expr(expr.value, ctx)
        temp = ctx.new_temp()
        ctx.lines.append(f"    float {temp}[{value.dim}u];")
        ctx.lines.append(f"    for (size_t i = 0u; i < {value.dim}u; ++i) {{")
        ctx.lines.append(f"        {temp}[i] = -{value.name}[i];")
        ctx.lines.append("    }")
        return _CValue(name=temp, dim=value.dim)

    if isinstance(expr, Add | Sub):
        left = _emit_expr(expr.left, ctx)
        right = _emit_expr(expr.right, ctx)
        if left.dim != right.dim:
            raise CBackendError("elementwise expression dimensions must match")
        temp = ctx.new_temp()
        op = "+" if isinstance(expr, Add) else "-"
        ctx.lines.append(f"    float {temp}[{left.dim}u];")
        ctx.lines.append(f"    for (size_t i = 0u; i < {left.dim}u; ++i) {{")
        ctx.lines.append(f"        {temp}[i] = {left.name}[i] {op} {right.name}[i];")
        ctx.lines.append("    }")
        return _CValue(name=temp, dim=left.dim)

    if isinstance(expr, Clip):
        return _emit_clip(expr, ctx)

    raise CBackendError(f"unsupported C expression: {type(expr).__name__}")


def _emit_matvec(expr: MatVecMul, ctx: _EmitContext) -> _CValue:
    if not isinstance(expr.matrix, Matrix):
        raise CBackendError("C backend requires matrix operands to be named matrices")
    if expr.matrix.values is None:
        raise CBackendError(f"matrix {expr.matrix.name} has no numeric values for C generation")
    vector = _emit_expr(expr.vector, ctx)
    matrix_name = _sanitize_identifier(expr.matrix.name)
    temp = ctx.new_temp()
    ctx.lines.append(f"    float {temp}[{expr.matrix.rows}u];")
    ctx.lines.append(f"    for (size_t row = 0u; row < {expr.matrix.rows}u; ++row) {{")
    ctx.lines.append("        float acc = 0.0f;")
    ctx.lines.append(f"        for (size_t col = 0u; col < {expr.matrix.cols}u; ++col) {{")
    ctx.lines.append(f"            acc += {matrix_name}[row][col] * {vector.name}[col];")
    ctx.lines.append("        }")
    ctx.lines.append(f"        {temp}[row] = acc;")
    ctx.lines.append("    }")
    return _CValue(name=temp, dim=expr.matrix.rows)


def _emit_clip(expr: Clip, ctx: _EmitContext) -> _CValue:
    if not isinstance(expr.lower, ScalarConstant) or not isinstance(expr.upper, ScalarConstant):
        raise CBackendError("C backend currently supports scalar clip bounds only")
    value = _emit_expr(expr.value, ctx)
    temp = ctx.new_temp()
    lower = _float_literal(expr.lower.value)
    upper = _float_literal(expr.upper.value)
    ctx.lines.append(f"    float {temp}[{value.dim}u];")
    ctx.lines.append(f"    for (size_t i = 0u; i < {value.dim}u; ++i) {{")
    ctx.lines.append(f"        float clipped = {value.name}[i];")
    ctx.lines.append(f"        if (clipped < {lower}) {{")
    ctx.lines.append(f"            clipped = {lower};")
    ctx.lines.append("        }")
    ctx.lines.append(f"        if (clipped > {upper}) {{")
    ctx.lines.append(f"            clipped = {upper};")
    ctx.lines.append("        }")
    ctx.lines.append(f"        {temp}[i] = clipped;")
    ctx.lines.append("    }")
    return _CValue(name=temp, dim=value.dim)


def _render_matrix(matrix: Matrix) -> list[str]:
    if matrix.values is None:
        raise CBackendError(f"matrix {matrix.name} has no numeric values for C generation")
    name = _sanitize_identifier(matrix.name)
    lines = [f"static const float {name}[{matrix.rows}u][{matrix.cols}u] = {{"]
    for row_index, row in enumerate(matrix.values):
        suffix = "," if row_index < matrix.rows - 1 else ""
        values = ", ".join(_float_literal(value) for value in row)
        lines.append(f"    {{{values}}}{suffix}")
    lines.append("};")
    return lines


def _collect_matrices(expr: Expr) -> tuple[Matrix, ...]:
    matrices: dict[str, Matrix] = {}
    for child in _walk_expr(expr):
        if isinstance(child, Matrix):
            matrices[child.name] = child
    return tuple(matrices[name] for name in sorted(matrices))


def _collect_vectors(expr: Expr) -> tuple[Vector, ...]:
    vectors: dict[str, Vector] = {}
    for child in _walk_expr(expr):
        if isinstance(child, Vector):
            vectors[child.name] = child
    return tuple(vectors[name] for name in sorted(vectors))


def _walk_expr(expr: Expr) -> tuple[Expr, ...]:
    children: list[Expr] = [expr]
    if isinstance(expr, MatVecMul):
        children.extend(_walk_expr(expr.matrix))
        children.extend(_walk_expr(expr.vector))
    elif isinstance(expr, Neg):
        children.extend(_walk_expr(expr.value))
    elif isinstance(expr, Add | Sub):
        children.extend(_walk_expr(expr.left))
        children.extend(_walk_expr(expr.right))
    elif isinstance(expr, Clip):
        children.extend(_walk_expr(expr.value))
        children.extend(_walk_expr(expr.lower))
        children.extend(_walk_expr(expr.upper))
    return tuple(children)


def _float_literal(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text and "." not in text:
        text = f"{text}.0"
    return f"{text}f"


def _sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value]
    sanitized = "".join(chars)
    if not sanitized:
        raise CBackendError("identifier cannot be empty")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
