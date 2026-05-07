"""Rust code generation backend."""

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
    ScalarMul,
    Sub,
    Vector,
    Zero,
)


class RustBackendError(ValueError):
    """Raised when an IR module cannot be lowered to Rust."""


@dataclass(frozen=True)
class RustGeneratedArtifact:
    """Generated Rust source artifact."""

    source_name: str
    source: str

    def write_to(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = output_dir / self.source_name
        source_path.write_text(self.source, encoding="utf-8")
        return source_path


@dataclass
class _RustValue:
    name: str
    dim: int


@dataclass
class _EmitContext:
    input_vector: Vector
    unroll_loops: bool
    lines: list[str] = field(default_factory=list)
    temp_index: int = 0

    def new_temp(self) -> str:
        name = f"tmp{self.temp_index}"
        self.temp_index += 1
        return name


@dataclass(frozen=True)
class RustBackend:
    """Generate deterministic no_std-compatible Rust for supported ControlKit IR modules."""

    unroll_loops: bool = False

    def generate(self, module: IRModule) -> RustGeneratedArtifact:
        law = self._select_control_law(module)
        input_vector = self._select_input_vector(module, law)
        module_id = _sanitize_identifier(module.name)
        matrices = _collect_matrices(law.expression)

        source = self._render_source(
            input_vector=input_vector,
            output_vector=law.output,
            expression=law.expression,
            matrices=matrices,
        )
        return RustGeneratedArtifact(source_name=f"{module_id}.rs", source=source)

    def _select_control_law(self, module: IRModule) -> ControlLaw:
        if len(module.control_laws) != 1:
            raise RustBackendError("Rust backend currently supports exactly one control law")
        return module.control_laws[0]

    def _select_input_vector(self, module: IRModule, law: ControlLaw) -> Vector:
        vectors = {
            vector.name: vector
            for vector in _collect_vectors(law.expression)
            if vector.name != law.output.name
        }
        if not vectors and "state_dim" in module.metadata:
            return Vector(
                module.metadata.get("state_name", "x"),
                dim=int(module.metadata["state_dim"]),
            )
        if len(vectors) != 1:
            raise RustBackendError("Rust backend currently supports exactly one input vector")
        return next(iter(vectors.values()))

    def _render_source(
        self,
        *,
        input_vector: Vector,
        output_vector: Vector,
        expression: Expr,
        matrices: tuple[Matrix, ...],
    ) -> str:
        lines = [
            "#![no_std]",
            "",
            f"pub const STATE_DIM: usize = {input_vector.dim};",
            f"pub const CONTROL_DIM: usize = {output_vector.dim};",
            "",
        ]
        for matrix in matrices:
            lines.extend(_render_matrix(matrix))
            lines.append("")

        input_name = _sanitize_identifier(input_vector.name)
        output_name = _sanitize_identifier(output_vector.name)
        lines.append(
            f"pub fn control_step({input_name}: &[f32; STATE_DIM], "
            f"{output_name}: &mut [f32; CONTROL_DIM]) {{"
        )
        ctx = _EmitContext(input_vector=input_vector, unroll_loops=self.unroll_loops)
        value = _emit_expr(expression, ctx)
        lines.extend(ctx.lines)
        if self.unroll_loops:
            for index in range(output_vector.dim):
                lines.append(f"    {output_name}[{index}] = {value.name}[{index}];")
        else:
            lines.append("    let mut i = 0usize;")
            lines.append("    while i < CONTROL_DIM {")
            lines.append(f"        {output_name}[i] = {value.name}[i];")
            lines.append("        i += 1;")
            lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)


def _emit_expr(expr: Expr, ctx: _EmitContext) -> _RustValue:
    if isinstance(expr, Vector):
        if expr.name != ctx.input_vector.name:
            raise RustBackendError(f"unsupported vector reference in expression: {expr.name}")
        return _RustValue(name=_sanitize_identifier(expr.name), dim=expr.dim)

    if isinstance(expr, Zero):
        temp = ctx.new_temp()
        ctx.lines.append(f"    let {temp} = [0.0_f32; {expr.shape.rows}];")
        return _RustValue(name=temp, dim=expr.shape.rows)

    if isinstance(expr, ScalarConstant):
        temp = ctx.new_temp()
        ctx.lines.append(f"    let {temp} = [{_float_literal(expr.value)}; 1];")
        return _RustValue(name=temp, dim=1)

    if isinstance(expr, MatVecMul):
        return _emit_matvec(expr, ctx)

    if isinstance(expr, Neg):
        value = _emit_expr(expr.value, ctx)
        temp = ctx.new_temp()
        ctx.lines.append(f"    let mut {temp} = [0.0_f32; {value.dim}];")
        if ctx.unroll_loops:
            for index in range(value.dim):
                ctx.lines.append(f"    {temp}[{index}] = -{value.name}[{index}];")
            return _RustValue(name=temp, dim=value.dim)
        ctx.lines.append("    let mut i = 0usize;")
        ctx.lines.append(f"    while i < {value.dim} {{")
        ctx.lines.append(f"        {temp}[i] = -{value.name}[i];")
        ctx.lines.append("        i += 1;")
        ctx.lines.append("    }")
        return _RustValue(name=temp, dim=value.dim)

    if isinstance(expr, ScalarMul):
        scalar = _emit_expr(expr.scalar, ctx)
        value = _emit_expr(expr.value, ctx)
        if scalar.dim != 1:
            raise RustBackendError("scalar multiplication requires a scalar temporary")
        temp = ctx.new_temp()
        ctx.lines.append(f"    let mut {temp} = [0.0_f32; {value.dim}];")
        if ctx.unroll_loops:
            for index in range(value.dim):
                ctx.lines.append(
                    f"    {temp}[{index}] = {scalar.name}[0] * {value.name}[{index}];"
                )
            return _RustValue(name=temp, dim=value.dim)
        ctx.lines.append("    let mut i = 0usize;")
        ctx.lines.append(f"    while i < {value.dim} {{")
        ctx.lines.append(f"        {temp}[i] = {scalar.name}[0] * {value.name}[i];")
        ctx.lines.append("        i += 1;")
        ctx.lines.append("    }")
        return _RustValue(name=temp, dim=value.dim)

    if isinstance(expr, Add | Sub):
        left = _emit_expr(expr.left, ctx)
        right = _emit_expr(expr.right, ctx)
        if left.dim != right.dim:
            raise RustBackendError("elementwise expression dimensions must match")
        temp = ctx.new_temp()
        op = "+" if isinstance(expr, Add) else "-"
        ctx.lines.append(f"    let mut {temp} = [0.0_f32; {left.dim}];")
        if ctx.unroll_loops:
            for index in range(left.dim):
                ctx.lines.append(
                    f"    {temp}[{index}] = {left.name}[{index}] {op} {right.name}[{index}];"
                )
            return _RustValue(name=temp, dim=left.dim)
        ctx.lines.append("    let mut i = 0usize;")
        ctx.lines.append(f"    while i < {left.dim} {{")
        ctx.lines.append(f"        {temp}[i] = {left.name}[i] {op} {right.name}[i];")
        ctx.lines.append("        i += 1;")
        ctx.lines.append("    }")
        return _RustValue(name=temp, dim=left.dim)

    if isinstance(expr, Clip):
        return _emit_clip(expr, ctx)

    raise RustBackendError(f"unsupported Rust expression: {type(expr).__name__}")


def _emit_matvec(expr: MatVecMul, ctx: _EmitContext) -> _RustValue:
    if not isinstance(expr.matrix, Matrix):
        raise RustBackendError("Rust backend requires matrix operands to be named matrices")
    if expr.matrix.values is None:
        raise RustBackendError(
            f"matrix {expr.matrix.name} has no numeric values for Rust generation"
        )
    vector = _emit_expr(expr.vector, ctx)
    matrix_name = _sanitize_identifier(expr.matrix.name).upper()
    temp = ctx.new_temp()
    ctx.lines.append(f"    let mut {temp} = [0.0_f32; {expr.matrix.rows}];")
    if ctx.unroll_loops:
        for row in range(expr.matrix.rows):
            terms = [
                f"{matrix_name}[{row}][{col}] * {vector.name}[{col}]"
                for col in range(expr.matrix.cols)
            ]
            ctx.lines.append(f"    {temp}[{row}] = {' + '.join(terms)};")
        return _RustValue(name=temp, dim=expr.matrix.rows)
    ctx.lines.append("    let mut row = 0usize;")
    ctx.lines.append(f"    while row < {expr.matrix.rows} {{")
    ctx.lines.append("        let mut acc = 0.0_f32;")
    ctx.lines.append("        let mut col = 0usize;")
    ctx.lines.append(f"        while col < {expr.matrix.cols} {{")
    ctx.lines.append(f"            acc += {matrix_name}[row][col] * {vector.name}[col];")
    ctx.lines.append("            col += 1;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        {temp}[row] = acc;")
    ctx.lines.append("        row += 1;")
    ctx.lines.append("    }")
    return _RustValue(name=temp, dim=expr.matrix.rows)


def _emit_clip(expr: Clip, ctx: _EmitContext) -> _RustValue:
    if not isinstance(expr.lower, ScalarConstant) or not isinstance(expr.upper, ScalarConstant):
        raise RustBackendError("Rust backend currently supports scalar clip bounds only")
    value = _emit_expr(expr.value, ctx)
    temp = ctx.new_temp()
    lower = _float_literal(expr.lower.value)
    upper = _float_literal(expr.upper.value)
    ctx.lines.append(f"    let mut {temp} = [0.0_f32; {value.dim}];")
    if ctx.unroll_loops:
        for index in range(value.dim):
            ctx.lines.append(f"    let mut clipped{index} = {value.name}[{index}];")
            ctx.lines.append(f"    if clipped{index} < {lower} {{")
            ctx.lines.append(f"        clipped{index} = {lower};")
            ctx.lines.append("    }")
            ctx.lines.append(f"    if clipped{index} > {upper} {{")
            ctx.lines.append(f"        clipped{index} = {upper};")
            ctx.lines.append("    }")
            ctx.lines.append(f"    {temp}[{index}] = clipped{index};")
        return _RustValue(name=temp, dim=value.dim)
    ctx.lines.append("    let mut i = 0usize;")
    ctx.lines.append(f"    while i < {value.dim} {{")
    ctx.lines.append(f"        let mut clipped = {value.name}[i];")
    ctx.lines.append(f"        if clipped < {lower} {{")
    ctx.lines.append(f"            clipped = {lower};")
    ctx.lines.append("        }")
    ctx.lines.append(f"        if clipped > {upper} {{")
    ctx.lines.append(f"            clipped = {upper};")
    ctx.lines.append("        }")
    ctx.lines.append(f"        {temp}[i] = clipped;")
    ctx.lines.append("        i += 1;")
    ctx.lines.append("    }")
    return _RustValue(name=temp, dim=value.dim)


def _render_matrix(matrix: Matrix) -> list[str]:
    if matrix.values is None:
        raise RustBackendError(f"matrix {matrix.name} has no numeric values for Rust generation")
    name = _sanitize_identifier(matrix.name).upper()
    lines = [f"const {name}: [[f32; {matrix.cols}]; {matrix.rows}] = ["]
    for row in matrix.values:
        values = ", ".join(_float_literal(value) for value in row)
        lines.append(f"    [{values}],")
    lines.append("];")
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
    return f"{text}_f32"


def _sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value]
    sanitized = "".join(chars)
    if not sanitized:
        raise RustBackendError("identifier cannot be empty")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
