# C generation backend

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from controlkit.compiler.ir import (
    Add,
    Clip,
    ControlLaw,
    Expr,
    IRModule,
    ActivationLayerIR,
    LinearLayerIR,
    Matrix,
    MatVecMul,
    MpcControllerIR,
    Neg,
    ScalarConstant,
    ScalarMul,
    Sub,
    Vector,
    Zero,
)


class CBackendError(ValueError):
    """Raised when an IR module cannot be lowered to C."""


@dataclass(frozen=True)
class CGeneratedArtifact:
    # Generated C header/source pair.
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
    unroll_loops: bool
    lines: list[str] = field(default_factory=list)
    temp_index: int = 0

    def new_temp(self) -> str:
        name = f"tmp{self.temp_index}"
        self.temp_index += 1
        return name


@dataclass(frozen=True)
class CBackend:
    unroll_loops: bool = False

    def generate(self, module: IRModule) -> CGeneratedArtifact:
        if module.rl_policies:
            return self._generate_rl(module)
        if module.mpc_controllers:
            return self._generate_mpc(module)
        law = self._select_control_law(module)
        input_vector = self._select_input_vector(module, law)

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

    def _generate_rl(self, module: IRModule) -> CGeneratedArtifact:
        if len(module.rl_policies) != 1:
            raise CBackendError("C backend currently supports exactly one RL policy")
        policy = module.rl_policies[0]
        module_id = _sanitize_identifier(module.name)
        header_name = f"{module_id}.h"
        source_name = f"{module_id}.c"
        function_name = f"{module_id}_control_step"
        guard = f"CONTROLKIT_{module_id.upper()}_H"
        header = self._render_header(
            guard=guard,
            function_name=function_name,
            input_vector=policy.input_vector,
            output_vector=policy.output_vector,
        )
        source = _render_rl_source(
            header_name=header_name,
            function_name=function_name,
            input_vector=policy.input_vector,
            output_vector=policy.output_vector,
            layers=policy.layers,
        )
        return CGeneratedArtifact(
            header_name=header_name,
            source_name=source_name,
            header=header,
            source=source,
        )

    def _generate_mpc(self, module: IRModule) -> CGeneratedArtifact:
        if len(module.mpc_controllers) != 1:
            raise CBackendError("C backend currently supports exactly one MPC controller")
        controller = module.mpc_controllers[0]
        module_id = _sanitize_identifier(module.name)
        header_name = f"{module_id}.h"
        source_name = f"{module_id}.c"
        function_name = f"{module_id}_control_step"
        guard = f"CONTROLKIT_{module_id.upper()}_H"
        header = self._render_header(
            guard=guard,
            function_name=function_name,
            input_vector=controller.state,
            output_vector=controller.control,
        )
        source = _render_mpc_source(
            header_name=header_name,
            function_name=function_name,
            controller=controller,
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
        ctx = _EmitContext(input_vector=input_vector, unroll_loops=self.unroll_loops)
        value = _emit_expr(expression, ctx)
        lines.extend(ctx.lines)
        if self.unroll_loops:
            for index in range(output_vector.dim):
                lines.append(f"    {output_name}[{index}u] = {value.name}[{index}u];")
        else:
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

    if isinstance(expr, Zero):
        temp = ctx.new_temp()
        ctx.lines.append(f"    float {temp}[{expr.shape.rows}u] = {{0.0f}};")
        return _CValue(name=temp, dim=expr.shape.rows)

    if isinstance(expr, ScalarConstant):
        temp = ctx.new_temp()
        ctx.lines.append(f"    float {temp}[1u] = {{{_float_literal(expr.value)}}};")
        return _CValue(name=temp, dim=1)

    if isinstance(expr, MatVecMul):
        return _emit_matvec(expr, ctx)

    if isinstance(expr, Neg):
        value = _emit_expr(expr.value, ctx)
        temp = ctx.new_temp()
        ctx.lines.append(f"    float {temp}[{value.dim}u];")
        if ctx.unroll_loops:
            for index in range(value.dim):
                ctx.lines.append(f"    {temp}[{index}u] = -{value.name}[{index}u];")
            return _CValue(name=temp, dim=value.dim)
        ctx.lines.append(f"    for (size_t i = 0u; i < {value.dim}u; ++i) {{")
        ctx.lines.append(f"        {temp}[i] = -{value.name}[i];")
        ctx.lines.append("    }")
        return _CValue(name=temp, dim=value.dim)

    if isinstance(expr, ScalarMul):
        scalar = _emit_expr(expr.scalar, ctx)
        value = _emit_expr(expr.value, ctx)
        if scalar.dim != 1:
            raise CBackendError("scalar multiplication requires a scalar temporary")
        temp = ctx.new_temp()
        ctx.lines.append(f"    float {temp}[{value.dim}u];")
        if ctx.unroll_loops:
            for index in range(value.dim):
                ctx.lines.append(
                    f"    {temp}[{index}u] = {scalar.name}[0u] * {value.name}[{index}u];"
                )
            return _CValue(name=temp, dim=value.dim)
        ctx.lines.append(f"    for (size_t i = 0u; i < {value.dim}u; ++i) {{")
        ctx.lines.append(f"        {temp}[i] = {scalar.name}[0u] * {value.name}[i];")
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
        if ctx.unroll_loops:
            for index in range(left.dim):
                ctx.lines.append(
                    f"    {temp}[{index}u] = {left.name}[{index}u] {op} {right.name}[{index}u];"
                )
            return _CValue(name=temp, dim=left.dim)
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
    if ctx.unroll_loops:
        for row in range(expr.matrix.rows):
            terms = [
                f"{matrix_name}[{row}u][{col}u] * {vector.name}[{col}u]"
                for col in range(expr.matrix.cols)
            ]
            ctx.lines.append(f"    {temp}[{row}u] = {' + '.join(terms)};")
        return _CValue(name=temp, dim=expr.matrix.rows)
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
    if ctx.unroll_loops:
        for index in range(value.dim):
            ctx.lines.append(f"    float clipped{index} = {value.name}[{index}u];")
            ctx.lines.append(f"    if (clipped{index} < {lower}) {{")
            ctx.lines.append(f"        clipped{index} = {lower};")
            ctx.lines.append("    }")
            ctx.lines.append(f"    if (clipped{index} > {upper}) {{")
            ctx.lines.append(f"        clipped{index} = {upper};")
            ctx.lines.append("    }")
            ctx.lines.append(f"    {temp}[{index}u] = clipped{index};")
        return _CValue(name=temp, dim=value.dim)
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


def _render_rl_source(
    *,
    header_name: str,
    function_name: str,
    input_vector: Vector,
    output_vector: Vector,
    layers: tuple[LinearLayerIR | ActivationLayerIR, ...],
) -> str:
    lines = [
        f'#include "{header_name}"',
        "",
        "#include <math.h>",
        "",
    ]
    for index, layer in enumerate(layers):
        if isinstance(layer, LinearLayerIR):
            lines.extend(_render_rl_matrix(f"LAYER_{index}_WEIGHTS", layer.weights))
            lines.append("")
            lines.extend(_render_vector_constant(f"LAYER_{index}_BIAS", layer.bias))
            lines.append("")

    input_name = _sanitize_identifier(input_vector.name)
    output_name = _sanitize_identifier(output_vector.name)
    current_name = input_name
    current_dim = input_vector.dim
    body: list[str] = []
    for index, layer in enumerate(layers):
        next_name = f"layer_{index}"
        body.append(f"    float {next_name}[{layer.output_dim}u];")
        if isinstance(layer, LinearLayerIR):
            weights_name = f"LAYER_{index}_WEIGHTS"
            bias_name = f"LAYER_{index}_BIAS"
            body.append(f"    for (size_t row = 0u; row < {layer.output_dim}u; ++row) {{")
            body.append(f"        float acc = {bias_name}[row];")
            body.append(f"        for (size_t col = 0u; col < {layer.input_dim}u; ++col) {{")
            body.append(f"            acc += {weights_name}[row][col] * {current_name}[col];")
            body.append("        }")
            body.append(f"        {next_name}[row] = acc;")
            body.append("    }")
        elif layer.kind.value == "relu":
            body.append(f"    for (size_t i = 0u; i < {current_dim}u; ++i) {{")
            body.append(f"        float value = {current_name}[i];")
            body.append(f"        {next_name}[i] = value > 0.0f ? value : 0.0f;")
            body.append("    }")
        else:
            body.append(f"    for (size_t i = 0u; i < {current_dim}u; ++i) {{")
            body.append(f"        {next_name}[i] = tanhf({current_name}[i]);")
            body.append("    }")
        current_name = next_name
        current_dim = layer.output_dim

    lines.append(
        f"void {function_name}(const float {input_name}[CONTROLKIT_STATE_DIM], "
        f"float {output_name}[CONTROLKIT_CONTROL_DIM])"
    )
    lines.append("{")
    lines.extend(body)
    lines.append("    for (size_t i = 0u; i < CONTROLKIT_CONTROL_DIM; ++i) {")
    lines.append(f"        {output_name}[i] = {current_name}[i];")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _render_rl_matrix(name: str, values: tuple[tuple[float, ...], ...]) -> list[str]:
    rows = len(values)
    cols = len(values[0])
    lines = [f"static const float {name}[{rows}u][{cols}u] = {{"]
    for row_index, row in enumerate(values):
        suffix = "," if row_index < rows - 1 else ""
        rendered = ", ".join(_float_literal(value) for value in row)
        lines.append(f"    {{{rendered}}}{suffix}")
    lines.append("};")
    return lines


def _render_mpc_source(
    *,
    header_name: str,
    function_name: str,
    controller: MpcControllerIR,
) -> str:
    n = controller.state.dim
    m = controller.control.dim
    h = controller.horizon
    lines = [
        f'#include "{header_name}"',
        "",
        f"#define CONTROLKIT_MPC_HORIZON {h}u",
        f"#define CONTROLKIT_MPC_SOLVER_ITERATIONS {controller.solver_iterations}u",
        "",
    ]
    lines.extend(_render_matrix(controller.a_matrix))
    lines.append("")
    lines.extend(_render_matrix(controller.b_matrix))
    lines.append("")
    lines.extend(_render_vector_constant("Q_DIAGONAL", controller.q_diagonal))
    lines.append("")
    lines.extend(_render_vector_constant("R_DIAGONAL", controller.r_diagonal))
    lines.append("")
    lines.extend(_render_vector_constant("Q_TERMINAL_DIAGONAL", controller.q_terminal_diagonal))
    lines.append("")
    lines.extend(_render_vector_constant("U_MIN", controller.u_min))
    lines.append("")
    lines.extend(_render_vector_constant("U_MAX", controller.u_max))
    lines.extend(
        [
            "",
            f"static const float STEP_SIZE = {_float_literal(controller.step_size)};",
            "",
            f"void {function_name}(const float x[CONTROLKIT_STATE_DIM], "
            "float u[CONTROLKIT_CONTROL_DIM])",
            "{",
            f"    float X[CONTROLKIT_MPC_HORIZON + 1u][{n}u];",
            f"    float U[CONTROLKIT_MPC_HORIZON][{m}u] = {{0}};",
            f"    float lambda[CONTROLKIT_MPC_HORIZON + 1u][{n}u];",
            f"    float grad[CONTROLKIT_MPC_HORIZON][{m}u];",
            "",
            "    for (size_t iter = 0u; iter < CONTROLKIT_MPC_SOLVER_ITERATIONS; ++iter) {",
            f"        for (size_t i = 0u; i < {n}u; ++i) {{",
            "            X[0u][i] = x[i];",
            "        }",
            "        for (size_t k = 0u; k < CONTROLKIT_MPC_HORIZON; ++k) {",
            f"            for (size_t row = 0u; row < {n}u; ++row) {{",
            "                float acc = 0.0f;",
            f"                for (size_t col = 0u; col < {n}u; ++col) {{",
            "                    acc += A[row][col] * X[k][col];",
            "                }",
            f"                for (size_t col = 0u; col < {m}u; ++col) {{",
            "                    acc += B[row][col] * U[k][col];",
            "                }",
            "                X[k + 1u][row] = acc;",
            "            }",
            "        }",
            f"        for (size_t i = 0u; i < {n}u; ++i) {{",
            "            lambda[CONTROLKIT_MPC_HORIZON][i] =",
            "                Q_TERMINAL_DIAGONAL[i] * X[CONTROLKIT_MPC_HORIZON][i];",
            "        }",
            "        for (size_t kk = CONTROLKIT_MPC_HORIZON; kk > 0u; --kk) {",
            "            size_t k = kk - 1u;",
            f"            for (size_t j = 0u; j < {m}u; ++j) {{",
            "                float acc = R_DIAGONAL[j] * U[k][j];",
            f"                for (size_t i = 0u; i < {n}u; ++i) {{",
            "                    acc += B[i][j] * lambda[k + 1u][i];",
            "                }",
            "                grad[k][j] = acc;",
            "            }",
            f"            for (size_t i = 0u; i < {n}u; ++i) {{",
            "                float acc = Q_DIAGONAL[i] * X[k][i];",
            f"                for (size_t row = 0u; row < {n}u; ++row) {{",
            "                    acc += A[row][i] * lambda[k + 1u][row];",
            "                }",
            "                lambda[k][i] = acc;",
            "            }",
            "        }",
            "        for (size_t k = 0u; k < CONTROLKIT_MPC_HORIZON; ++k) {",
            f"            for (size_t j = 0u; j < {m}u; ++j) {{",
            "                float next_u = U[k][j] - STEP_SIZE * grad[k][j];",
            "                if (next_u < U_MIN[j]) {",
            "                    next_u = U_MIN[j];",
            "                }",
            "                if (next_u > U_MAX[j]) {",
            "                    next_u = U_MAX[j];",
            "                }",
            "                U[k][j] = next_u;",
            "            }",
            "        }",
            "    }",
            f"    for (size_t j = 0u; j < {m}u; ++j) {{",
            "        u[j] = U[0u][j];",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_vector_constant(name: str, values: tuple[float, ...]) -> list[str]:
    rendered = ", ".join(_float_literal(value) for value in values)
    return [f"static const float {name}[{len(values)}u] = {{{rendered}}};"]


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
