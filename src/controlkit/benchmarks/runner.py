#Benchmark and generated backends

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from controlkit.backends.c import CBackend
from controlkit.backends.rust import RustBackend
from controlkit.benchmarks.metrics import ClosedLoopMetrics, control_effort, percentile, vector_norm
from controlkit.benchmarks.reporting import write_benchmark_outputs
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
from controlkit.optimization import estimate_operation_count
from controlkit.frontend.specs import _parse_simple_yaml, load_controller_spec


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int = 10_000
    warmup_iterations: int = 1_000
    include_c: bool = True
    include_rust: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency result for one implementation."""

    name: str
    status: str
    iterations: int
    latency_ns_per_call: float | None
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkReport:
    module_name: str
    operation_count_estimate: int
    memory_footprint_bytes: int
    results: tuple[BenchmarkResult, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Report: {self.module_name}",
            "",
            f"- Operation count estimate: `{self.operation_count_estimate}`",
            f"- Memory footprint estimate: `{self.memory_footprint_bytes}` bytes",
            "",
            "| Implementation | Status | Iterations | Latency ns/call | Notes |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for result in self.results:
            latency = (
                ""
                if result.latency_ns_per_call is None
                else f"{result.latency_ns_per_call:.2f}"
            )
            lines.append(
                f"| {result.name} | {result.status} | {result.iterations} | "
                f"{latency} | {result.notes} |"
            )
        lines.append("")
        return "\n".join(lines)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def benchmark_module(module: IRModule, config: BenchmarkConfig | None = None) -> BenchmarkReport:
    resolved_config = config or BenchmarkConfig()
    results = [_benchmark_python(module, resolved_config)]
    if resolved_config.include_c:
        results.append(_benchmark_c(module, resolved_config))
    if resolved_config.include_rust:
        results.append(_benchmark_rust(module, resolved_config))
    return BenchmarkReport(
        module_name=module.name,
        operation_count_estimate=estimate_operation_count(module),
        memory_footprint_bytes=estimate_memory_footprint(module),
        results=tuple(results),
    )


def run_benchmark_case(
    controller_path: Path,
    *,
    output_root: Path = Path("outputs/benchmarks"),
    iterations: int = 200,
    include_generated: bool = True,
) -> ClosedLoopMetrics:
    """Run a closed-loop benchmark case folder and write report artifacts."""

    case_dir = controller_path.parent
    model = _load_model(case_dir / "model.py")
    raw = _parse_simple_yaml(controller_path.read_text(encoding="utf-8"))
    config = model.BENCHMARK
    name = str(config["name"])
    dt = float(config["dt"])
    horizon_steps = int(config["horizon_steps"])
    initial_state = [float(value) for value in config["initial_state"]]
    controller_type = str(raw["policy"])
    state = list(initial_state)
    runtimes_us: list[float] = []
    total_effort = 0.0
    max_norm = vector_norm(state)
    pid_integral = 0.0
    previous_error = 0.0

    for _ in range(horizon_steps):
        start = time.perf_counter_ns()
        if controller_type == "pid":
            control, pid_integral, previous_error = _pid_control(
                raw=raw,
                state=state,
                dt=dt,
                integral=pid_integral,
                previous_error=previous_error,
            )
        else:
            control = _feedback_control(raw=raw, state=state)
        elapsed = time.perf_counter_ns() - start
        runtimes_us.append(elapsed / 1000.0)
        total_effort += control_effort(control, dt)
        state = [float(value) for value in model.step(state, control, dt)]
        max_norm = max(max_norm, vector_norm(state))

    final_norm = vector_norm(state)
    criteria = config.get("pass_criteria", {})
    failures: list[str] = []
    max_final = criteria.get("max_final_state_norm")
    if max_final is not None and final_norm > float(max_final):
        failures.append(f"final_state_norm {final_norm:.6g} > {float(max_final):.6g}")
    max_allowed_norm = criteria.get("max_state_norm")
    if max_allowed_norm is not None and max_norm > float(max_allowed_norm):
        failures.append(f"max_state_norm {max_norm:.6g} > {float(max_allowed_norm):.6g}")
    max_effort = criteria.get("max_control_effort")
    if max_effort is not None and total_effort > float(max_effort):
        failures.append(f"total_control_effort {total_effort:.6g} > {float(max_effort):.6g}")

    generated_runtime = None
    if include_generated and controller_type in {"lqr", "mpc", "rl"}:
        generated_runtime = _generated_runtime_us(controller_path, iterations)

    metrics = ClosedLoopMetrics(
        benchmark_name=name,
        controller_type=controller_type,
        dt=dt,
        horizon_steps=horizon_steps,
        mean_runtime_us=sum(runtimes_us) / len(runtimes_us),
        max_runtime_us=max(runtimes_us),
        p95_runtime_us=percentile(runtimes_us, 95.0),
        final_state_norm=final_norm,
        max_state_norm=max_norm,
        total_control_effort=total_effort,
        passed=not failures,
        failure_reason="; ".join(failures),
        generated_mean_runtime_us=generated_runtime,
    )
    write_benchmark_outputs(
        output_dir=output_root / name,
        metrics=metrics,
        description=str(config["description"]),
        dynamics_equations=str(config["dynamics_equations"]),
        controller_description=str(config["controller_description"]),
        limitations=str(config.get("limitations", "Simple deterministic benchmark model.")),
    )
    return metrics


def run_all_benchmark_cases(
    benchmarks_root: Path = Path("benchmarks"),
    *,
    output_root: Path = Path("outputs/benchmarks"),
    iterations: int = 200,
) -> tuple[ClosedLoopMetrics, ...]:
    cases = []
    for controller_path in sorted(benchmarks_root.glob("*/controller.yaml")):
        cases.append(
            run_benchmark_case(
                controller_path,
                output_root=output_root,
                iterations=iterations,
            )
        )
    return tuple(cases)


def is_benchmark_case_path(path: Path) -> bool:
    return path.name == "controller.yaml" and (path.parent / "model.py").exists()


def estimate_memory_footprint(module: IRModule) -> int:
    matrices = _collect_matrices(module)
    matrix_bytes = sum(matrix.rows * matrix.cols * 4 for matrix in matrices)
    rl_bytes = 0
    for policy in module.rl_policies:
        max_activation_dim = max(layer.output_dim for layer in policy.layers)
        rl_bytes += max_activation_dim * 2 * 4
        for layer in policy.layers:
            if isinstance(layer, LinearLayerIR):
                rl_bytes += layer.input_dim * layer.output_dim * 4
                rl_bytes += layer.output_dim * 4
    mpc_bytes = 0
    for controller in module.mpc_controllers:
        n = controller.state.dim
        m = controller.control.dim
        h = controller.horizon
        diagonal_and_bounds = len(controller.q_diagonal)
        diagonal_and_bounds += len(controller.r_diagonal)
        diagonal_and_bounds += len(controller.q_terminal_diagonal)
        diagonal_and_bounds += len(controller.u_min)
        diagonal_and_bounds += len(controller.u_max)
        scratch_values = (h + 1) * n
        scratch_values += h * m
        scratch_values += (h + 1) * n
        scratch_values += h * m
        mpc_bytes += (diagonal_and_bounds + scratch_values) * 4
    vector_dims = 0
    for law in module.control_laws:
        vector_dims += law.output.dim
        for vector in _collect_vectors(law.expression):
            vector_dims += vector.dim
    return matrix_bytes + vector_dims * 4 + mpc_bytes + rl_bytes


def _load_model(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"benchmark model does not exist: {path}")
    spec = importlib.util.spec_from_file_location(f"controlkit_benchmark_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "BENCHMARK") or not hasattr(module, "step"):
        raise ValueError(f"benchmark model must define BENCHMARK and step(): {path}")
    return module


def _feedback_control(*, raw: dict[str, object], state: list[float]) -> list[float]:
    gain = raw.get("gain_matrix")
    if not isinstance(gain, list):
        raise ValueError("feedback benchmark requires gain_matrix")
    output = [-sum(float(value) * state[col] for col, value in enumerate(row)) for row in gain]
    lower, upper = _control_bounds(raw, len(output))
    return [min(max(output[index], lower[index]), upper[index]) for index in range(len(output))]


def _pid_control(
    *,
    raw: dict[str, object],
    state: list[float],
    dt: float,
    integral: float,
    previous_error: float,
) -> tuple[list[float], float, float]:
    target = float(raw.get("target", 0.0))
    error = target - state[0]
    next_integral = integral + error * dt
    derivative = (error - previous_error) / dt if dt > 0.0 else 0.0
    control = float(raw.get("kp", 0.0)) * error
    control += float(raw.get("ki", 0.0)) * next_integral
    control += float(raw.get("kd", 0.0)) * derivative
    lower, upper = _control_bounds(raw, 1)
    return [min(max(control, lower[0]), upper[0])], next_integral, error


def _control_bounds(raw: dict[str, object], control_dim: int) -> tuple[list[float], list[float]]:
    saturation = raw.get("saturation")
    if isinstance(saturation, dict):
        lower = [float(saturation["lower"]) for _ in range(control_dim)]
        upper = [float(saturation["upper"]) for _ in range(control_dim)]
        return lower, upper
    u_min = raw.get("u_min", [-math.inf for _ in range(control_dim)])
    u_max = raw.get("u_max", [math.inf for _ in range(control_dim)])
    if not isinstance(u_min, list) or not isinstance(u_max, list):
        raise ValueError("u_min and u_max must be lists")
    return [float(value) for value in u_min], [float(value) for value in u_max]


def _generated_runtime_us(controller_path: Path, iterations: int) -> float | None:
    try:
        loaded = load_controller_spec(controller_path)
        report = benchmark_module(
            loaded.module,
            BenchmarkConfig(
                iterations=max(1, iterations),
                warmup_iterations=5,
                include_c=True,
                include_rust=False,
            ),
        )
    except Exception:
        return None
    for result in report.results:
        if result.name == "c" and result.status == "ok" and result.latency_ns_per_call is not None:
            return result.latency_ns_per_call / 1000.0
    return None


def _benchmark_python(module: IRModule, config: BenchmarkConfig) -> BenchmarkResult:
    if module.rl_policies:
        return _benchmark_python_rl(module, config)
    if module.mpc_controllers:
        return _benchmark_python_mpc(module, config)
    law = _select_control_law(module)
    input_vector = _select_input_vector(module, law)
    sample = [1.0 / float(index + 1) for index in range(input_vector.dim)]

    for _ in range(config.warmup_iterations):
        _evaluate_control_law(law, input_vector, sample)

    start = time.perf_counter_ns()
    output = []
    for _ in range(config.iterations):
        output = _evaluate_control_law(law, input_vector, sample)
    elapsed = time.perf_counter_ns() - start
    return BenchmarkResult(
        name="python",
        status="ok",
        iterations=config.iterations,
        latency_ns_per_call=elapsed / config.iterations,
        notes=f"last_output={_format_vector(output)}",
    )


def _benchmark_python_rl(module: IRModule, config: BenchmarkConfig) -> BenchmarkResult:
    if len(module.rl_policies) != 1:
        raise ValueError("benchmarking currently supports exactly one RL policy")
    policy = module.rl_policies[0]
    sample = [1.0 / float(index + 1) for index in range(policy.input_vector.dim)]

    for _ in range(config.warmup_iterations):
        _evaluate_rl_policy(policy.layers, sample)

    start = time.perf_counter_ns()
    output = []
    for _ in range(config.iterations):
        output = _evaluate_rl_policy(policy.layers, sample)
    elapsed = time.perf_counter_ns() - start
    return BenchmarkResult(
        name="python",
        status="ok",
        iterations=config.iterations,
        latency_ns_per_call=elapsed / config.iterations,
        notes=f"last_output={_format_vector(output)}",
    )


def _benchmark_python_mpc(module: IRModule, config: BenchmarkConfig) -> BenchmarkResult:
    if len(module.mpc_controllers) != 1:
        raise ValueError("benchmarking currently supports exactly one MPC controller")
    controller = module.mpc_controllers[0]
    sample = [1.0 / float(index + 1) for index in range(controller.state.dim)]

    for _ in range(config.warmup_iterations):
        _evaluate_mpc_controller(controller, sample)

    start = time.perf_counter_ns()
    output = []
    for _ in range(config.iterations):
        output = _evaluate_mpc_controller(controller, sample)
    elapsed = time.perf_counter_ns() - start
    return BenchmarkResult(
        name="python",
        status="ok",
        iterations=config.iterations,
        latency_ns_per_call=elapsed / config.iterations,
        notes=f"last_output={_format_vector(output)}",
    )


def _benchmark_c(module: IRModule, config: BenchmarkConfig) -> BenchmarkResult:
    cc = shutil.which("cc")
    if cc is None:
        return BenchmarkResult("c", "skipped", 0, None, "cc not found")

    try:
        with TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            artifact = CBackend().generate(module)
            header_path, source_path = artifact.write_to(workdir)
            runner_path = workdir / "bench.c"
            binary_path = workdir / "bench_c"
            runner_path.write_text(
                _render_c_runner(
                    header_name=header_path.name,
                    function_name=f"{_sanitize_identifier(module.name)}_control_step",
                    iterations=config.iterations,
                    warmup_iterations=config.warmup_iterations,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    cc,
                    "-std=c99",
                    "-O2",
                    str(source_path),
                    str(runner_path),
                    "-o",
                    str(binary_path),
                    "-lm",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(binary_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return BenchmarkResult(
                name="c",
                status="ok",
                iterations=config.iterations,
                latency_ns_per_call=float(completed.stdout.strip()),
                notes="compiled with cc -O2",
            )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return BenchmarkResult("c", "error", 0, None, str(exc))


def _benchmark_rust(module: IRModule, config: BenchmarkConfig) -> BenchmarkResult:
    rustc = shutil.which("rustc")
    if rustc is None:
        return BenchmarkResult("rust", "skipped", 0, None, "rustc not found")

    try:
        with TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            artifact = RustBackend().generate(module)
            runner_path = workdir / "bench.rs"
            binary_path = workdir / "bench_rust"
            runner_path.write_text(
                _render_rust_runner(
                    generated_source=artifact.source,
                    iterations=config.iterations,
                    warmup_iterations=config.warmup_iterations,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    rustc,
                    "--edition=2021",
                    "-O",
                    str(runner_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(binary_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return BenchmarkResult(
                name="rust",
                status="ok",
                iterations=config.iterations,
                latency_ns_per_call=float(completed.stdout.strip()),
                notes="compiled with rustc -O",
            )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return BenchmarkResult("rust", "error", 0, None, str(exc))


def _evaluate_control_law(
    law: ControlLaw,
    input_vector: Vector,
    values: list[float],
) -> list[float]:
    env = {input_vector.name: values}
    result = _evaluate_expr(law.expression, env)
    if not isinstance(result, list):
        result = [float(result)]
    return result


def _evaluate_mpc_controller(controller: MpcControllerIR, x: list[float]) -> list[float]:
    n = controller.state.dim
    m = controller.control.dim
    h = controller.horizon
    a_matrix = _as_matrix(controller.a_matrix.values)
    b_matrix = _as_matrix(controller.b_matrix.values)
    u_seq = [[0.0 for _ in range(m)] for _ in range(h)]

    for _ in range(controller.solver_iterations):
        x_seq = [[0.0 for _ in range(n)] for _ in range(h + 1)]
        x_seq[0] = list(x)
        for k in range(h):
            for row in range(n):
                acc = sum(a_matrix[row][col] * x_seq[k][col] for col in range(n))
                acc += sum(b_matrix[row][col] * u_seq[k][col] for col in range(m))
                x_seq[k + 1][row] = acc

        costate = [[0.0 for _ in range(n)] for _ in range(h + 1)]
        for index in range(n):
            costate[h][index] = controller.q_terminal_diagonal[index] * x_seq[h][index]

        grad = [[0.0 for _ in range(m)] for _ in range(h)]
        for k in range(h - 1, -1, -1):
            for col in range(m):
                acc = controller.r_diagonal[col] * u_seq[k][col]
                acc += sum(b_matrix[row][col] * costate[k + 1][row] for row in range(n))
                grad[k][col] = acc
            for index in range(n):
                acc = controller.q_diagonal[index] * x_seq[k][index]
                acc += sum(a_matrix[row][index] * costate[k + 1][row] for row in range(n))
                costate[k][index] = acc

        for k in range(h):
            for col in range(m):
                next_u = u_seq[k][col] - controller.step_size * grad[k][col]
                next_u = min(max(next_u, controller.u_min[col]), controller.u_max[col])
                u_seq[k][col] = next_u

    return list(u_seq[0])


def _evaluate_rl_policy(
    layers: tuple[LinearLayerIR | ActivationLayerIR, ...],
    values: list[float],
) -> list[float]:
    current = list(values)
    for layer in layers:
        if isinstance(layer, LinearLayerIR):
            current = [
                layer.bias[row]
                + sum(layer.weights[row][col] * current[col] for col in range(layer.input_dim))
                for row in range(layer.output_dim)
            ]
        elif layer.kind.value == "relu":
            current = [max(0.0, value) for value in current]
        else:
            current = [math.tanh(value) for value in current]
    return current


def _evaluate_expr(expr: Expr, env: dict[str, list[float]]) -> float | list[float]:
    if isinstance(expr, Vector):
        return env[expr.name]
    if isinstance(expr, Matrix):
        if expr.values is None:
            raise ValueError(f"matrix {expr.name} has no values")
        return [list(row) for row in expr.values]  # type: ignore[return-value]
    if isinstance(expr, ScalarConstant):
        return expr.value
    if isinstance(expr, Zero):
        return [0.0 for _ in range(expr.shape.rows)]
    if isinstance(expr, Neg):
        value = _as_vector(_evaluate_expr(expr.value, env))
        return [-item for item in value]
    if isinstance(expr, ScalarMul):
        scalar = float(_evaluate_expr(expr.scalar, env))
        value = _as_vector(_evaluate_expr(expr.value, env))
        return [scalar * item for item in value]
    if isinstance(expr, MatVecMul):
        matrix = _as_matrix(_evaluate_expr(expr.matrix, env))
        vector = _as_vector(_evaluate_expr(expr.vector, env))
        return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]
    if isinstance(expr, Add):
        left = _as_vector(_evaluate_expr(expr.left, env))
        right = _as_vector(_evaluate_expr(expr.right, env))
        return [left[index] + right[index] for index in range(len(left))]
    if isinstance(expr, Sub):
        left = _as_vector(_evaluate_expr(expr.left, env))
        right = _as_vector(_evaluate_expr(expr.right, env))
        return [left[index] - right[index] for index in range(len(left))]
    if isinstance(expr, Clip):
        value = _as_vector(_evaluate_expr(expr.value, env))
        lower = float(_evaluate_expr(expr.lower, env))
        upper = float(_evaluate_expr(expr.upper, env))
        return [min(max(item, lower), upper) for item in value]
    raise ValueError(f"unsupported expression: {type(expr).__name__}")


def _render_c_runner(
    *,
    header_name: str,
    function_name: str,
    iterations: int,
    warmup_iterations: int,
) -> str:
    return "\n".join(
        [
            "#define _POSIX_C_SOURCE 199309L",
            "#include <stdio.h>",
            "#include <time.h>",
            f'#include "{header_name}"',
            "",
            "static double elapsed_ns(struct timespec start, struct timespec end) {",
            "    return (double)(end.tv_sec - start.tv_sec) * 1000000000.0 +",
            "        (double)(end.tv_nsec - start.tv_nsec);",
            "}",
            "",
            "int main(void) {",
            "    float x[CONTROLKIT_STATE_DIM];",
            "    float u[CONTROLKIT_CONTROL_DIM];",
            "    volatile float sink = 0.0f;",
            "    for (size_t i = 0u; i < CONTROLKIT_STATE_DIM; ++i) {",
            "        x[i] = 1.0f / (float)(i + 1u);",
            "    }",
            f"    for (size_t i = 0u; i < {warmup_iterations}u; ++i) {{",
            f"        {function_name}(x, u);",
            "        sink += u[0];",
            "    }",
            "    struct timespec start;",
            "    struct timespec end;",
            "    clock_gettime(CLOCK_MONOTONIC, &start);",
            f"    for (size_t i = 0u; i < {iterations}u; ++i) {{",
            f"        {function_name}(x, u);",
            "        sink += u[0];",
            "    }",
            "    clock_gettime(CLOCK_MONOTONIC, &end);",
            "    if (sink == 1234567.0f) {",
            '        printf("0\\n");',
            "        return 0;",
            "    }",
            f'    printf("%.6f\\n", elapsed_ns(start, end) / (double){iterations}u);',
            "    return 0;",
            "}",
            "",
        ]
    )


def _render_rust_runner(
    *,
    generated_source: str,
    iterations: int,
    warmup_iterations: int,
) -> str:
    source = "\n".join(
        line for line in generated_source.splitlines() if line.strip() != "#![no_std]"
    )
    return "\n".join(
        [
            source,
            "",
            "fn main() {",
            "    let mut x = [0.0_f32; STATE_DIM];",
            "    let mut u = [0.0_f32; CONTROL_DIM];",
            "    let mut i = 0usize;",
            "    while i < STATE_DIM {",
            "        x[i] = 1.0_f32 / ((i + 1) as f32);",
            "        i += 1;",
            "    }",
            "    let mut sink = 0.0_f32;",
            f"    let mut warmup = 0usize;",
            f"    while warmup < {warmup_iterations} {{",
            "        control_step(&x, &mut u);",
            "        sink += u[0];",
            "        warmup += 1;",
            "    }",
            "    let start = std::time::Instant::now();",
            "    let mut iter = 0usize;",
            f"    while iter < {iterations} {{",
            "        control_step(&x, &mut u);",
            "        sink += u[0];",
            "        iter += 1;",
            "    }",
            "    let elapsed = start.elapsed().as_nanos() as f64;",
            "    if sink == 1234567.0_f32 {",
            '        println!("0");',
            "        return;",
            "    }",
            f'    println!("{{:.6}}", elapsed / ({iterations} as f64));',
            "}",
            "",
        ]
    )


def _select_control_law(module: IRModule) -> ControlLaw:
    if len(module.control_laws) != 1:
        raise ValueError("benchmarking currently supports exactly one control law")
    return module.control_laws[0]


def _select_input_vector(module: IRModule, law: ControlLaw) -> Vector:
    vectors = {
        vector.name: vector
        for vector in _collect_vectors_from_expr(law.expression)
        if vector.name != law.output.name
    }
    if not vectors and "state_dim" in module.metadata:
        return Vector(module.metadata.get("state_name", "x"), dim=int(module.metadata["state_dim"]))
    if len(vectors) != 1:
        raise ValueError("benchmarking currently supports exactly one input vector")
    return next(iter(vectors.values()))


def _collect_matrices(module: IRModule) -> tuple[Matrix, ...]:
    matrices: dict[str, Matrix] = {}
    for law in module.control_laws:
        for matrix in _collect_matrices_from_expr(law.expression):
            matrices[matrix.name] = matrix
    for controller in module.mpc_controllers:
        matrices[controller.a_matrix.name] = controller.a_matrix
        matrices[controller.b_matrix.name] = controller.b_matrix
    return tuple(matrices[name] for name in sorted(matrices))


def _collect_matrices_from_expr(expr: Expr) -> tuple[Matrix, ...]:
    return tuple(child for child in _walk_expr(expr) if isinstance(child, Matrix))


def _collect_vectors(law_expr: Expr) -> tuple[Vector, ...]:
    return tuple(child for child in _walk_expr(law_expr) if isinstance(child, Vector))


def _collect_vectors_from_expr(expr: Expr) -> tuple[Vector, ...]:
    return tuple(child for child in _walk_expr(expr) if isinstance(child, Vector))


def _walk_expr(expr: Expr) -> tuple[Expr, ...]:
    children: list[Expr] = [expr]
    if isinstance(expr, MatVecMul):
        children.extend(_walk_expr(expr.matrix))
        children.extend(_walk_expr(expr.vector))
    elif isinstance(expr, Neg):
        children.extend(_walk_expr(expr.value))
    elif isinstance(expr, ScalarMul):
        children.extend(_walk_expr(expr.scalar))
        children.extend(_walk_expr(expr.value))
    elif isinstance(expr, Add | Sub):
        children.extend(_walk_expr(expr.left))
        children.extend(_walk_expr(expr.right))
    elif isinstance(expr, Clip):
        children.extend(_walk_expr(expr.value))
        children.extend(_walk_expr(expr.lower))
        children.extend(_walk_expr(expr.upper))
    return tuple(children)


def _as_vector(value: float | list[float]) -> list[float]:
    if isinstance(value, float):
        return [value]
    return value


def _as_matrix(value: object) -> list[list[float]]:
    return value  # type: ignore[return-value]


def _format_vector(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6g}" for value in values) + "]"


def _sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value]
    sanitized = "".join(chars)
    if not sanitized:
        raise ValueError("identifier cannot be empty")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
