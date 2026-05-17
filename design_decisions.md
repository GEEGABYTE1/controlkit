# Design Decisions

## 001: Use a `src/` Layout

ControlKit uses a `src/controlkit` package layout to prevent tests from accidentally importing from
the repository root instead of the installed package.

## 002: Keep Frontends Separate From Backends

PID, LQR, MPC, and RL policy support will enter through policy frontends. C and Rust code generation
will live behind backend target contracts. This keeps policy semantics separate from embedded output
concerns.

## 003: Introduce IR Before Optimization

The compiler will lower every policy into a backend-neutral IR before optimization or code
generation. This gives the project one place to represent timing, state dimensions, constraints,
precision, and memory layout.

## 004: Placeholder Behavior Must Be Honest

The CLI may validate inputs now, but it must clearly report that compilation is not implemented.
This avoids misleading users while preserving a stable command shape for future work.

## 005: IR Nodes Validate Shape at Construction

ControlKit IR nodes validate types and shapes when they are constructed. Invalid expressions such
as `Kx` with mismatched dimensions, addition of differently shaped vectors, or linear systems with
incorrect `A`/`B` matrix dimensions fail before optimization or backend code generation.

## 006: Represent Control Math Symbolically Before Numeric Storage

The first IR phase represents vectors, matrices, scalar constants, and operations symbolically.
Concrete numeric storage formats are deferred until schema validation and backend requirements are
clear. This keeps the IR focused on compiler semantics: shape, value kind, dynamics form, and
control-law structure.

## 007: Linear Dynamics Are First-Class IR Objects

Linear systems are represented explicitly as continuous `x_dot = Ax + Bu` or discrete
`x_next = Ax + Bu` dynamics. Treating dynamics as a first-class object lets future passes reason
about sample time, discretization, stability checks, memory layout, and backend-specific code
generation without reverse-engineering generic expression trees.

## 008: Maintain a Phase-Aligned Textbook

ControlKit includes an Obsidian-compatible textbook that is updated alongside each implementation
phase. The textbook should explain the concepts introduced by the code, use real project examples,
and link related ideas so the project remains teachable as it grows from IR design through
frontends, backends, optimization, benchmarking, MPC, and RL policy compilation.

## 009: LQR Frontend Owns Numeric Gain Validation

The Phase 2 LQR frontend accepts a numeric gain matrix and validates its rectangular shape,
controller dimensions, optional saturation, and naming metadata before lowering into IR. The current
IR remains symbolic, so the frontend preserves numeric-gain intent through validated dimensions and
metadata while future backend phases decide how numeric parameters are stored and emitted.

## 010: LQR Lowering Emits Negative Feedback IR

An LQR controller lowers to a `ControlLaw` whose expression is `-(K @ x)`, optionally wrapped in
`Clip` when saturation is configured. This keeps the compiler-facing representation aligned with
standard linear state feedback while giving later code generators a small, predictable expression
tree.

## 011: C Backend Requires Numeric Matrix Values

Phase 3 extends symbolic IR matrices with optional numeric values. The C backend requires those
values for code generation and rejects purely symbolic matrices. This keeps Phase 1 symbolic use
cases intact while giving backend phases a validated path to emit standalone coefficient arrays.

## 012: Generate Deterministic C Before Aggressive Optimization

The first C backend emits straightforward float32 code with explicit temporaries, fixed-size arrays,
matrix-vector loops, and scalar saturation branches. The output is intentionally deterministic and
readable before later optimization phases add simplification, unrolling, or operation-count driven
rewrites.

## 013: Rust Backend Uses a no_std-Compatible Surface

The Phase 4 Rust backend emits a single `.rs` source file with `#![no_std]`, fixed-size arrays,
module-level constants, and a `control_step` function that writes into a caller-provided mutable
output buffer. This keeps the generated code friendly to embedded integration without requiring a
Rust embedded runtime yet.

## 014: Rust and C Backends Share IR Semantics, Not Implementation

The Rust backend mirrors the C backend's supported IR semantics but owns its own emitter. This
allows Rust-specific choices, such as array references, mutable output buffers, and `usize` loop
indices, without forcing both target languages through a lowest-common-denominator code generator.

## 015: Optimization Passes Preserve IR Invariants

Phase 5 optimization rewrites expressions by constructing new validated IR nodes instead of
mutating trees in place. This keeps shape and type invariants enforced after simplification, so
later backends can trust optimized modules the same way they trust frontend-lowered modules.

## 016: Loop Unrolling Is a Backend Option

Loop unrolling is visible in generated target code, so Phase 5 exposes it as an optional C/Rust
backend setting rather than encoding unrolled loops inside the mathematical IR. The default remains
looped, deterministic output; users can request unrolled code for small fixed-size controllers.

## 017: Benchmark Reports Are Toolchain-Aware

Phase 6 benchmark reports always include Python reference latency and static estimates, then attempt
generated C and Rust latency only when the local toolchain is available. Missing `cc` or `rustc`
does not fail the benchmark; it records a skipped backend result so reports remain portable.

## 018: Benchmarking Produces JSON and Markdown

Benchmark results are serialized to both machine-readable JSON and review-friendly Markdown. This
keeps reports useful for automated comparisons while still being easy to inspect during research
and engineering iteration.

## 019: CLI Specs Use a Small Internal YAML Subset

Phase 7 loads project-owned controller YAML specs without adding a runtime YAML dependency. The
parser intentionally supports the small structured subset needed by current LQR specs: mappings,
inline lists, and block lists. Broader YAML support can be added later if user specs require it.

## 020: CLI Commands Lower Through the Same Frontend Path

The CLI does not bypass frontend validation. `inspect`, `validate`, `compile`, and `benchmark` all
load the YAML spec through the frontend loader and lower it into ControlKit IR before doing their
command-specific work.

## 021: MPC-lite Uses Projected Gradient Over a Control Sequence

Phase 8 implements a deliberately small finite-horizon MPC solver: each `control_step` call
zero-initializes the full control sequence, rolls out discrete dynamics, backpropagates costates,
and applies projected-gradient updates to the sequence. The generated controller returns only the
first control input `U[0]`, matching receding-horizon MPC while keeping runtime state out of the
generated function interface.

## 022: MPC Costs Are Diagonal in Phase 8

MPC-lite accepts diagonal `q`, `r`, and optional `q_terminal` vectors rather than dense quadratic
cost matrices. This keeps validation, code generation, operation counting, and embedded memory use
simple while preserving the core finite-horizon optimization structure needed for later MPC work.

## 023: MPC Constraints Are Input Box Bounds

Phase 8 represents constraints as elementwise `u_min` and `u_max` vectors. Generated C and Rust
apply projection by clipping each updated control value after every gradient step. General state
constraints and coupled constraints are deferred until the solver architecture is broader.

## 024: RL Weights Use JSON in Phase 9

Phase 9 accepts RL policy weights from JSON only. JSON keeps the implementation dependency-free,
easy to review in examples, and compatible with the current lightweight CLI loader. NPZ, ONNX, and
framework-native formats are deferred until the compiler has a broader model-import layer.

## 025: RL Compilation Targets Fixed-Shape MLPs

The first RL compiler path supports small feed-forward MLPs with `linear`, `relu`, and `tanh`
layers. Every layer validates its input and output dimensions before backend generation, so C and
Rust can emit fixed-size arrays and deterministic loops without runtime shape checks.

## 026: Tanh Is Target-Specific

Generated C uses `<math.h>` and `tanhf` for Tanh activations. Generated Rust remains
`#![no_std]` by emitting a small deterministic approximation instead of depending on `std` or an
external math crate. Numerical equivalence tests should account for this backend choice.

## 027: Phase 10 Website Is Static And Repo-Native

The public-facing ControlKit page is deployed at `https://controlkit.vercel.app/` from the static
HTML/CSS/JavaScript artifact under `website/`. Keeping the source repo-native makes the product
page easy to review locally while the deployed URL gives GitHub visitors a stable entry point.

## 028: Product Messaging And Technical Detail Share One Page

The website intentionally mixes product positioning with implementation detail. The first viewport
explains what ControlKit is and links to GitHub, while lower sections document the compiler
pipeline, current policy support, code generation, benchmarks, and demo commands.
