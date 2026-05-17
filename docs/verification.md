# Verification

ControlKit verification adds static checks before a controller is treated as deployable. These
checks are intentionally conservative and dependency-free.

## Run Verification

```bash
controlkit verify benchmarks/double_integrator_lqr/controller.yaml
```

Outputs are written to:

```text
outputs/verification/double_integrator_lqr_verification.json
outputs/verification/double_integrator_lqr_verification.md
```

## Checks

### Dimensions

- `A` must be square.
- `B` rows must match `A` rows.
- `K` must be compatible with `u = -Kx`.
- Optional `Q` and `R` matrices must match state and input dimensions.

### Closed-Loop Stability

For feedback `u = -Kx`, ControlKit forms:

```text
A_cl = A - B K
```

For continuous systems, the controller passes when all eigenvalues have negative real parts.

For discrete systems, the controller passes when:

```text
spectral_radius(A_cl) < 1
```

### Constraints

Input and state bounds must have `lower < upper`. If actuator limits are declared, the controller
must also declare a saturation policy or explicit input bounds.

### Numerical Robustness

Verification rejects NaN/Inf matrix entries, computes condition numbers for square matrices, warns
on singular or poorly conditioned matrices, and warns when eigenvalues are close to the stability
boundary.

## Limitations

Verification is not a proof of safety. It does not replace nonlinear simulation, hardware-in-loop
testing, fixed-point analysis, actuator modeling, sensor modeling, or domain-specific certification.
