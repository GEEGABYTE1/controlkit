# Double Integrator LQR

State is `[position, velocity]`, and control is acceleration. The benchmark measures whether a
small saturated feedback controller stabilizes the state near zero.

Dynamics:

```text
x[k+1] = A x[k] + B u[k]
u[k] = clip(-Kx[k], -2, 2)
```
