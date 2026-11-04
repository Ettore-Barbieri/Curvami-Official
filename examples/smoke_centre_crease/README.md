# Smoke test: a crease grown in a 20 × 20 sheet

This is the same set-up as `examples/paperA_centre_crease/` (ξ = 1, θ₀* = 0.5°,
h = 0.005, the same supports and tolerances) on a 20 × 20 grid. The crease grows
from the centre to a = 0.5 in 5 steps. There are no domain integrals.

It checks that the build runs; the mesh is too coarse for the results to mean
anything physically.

```bash
build/solver examples/smoke_centre_crease/smoke.json
```

It takes about 5 s. Every step should converge within 5 Newton iterations. The
last row of `examples/smoke_centre_crease/out/smoke_steps.csv` should read, to
round-off:

| step | E_mem | E_ben | E_hin | mean θ (deg) |
|---|---|---|---|---|
| 5 | 4.006e-08 | 9.404e-06 | 1.594e-05 | 0.1962 |

`CB_n20.msh` was generated from `CB_n20.geo` with:

```bash
gmsh CB_n20.geo -2 -format msh41 -o CB_n20.msh
```
