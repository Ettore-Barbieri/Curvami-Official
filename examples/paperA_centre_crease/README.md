# Centre crease grown in a square sheet (Paper A, Section 8)

These are the inputs of the centre-crease runs in *Creases in the light of fracture
mechanics*, Section 8.

**Set-up**
- **Sheet:** [−1, 1]², 80 × 80 cells (`CB_n80.geo`, `CB_n80.msh`). E = 9.6 × 10⁷,
  ν = 0.3, h = 0.005, so D = 1.0989.
- **Crease:** along y = 0, on a node row. It is grown symmetrically from the centre
  (`ProgressiveCreasing`, `middle_out`), one element per step at each end, from
  half-length a = 0.025 to a = 0.5. That is 20 steps.
- **Hinge:** rest angle θ₀* = 0.5°, stiffness K_t = ξ D / L with L = 1. The three
  files are ξ = 1, 10 and 100.
- **Supports:** three corner supports that remove only the rigid-body motions:
  (−1, −1) holds u, v, w; (1, −1) holds v, w; (−1, 1) holds w.
- **Newton tolerances:** 1 × 10⁻¹⁰ (force) and 1 × 10⁻¹² (energy).
- **Domain integrals:** the `j_integral` objects evaluate the domain form of the
  energy release rate at a = 0.25–0.4, at both tips.

**Run** (from the repository root; about 15 min each with 4 threads):

```bash
OMP_NUM_THREADS=4 build/solver examples/paperA_centre_crease/CB_n80_t0.5_xi10.json
```

**Outputs** go to `examples/paperA_centre_crease/out/`:
- `<run>.h5`: the fields at the face Gauss points, one dataset per step.
- `<run>_steps.csv`: per step, a, Newton iterations, E_mem, E_ben, E_hin and the
  fold angles.
- `<run>_G*.csv`: the domain-integral results.

The stored energy is Π = E_mem + E_ben + E_hin. Step k has a = 0.025 k.
