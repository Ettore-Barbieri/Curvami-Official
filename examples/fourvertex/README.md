# Square twist (four-vertex folding paper)

These are two cases of the square-twist study in the four-vertex folding paper. Both
use the 61 × 61 background grid with 30 nodes per fold (`SquareTwistFiner.msh`)
and the `TargetDirectedArcLength` analysis. The four creases are tagged T = 6,
R = 5, B = 8, L = 7.

| File | Drivers | K_t | Solver profile | What it shows |
|---|---|---|---|---|
| `ST-nx61nf30-TRBL-Kt5-p0.json` | all four | 5 | `p0`: ds 0.1, dsmax 0.75, up to 500 steps | Four-driver actuation jams near 135°, well short of flat. |
| `ST-nx61nf30-T__L-Kt0p7-p1v.json` | T and L; B and R passive | 0.7 | `p1v`: ds 0.01, dsmax 0.02, up to 1000 steps, snap-back veto on | Inside the stiffness window, the driven creases fold almost flat and the passive creases engage. |

**Run** (from the repository root; these are long runs, hours rather than minutes):

```bash
OMP_NUM_THREADS=4 build/solver examples/fourvertex/ST-nx61nf30-TRBL-Kt5-p0.json
```

Outputs go to `examples/fourvertex/out/`.
