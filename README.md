# Curvami

Curvami is a meshfree solver for thin plates and shells with folds. It models
Kirchhoff–Love plates and shells under large rotations, using reproducing-kernel
(RKPM) shape functions. A crease is a cut in the approximation that carries an
elastic or rigid hinge. Creases may be straight or curved, and a curved crease can
cross the node layout freely (but see *Known limitations* for straight creases).

The code is C++17, header-only apart from `main.cpp`. A run is driven by a JSON
input file and a Gmsh mesh, and writes its results to HDF5.

## Build

Requirements, as used for the release (macOS 27, Apple M2, Homebrew):

| Dependency | Version |
|---|---|
| Apple clang | 21.0.0 |
| CMake | 4.3.2 (≥ 3.18 required) |
| Eigen | 5.0.1 |
| HDF5 (C++ API) | 2.2.0 |
| MUMPS, sequential (`coin-or-tools/coinor/mumps-seq`) | 5.5.1 |
| SuiteSparse | 7.12.2 |
| OpenMP (`libomp`) | 22.1.8 |
| nlohmann-json | 3.12.0 |
| Gmsh (to regenerate meshes only) | 4.15.2 |

```bash
brew install cmake eigen hdf5 suite-sparse libomp nlohmann-json gmsh
brew tap coin-or-tools/coinor && brew install coin-or-tools/coinor/mumps-seq
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build -j
```

This produces `build/solver`. The Homebrew prefixes are the defaults. Override them
with `-DHDF5_ROOT=…`, `-DCURVAMI_SUITESPARSE_ROOT=…`, `-DCURVAMI_MUMPS_ROOT=…` or
`-DCURVAMI_EXTRA_PREFIX=…` for another layout.

On Apple clang the Release build passes `-mcpu=apple-m4` (see `CMakeLists.txt`).
That is the setting the published results were computed with. On an older
toolchain that does not know this CPU, change the flag to `apple-m1` or `native`.
Results then agree to round-off, not bit for bit.

The generic core (`core/`, no plate code and no HDF5) can also be used on its own,
through `add_subdirectory`; `tests/consumer/` is a minimal example.

## Run

Run from the repository root, since the input files use repository-relative paths:

```bash
build/solver examples/smoke_centre_crease/smoke.json
```

Each binary stamps the git commit it was built from into every `.h5` it writes
(attribute `git_commit` on each step), so a result can always be traced to its
code. A tree with uncommitted changes to the solver sources is stamped `-dirty`,
and a build outside a git checkout (for example from the Zenodo archive) is stamped
`unknown`.

The thread count follows `OMP_NUM_THREADS`. Multithreaded runs agree to round-off;
only single-threaded runs are bit-reproducible.

Analysis types (`"analysis_type"` in the input file): `LineSearch`, `ArcLength`,
`TargetDirectedArcLength`, `KinematicDeployment`, `KinematicArcLength`,
`ProgressiveCreasing`, `Landscape`, `Initial`, `TangentCheck`.

## Examples

| Folder | What it is | Run time |
|---|---|---|
| `examples/smoke_centre_crease/` | A crease grown in a 20 × 20 sheet, as a smoke test | ≈ 5 s |
| `examples/paperA_centre_crease/` | The centre-crease runs of Paper A, Section 8 | ≈ 15 min each |
| `examples/fourvertex/` | The square-twist cases of the four-vertex folding paper | hours |

Each folder has its own README.

## Known limitations

- **A straight crease must lie on a node row.** If a straight crease runs parallel
  to a row of nodes but between two rows, Gauss points can fall exactly on it, lose
  their side, and silently tie the two facets together: the sheet then bends
  instead of folding. Put straight creases on a node row (the example meshes do).
- **Fold angles near ±180°** need care: report results only up to the last step
  where the local fold angle stays at or below 180°, since beyond it the facets have
  rotated past each other.

## Tests

`tests/regression/` holds a regression gate: five short runs that compare two
builds field by field. See its README.

## How to cite

If you use Curvami, please cite the software and the papers that describe it:

- E. Barbieri, *Creases in the light of fracture mechanics*, submitted to the
  Journal of the Mechanics and Physics of Solids.
- E. Barbieri, the four-vertex folding paper (under review; reference to follow).
- Curvami, version 1.0.0, Zenodo, DOI to follow.

`CITATION.cff` has the machine-readable version.

## Licence

MIT; see `LICENSE`.
