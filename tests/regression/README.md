# Regression gate

These are five short runs that cover the main analysis types. They are used to check
that a change to the code does not change its results.

| Case | Analysis | Mesh | Shortened by |
|---|---|---|---|
| `al_arccrease` | ArcLength (force on a curved crease, fold pre-set to 30°) | `Arc_LR0p4_Phi30.msh` | `max_load` 0.05 |
| `tdal_arccrease` | TargetDirectedArcLength, curved crease | `Arc_LR0p4_Phi30.msh` | `kmax` 12 |
| `tdal_squaretwist` | TargetDirectedArcLength, 4 drivers, 2 cuts | `SquareTwist.msh` | `kmax` 10 |
| `kd_miura` | KinematicDeployment | `Miura.msh` | `angle_target` 20°, `num_steps` 10 |
| `ls_miura` | LineSearch, rigid folds | `Miura.msh` | — |

The five cases take about 18 minutes single-threaded on an Apple M2.

## Use

Run from the repository root. First build and run a reference binary, for example
the release tag:

```bash
python3 tests/regression/run_gate.py --solver build_ref/solver --out tests/regression/out/reference
```

Then do the same with the candidate build:

```bash
python3 tests/regression/run_gate.py --solver build/solver --out tests/regression/out/candidate
```

Then compare the two:

```bash
python3 tests/regression/compare_gate.py tests/regression/out/reference tests/regression/out/candidate
```

`run_gate.py` pins every run to one thread, so the same code gives bit-identical
output. The verdict per case is one of:
- `IDENTICAL`;
- `ROUND-OFF`: at most 1e-9 relative;
- a failure.

The script needs `h5py` and `numpy`.

Two manual checks sit next to the gate:
- `restart_check.py`: a restarted run must reproduce the uninterrupted one.
- `kmax_check.py`: a shorter `kmax` must only cut a run short, not change it.
