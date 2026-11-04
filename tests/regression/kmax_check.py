#!/usr/bin/env python3
"""Check that kmax no longer truncates LineSearch (fixed 2026-09-25).

Runs al_arccrease with kmax = 10 and compares steps 0..10 with the full gate
run of the same binary. The initial line search of this case needs 53 Newton
iterations: while kmax was also LineSearch's iteration cap, it stopped at 10,
rolled back, and the continuation started from an unconverged state.

    python3 tests/regression/kmax_check.py --solver build/solver \\
        --out tests/regression/out/kmax_X --full tests/regression/out/X/al_arccrease.h5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compare_gate import rel  # noqa: E402

KMAX = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", required=True, help="uninterrupted al_arccrease.h5 from the same binary")
    a = ap.parse_args()

    solver, out = Path(a.solver).resolve(), Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    c = json.loads((HERE / "configs" / "al_arccrease.json").read_text())
    c["output_file"] = str(out / "short.h5")
    c["restart"] = False
    c["solver"]["kmax"] = KMAX
    (out / "short.json").write_text(json.dumps(c, indent=2))
    with open(out / "short.log", "w") as log:
        rc = subprocess.run([str(solver), str(out / "short.json")], cwd=REPO, env=env,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"run: exit {rc}")
    if "LineSearch reached iteration limit" in (out / "short.log").read_text():
        print("initial LineSearch hit its iteration cap")

    with h5py.File(a.full, "r") as ff, h5py.File(out / "short.h5", "r") as fs:
        steps = sorted(k for k in fs if not k.endswith("_creases"))
        worst = 0.0
        for k in steps:
            if k not in ff:
                print(f"{k}: not in the full run")
                worst = np.inf
                continue
            r = rel(ff[k][()], fs[k][()])
            for at in ff[k].attrs:
                if at not in ("git_commit", "columns"):
                    r = max(r, rel(ff[k].attrs[at], fs[k].attrs[at]))
            worst = max(worst, r)
        print(f"{len(steps)} steps, max rel diff vs full run {worst:.3e}")
        print("kmax is a pure step cap" if worst == 0.0 and len(steps) == KMAX + 1
              else "kmax CHANGES the path")


if __name__ == "__main__":
    main()
