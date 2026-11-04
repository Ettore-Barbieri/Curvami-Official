#!/usr/bin/env python3
"""Manual restart check (not a gate case): a gate case stopped part-way,
restarted from its .res file to the end, and compared step by step with the
uninterrupted gate run of the same binary.

    python3 tests/regression/restart_check.py --case al_arccrease --solver build/solver \
        --out tests/regression/out/restart_X --full tests/regression/out/X/al_arccrease.h5

Where each case is cut (neither setting changes the path up to the cut, both
only enter the loop condition):
  al_arccrease   (ArcLength)               max_load 0.021 -> stops after step 8 of 16
  tdal_arccrease (TargetDirectedArcLength) kmax 6         -> stops after step 6 of 12

Before 2026-09-25 (fixed after the core split) W_ext carried a constant offset
after a restart, because x_prev_step was set before RestartManager::load.
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

# case -> (solver key that cuts the run short, value that cuts it)
CUTS = {"al_arccrease": ("max_load", 0.021), "tdal_arccrease": ("kmax", 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="al_arccrease", choices=sorted(CUTS))
    ap.add_argument("--solver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", required=True, help="uninterrupted <case>.h5 from the same binary")
    a = ap.parse_args()

    solver, out = Path(a.solver).resolve(), Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    base = json.loads((HERE / "configs" / f"{a.case}.json").read_text())
    key, cut = CUTS[a.case]
    for part, (value, restart) in enumerate([(cut, False), (base["solver"][key], True)], 1):
        c = json.loads(json.dumps(base))
        c["output_file"] = str(out / "restarted.h5")
        c["restart"] = restart
        c["solver"][key] = value
        p = out / f"part{part}.json"
        p.write_text(json.dumps(c, indent=2))
        with open(out / f"part{part}.log", "w") as log:
            rc = subprocess.run([str(solver), str(p)], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"part {part}: exit {rc}")

    with h5py.File(a.full, "r") as fa, h5py.File(out / "restarted.h5", "r") as fb:
        if sorted(fa) != sorted(fb):
            print("step lists differ")
            sys.exit(1)
        clean = True
        for k in sorted(fa):
            diffs = []
            if rel(fa[k][()], fb[k][()]) > 0:
                diffs.append("data")
            for at in fa[k].attrs:
                if at not in ("git_commit", "columns") and not np.array_equal(fa[k].attrs[at], fb[k].attrs[at]):
                    diffs.append(at)
            if diffs:
                clean = False
                print(f"{k}: differs in {', '.join(diffs)}")
        print("restart reproduces the uninterrupted run" if clean else "restart DIFFERS (see above)")


if __name__ == "__main__":
    main()
