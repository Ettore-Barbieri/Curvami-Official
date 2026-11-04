#!/usr/bin/env python3
"""Run the core-split regression gate with a given solver binary.

    python3 tests/regression/run_gate.py --solver build/solver --out tests/regression/out/phase1

Run from the repository root (the configs use repo-relative mesh paths). Each case in
configs/ is copied into --out with output_file redirected there, so the baseline and
every phase write to their own folder and nothing is ever re-used across builds.

Threads are pinned to 1: the multithreaded solver is not bit-reproducible run to run
(OpenMP assembly and threaded BLAS reductions), single-threaded runs are. That makes
"identical" a meaningful verdict in compare_gate.py.

Shortening rule: the configs were written when solver.kmax was also LineSearch's
iteration cap, so where LineSearch runs (ArcLength's initial solve,
KinematicDeployment, LineSearch) they shorten through max_load / angle_target, not
kmax. Since 2026-09-25 LineSearch has its own cap (ls_max_iters); the configs are
left as they are so every phase stays comparable with the baseline.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, help="solver binary to test")
    ap.add_argument("--out", required=True, help="output folder (must not exist, or use --force)")
    ap.add_argument("--force", action="store_true", help="delete --out first")
    ap.add_argument("cases", nargs="*", help="case names (default: all in configs/)")
    a = ap.parse_args()

    solver = Path(a.solver).resolve()
    out = Path(a.out).resolve()
    if out.exists():
        if not a.force:
            sys.exit(f"{out} exists. Outputs are never re-used across builds; pass --force to delete it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cases = a.cases or sorted(p.stem for p in (HERE / "configs").glob("*.json"))
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    manifest = {"solver": str(solver),
                "solver_sha256": hashlib.sha256(solver.read_bytes()).hexdigest(),
                "cases": {}}
    for name in cases:
        cfg = json.loads((HERE / "configs" / f"{name}.json").read_text())
        cfg["output_file"] = str(out / f"{name}.h5")
        cfg["restart"] = False
        cfg_path = out / f"{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        print(f"[gate] {name} ...", flush=True)
        t0 = time.time()
        with open(out / f"{name}.log", "w") as log:
            rc = subprocess.run([str(solver), str(cfg_path)], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        print(f"[gate] {name}: exit {rc}, {dt:.0f} s", flush=True)
        manifest["cases"][name] = {"exit": rc, "seconds": round(dt, 1)}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
