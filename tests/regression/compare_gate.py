#!/usr/bin/env python3
"""Compare two regression-gate output folders written by run_gate.py.

    python3 tests/regression/compare_gate.py tests/regression/out/reference tests/regression/out/candidate

Per case it compares:
  - the step list (same steps, same count),
  - every step dataset (Gauss-point positions, displacements, rotations, fields),
    every per-crease dataset (pointwise fold angles), and every numeric attribute
    (fold angles / load factor, disp, W_ext, energies, planarity),
  - the final state vector x in the .res restart file, when there is one,
  - the solver path in the log: load factor, disp and ds per step, residuals at
    convergence, KinematicDeployment step lines.

Differences are relative to the magnitude of the baseline array (max |a-b| / max(|a|, tiny)),
per column for the 2-D step and crease datasets.
Verdicts: IDENTICAL (bitwise), ROUND-OFF (<= --tol), FAIL. Exit status 1 on any FAIL.
The git_commit stamp is expected to differ and is only printed.
Attributes listed in NEW_ATTRS (added after the baseline) are ignored when they
appear in the second folder only.
"""
import argparse
import re
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

# Attributes added to the step datasets after the baseline was written. They are
# allowed in the second folder only and are not compared (the baseline has no value
# for them); every baseline attribute must still be present and equal.
#   Fext_dot_x  (460d645)  F_ext . x of the unit load, for the total potential energy
NEW_ATTRS = {"Fext_dot_x"}

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
PATH_LINE = re.compile(r"LOAD STEP|Convergence!|Global Step|Yielded Angle|Lambda")


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    if np.array_equal(a, b, equal_nan=True):
        return 0.0
    # Per column for 2-D datasets: coordinates (~1) would otherwise swamp a
    # displacement or strain column (~1e-3) and hide a real change in it.
    if a.ndim == 2:
        scale = np.maximum(np.nanmax(np.abs(a), axis=0), 1e-300)
        return float(np.nanmax(np.nanmax(np.abs(a - b), axis=0) / scale))
    scale = max(np.nanmax(np.abs(a)), 1e-300)
    return float(np.nanmax(np.abs(a - b)) / scale)


def read_res(p):
    raw = p.read_bytes()
    # RestartManager: magic, version, k, lambda, ds, dl_old, W_ext, ns, x[ns], dU[ns].
    # Locate ns by trying the magic lengths the format could have (8 is current).
    for m in (8, 4, 16):
        off = m + 4 + 4 + 8 * 4
        if off + 4 > len(raw):
            continue
        ns = struct.unpack_from("i", raw, off)[0]
        if ns > 0 and off + 4 + 16 * ns == len(raw):
            x = np.frombuffer(raw, dtype=np.float64, count=ns, offset=off + 4)
            head = struct.unpack_from("i d d d d", raw, m + 4)
            return head, x
    return None, None


def path_numbers(log):
    rows = [l for l in log.read_text(errors="replace").splitlines() if PATH_LINE.search(l)]
    return rows, [list(map(float, NUM.findall(r))) for r in rows]


def compare_case(name, A, B, tol):
    worst = 0.0
    notes = []

    def acc(label, r):
        nonlocal worst
        worst = max(worst, r)
        if r > tol:
            notes.append(f"    {label}: rel diff {r:.3e}")

    ha, hb = A / f"{name}.h5", B / f"{name}.h5"
    if not ha.exists() or not hb.exists():
        return "FAIL", [f"    missing h5: {ha.exists()=} {hb.exists()=}"], np.inf, ""
    with h5py.File(ha, "r") as fa, h5py.File(hb, "r") as fb:
        ka, kb = sorted(fa.keys()), sorted(fb.keys())
        if ka != kb:
            only_a = sorted(set(ka) - set(kb))
            only_b = sorted(set(kb) - set(ka))
            return "FAIL", [f"    step lists differ: only in A {only_a[:5]}, only in B {only_b[:5]}"], np.inf, ""
        stamps = set()
        for k in ka:
            acc(f"{k} data", rel(fa[k][()], fb[k][()]))
            aa, ab = fa[k].attrs, fb[k].attrs
            if set(aa.keys()) != set(ab.keys()) - NEW_ATTRS:
                notes.append(f"    {k}: attribute sets differ")
                worst = np.inf
            for at in aa.keys():
                if at in ("git_commit", "columns"):
                    if at == "git_commit":
                        sa, sb = aa[at], ab.get(at)
                        stamps.add((sa.decode() if isinstance(sa, bytes) else str(sa),
                                    sb.decode() if isinstance(sb, bytes) else str(sb)))
                    continue
                acc(f"{k}.{at}", rel(aa[at], ab[at]))
        nsteps = sum(1 for k in ka if not k.endswith("_creases"))

    ra, rb = A / f"{name}.res", B / f"{name}.res"
    if ra.exists() != rb.exists():
        notes.append("    .res present in only one run")
        worst = np.inf
    elif ra.exists():
        (hA, xA), (hB, xB) = read_res(ra), read_res(rb)
        if xA is None or xB is None:
            notes.append("    .res unreadable")
            worst = np.inf
        else:
            acc(".res header (k, lambda, ds, dl, W_ext)", rel(hA, hB))
            acc(".res final x", rel(xA, xB))

    rowsA, pa = path_numbers(A / f"{name}.log")
    rowsB, pb = path_numbers(B / f"{name}.log")
    if len(pa) != len(pb) or any(len(x) != len(y) for x, y in zip(pa, pb)):
        notes.append(f"    solver path differs in shape: {len(pa)} vs {len(pb)} path lines")
        for x, y in zip(rowsA, rowsB):
            if x != y:
                notes.append(f"      first difference:\n        A: {x.strip()}\n        B: {y.strip()}")
                break
        worst = np.inf
    else:
        for i, (x, y) in enumerate(zip(pa, pb)):
            r = rel(x, y)
            if r > tol:
                acc(f"log path line {i}: {rowsA[i].strip()[:60]}", r)
            worst = max(worst, r)

    stamp_txt = ", ".join(f"{a} -> {b}" for a, b in sorted(stamps))
    if worst == 0.0:
        verdict = "IDENTICAL"
    elif worst <= tol:
        verdict = "ROUND-OFF"
    else:
        verdict = "FAIL"
    return verdict, notes, worst, f"{nsteps} steps, stamp {stamp_txt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="max relative difference accepted as round-off (default 1e-9)")
    a = ap.parse_args()
    A, B = Path(a.baseline), Path(a.candidate)

    cases = sorted(p.stem for p in A.glob("*.h5"))
    missing = sorted(p.stem for p in (Path(__file__).parent / "configs").glob("*.json") if not (A / f"{p.stem}.h5").exists())
    failed = bool(missing)
    for name in missing:
        print(f"FAIL       {name}: no baseline output")
    for name in cases:
        verdict, notes, worst, info = compare_case(name, A, B, a.tol)
        print(f"{verdict:10s} {name}: max rel diff {worst:.3e}  ({info})")
        for n in notes[:20]:
            print(n)
        failed |= verdict == "FAIL"
    print("\nGATE", "FAILED" if failed else "PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
