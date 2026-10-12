"""
Amdahl test for RT-accelerated persistent homology.

A sparse (thresholded) Vietoris-Rips computation has two phases:
  1. neighbor search: enumerate edges with d <= r       <- the only part an RT radius query does
  2. filtration + matrix reduction (Ripser)             <- unchanged by RT
profile_rips() times phase 1 with each exact method (k-d tree, GPU brute force) and phase 2 with
Ripser on the resulting sparse distance matrix, and reports the best-case speedup if phase 1
became free: amdahl_ceiling = t_total / t_reduce, using the FASTEST non-RT neighbor method.

Notes:
  * Ripser's coboundary enumeration inside the reduction also walks neighbor lists; that work
    stays on the CPU/SMs whatever finds the edges, so it is counted in phase 2.
  * ripser.py adds Python-side conversion overhead to phase 2, which slightly favors RT here.
  * Ripser++ (GPU) is timed too when `ripserplusplus` is importable: it shrinks phase 2, which
    raises the ceiling, so the test also reports the ceiling against it.
"""
import time
import numpy as np
from ripser import ripser

from . import neighbors as N


def ripser_sparse(n, P, d, maxdim):
    t0 = time.perf_counter()
    res = ripser(N.sparse_distance_matrix(n, P, d), distance_matrix=True, maxdim=maxdim)
    return res, time.perf_counter() - t0


def ripserpp_available():
    try:
        import ripserplusplus  # noqa: F401
        return True
    except Exception:
        return False


def ripserpp_time(X, r, maxdim):
    import ripserplusplus as rpp
    t0 = time.perf_counter()
    rpp.run(f"--format point-cloud --dim {maxdim} --threshold {r}", X.astype(np.float32))
    return time.perf_counter() - t0


def profile_rips(X, r, maxdim=1, methods=("kdtree", "torch"), device="cuda", repeats=3, with_ripserpp=True):
    n, D = X.shape
    out = {"n": n, "D": D, "r": float(r), "maxdim": maxdim}
    edges = None
    for m in methods:
        kw = {"device": device} if m == "torch" else {}
        try:
            P, d, t = N.timed(m, X, r, repeats=repeats, **kw)
        except Exception as e:                  # e.g. no GPU
            out[f"t_nbr_{m}"] = None
            out[f"err_{m}"] = repr(e)[:200]
            continue
        out[f"t_nbr_{m}"] = t
        if edges is None:
            edges = (P, d)
        else:
            if not (len(P) == len(edges[0]) and np.array_equal(P, edges[0])):
                raise AssertionError(f"{m} found {len(P):,} edges, {methods[0]} found {len(edges[0]):,} (r={r:.6g})")
    P, d = edges
    out["edges"] = int(len(P))
    out["avg_degree"] = 2 * len(P) / n
    res, t_red = ripser_sparse(n, P, d, maxdim)
    for _ in range(repeats - 1):
        t_red = min(t_red, ripser_sparse(n, P, d, maxdim)[1])
    out["t_reduce"] = t_red
    out["bars"] = [int(len(dg)) for dg in res["dgms"]]
    ts = [v for k, v in out.items() if k.startswith("t_nbr_") and v is not None]
    best = min(ts)
    out["t_nbr_best"] = best
    out["nbr_fraction"] = best / (best + t_red)
    out["amdahl_ceiling"] = (best + t_red) / t_red
    if with_ripserpp and ripserpp_available():
        try:
            tpp = ripserpp_time(X, r, maxdim)
            out["t_ripserpp_total"] = tpp
        except Exception as e:
            out["err_ripserpp"] = repr(e)[:200]
    return out
