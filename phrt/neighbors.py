"""
Fixed-radius neighbor enumeration (the Vietoris-Rips edge set) three ways, all exact:
k-d tree (scipy, CPU), chunked brute force (numpy), chunked brute force on the GPU (torch).
The GPU brute force is the non-RT accelerator baseline an RT version would have to beat.
Edges are returned as (i, j, d) with i < j and d <= r.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse


def edges_kdtree(X, r):
    t = cKDTree(X)
    P = t.query_pairs(r, output_type="ndarray")
    if len(P) == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    P = np.sort(P, axis=1)
    d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
    o = np.lexsort((P[:, 1], P[:, 0]))
    return P[o].astype(np.int64), d[o]


def edges_brute(X, r, chunk=1024):
    n = len(X)
    sq = (X * X).sum(1)
    lim = (r * (1 + 1e-9) + 1e-12) ** 2          # small margin; exact recheck below
    I, J = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d2 = sq[a:b, None] + sq[None, :] - 2 * X[a:b] @ X.T
        i, j = np.nonzero(d2 <= lim)
        keep = j > i + a
        I.append(i[keep] + a); J.append(j[keep])
    P = np.stack([np.concatenate(I), np.concatenate(J)], 1).astype(np.int64)
    d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
    ok = d <= r
    P, d = P[ok], d[ok]
    o = np.lexsort((P[:, 1], P[:, 0]))
    return P[o], d[o]


def edges_torch(X, r, device="cuda", chunk=1024):
    """GPU brute force (fp32 distances, boundary pairs rechecked in fp64 on the host)."""
    import torch
    # Center first (shrinks |x| and hence fp32 cancellation), use the direct-difference cdist (no
    # |x|^2 + |y|^2 - 2xy expansion), and keep a generous margin; the fp64 recheck below is exact.
    Xt = torch.as_tensor(X - X.mean(0), dtype=torch.float32, device=device)
    n = len(X)
    eps = 1e-3 * max(r, 1e-12) + 1e-6
    I, J = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = torch.cdist(Xt[a:b], Xt, compute_mode="donot_use_mm_for_euclid_dist")
        i, j = torch.nonzero(d <= r + eps, as_tuple=True)
        keep = j > i + a
        I.append((i[keep] + a).cpu()); J.append(j[keep].cpu())
    if device != "cpu":
        torch.cuda.synchronize()
    P = np.stack([torch.cat(I).numpy(), torch.cat(J).numpy()], 1).astype(np.int64)
    d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
    ok = d <= r
    P, d = P[ok], d[ok]
    o = np.lexsort((P[:, 1], P[:, 0]))
    return P[o], d[o]


METHODS = {"kdtree": edges_kdtree, "brute": edges_brute, "torch": edges_torch}


def timed(method, X, r, repeats=3, **kw):
    """(edges, dists, best seconds over repeats)."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = METHODS[method](X, r, **kw)
        best = min(best, time.perf_counter() - t0)
    return out[0], out[1], best


def radius_for_degree(X, degree, sample=2000, seed=0):
    """Radius whose mean neighbor count (excluding self) is about `degree` (bisection on a sample)."""
    rng = np.random.default_rng(seed)
    t = cKDTree(X)
    S = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    kd = t.query(S, k=min(degree + 1, len(X)))[0][:, -1]
    lo, hi = 0.0, float(np.max(kd)) * 4 + 1e-12
    for _ in range(40):
        mid = (lo + hi) / 2
        m = np.mean(t.query_ball_point(S, mid, return_length=True)) - 1
        lo, hi = (mid, hi) if m < degree else (lo, mid)
    return hi


def sparse_distance_matrix(n, P, d):
    """Symmetric scipy COO matrix for ripser(distance_matrix=True). Zero-length edges are nudged so
    they are not dropped as structural zeros."""
    d = np.where(d > 0, d, 1e-12)
    return sparse.coo_matrix((np.concatenate([d, d]), (np.concatenate([P[:, 0], P[:, 1]]),
                                                       np.concatenate([P[:, 1], P[:, 0]]))), shape=(n, n))
