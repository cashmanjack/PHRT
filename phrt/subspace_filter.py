"""
Can an RT radius query on 3D slices of the data find the Vietoris-Rips edges in D > 3?

PCA-rotate the cloud and cut it into consecutive 3D slices (slice 0 = top 3 components).
For an edge (i, j) with ||x_i - x_j|| <= r, every slice distance is <= r too, because each
slice is a coordinate projection. So these candidate sets are LOSSLESS supersets of the edges:

  slice_k : pairs within r in each of slices 0..k-1 (intersection of k RT radius queries)
  pyth_k  : pairs with sum_{s<k} ||x_i^s - x_j^s||^2 <= r^2 (the best any filter can do that
            only sees the first k slices; slice_k contains it)

The question for RT is the size of slice_1 (one RT query, what RT actually returns) relative
to the true edge count: the RT core enumerates slice_1, and everything beyond the true edges
is wasted hits that must then be verified in full dimension on the SMs. RAGRT's version of
this (diag_hitcount.py) kept 41-97% of points for exact search on 128D embeddings.
"""
import numpy as np


def pca_rotate(X):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt.T, s ** 2 / np.sum(s ** 2)


def slice_ranges(D, slice_dim=3):
    return [(a, min(a + slice_dim, D)) for a in range(0, D, slice_dim)]


def filter_counts(X, r, slice_dim=3, max_slices=None, chunk=512, device=None):
    """
    Exact counts over all i < j: true edges, |slice_k| and |pyth_k| for k = 1..S.
    Uses torch on `device` if given (GPU), else numpy. O(n^2 S) work: keep n <= ~30k.
    """
    Xr, var = pca_rotate(X)
    n, D = Xr.shape
    rng_ = slice_ranges(D, slice_dim)
    if max_slices:
        rng_ = rng_[:max_slices]
    S = len(rng_)
    r2 = r * r
    slice_c = np.zeros(S, dtype=np.int64)
    pyth_c = np.zeros(S, dtype=np.int64)
    true_c = 0
    if device:
        import torch
        T = torch.as_tensor(Xr, dtype=torch.float64, device=device)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            up = torch.arange(n, device=device)[None, :] > torch.arange(a, b, device=device)[:, None]
            allin = up.clone()
            cum = torch.zeros(b - a, n, dtype=torch.float64, device=device)
            for k, (lo, hi) in enumerate(rng_):
                ds = torch.cdist(T[a:b, lo:hi], T[:, lo:hi]) ** 2
                cum += ds
                allin &= ds <= r2
                slice_c[k] += int(allin.sum()); pyth_c[k] += int((up & (cum <= r2)).sum())
            full = torch.cdist(T[a:b], T) ** 2 if S < len(slice_ranges(D, slice_dim)) else cum
            true_c += int((up & (full <= r2)).sum())
    else:
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            up = np.arange(n)[None, :] > np.arange(a, b)[:, None]
            allin = up.copy()
            cum = np.zeros((b - a, n))
            for k, (lo, hi) in enumerate(rng_):
                diff = Xr[a:b, None, lo:hi] - Xr[None, :, lo:hi]
                ds = np.einsum("ijk,ijk->ij", diff, diff)
                cum += ds
                allin &= ds <= r2
                slice_c[k] += int(allin.sum()); pyth_c[k] += int((up & (cum <= r2)).sum())
            if S < len(slice_ranges(D, slice_dim)):      # full distance without an [m, n, D] temp
                sq = (Xr * Xr).sum(1)
                cum = np.maximum(sq[a:b, None] + sq[None, :] - 2 * Xr[a:b] @ Xr.T, 0)
            true_c += int((up & (cum <= r2)).sum())
    pairs = n * (n - 1) // 2
    return {"n": n, "D": D, "r": r, "slices": S, "slice_dim": slice_dim, "pairs": pairs,
            "true_edges": true_c, "slice_k": slice_c.tolist(), "pyth_k": pyth_c.tolist(),
            "rt_candidates_per_edge": float(slice_c[0] / max(true_c, 1)),
            "rt_candidate_fraction_of_pairs": float(slice_c[0] / pairs),
            "var_top_slice": float(var[:slice_dim].sum())}
